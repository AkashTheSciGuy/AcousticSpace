"""Construct controlled same-room vs different-room RIR mixtures.

This is a *reference-known, synthetic* acoustics experiment, NOT detection
from arbitrary speech or a validated audio-deepfake dataset. A 'consistent'
label means speech and ambience RIRs were measured/simulated in the same
declared room; a 'mismatched' label means different declared rooms.

Required CSVs (paths relative to input_root):
  speech.csv: path,speaker_id,split
  responses.csv: path,rir_id,room_id,split
Both must have train/val/test groups; room and speaker IDs cannot cross
splits. Each room needs at least two different response recordings so a
'consistent' example never shares the exact same RIR for both sources.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from app.preprocessing import load_audio
from app.rir_analysis import analyze_rir

SPLITS = ("train", "val", "test")
SAMPLE_RATE = 16_000
SPEECH_FIELDS = ("path", "speaker_id", "split")
RIR_FIELDS = ("path", "rir_id", "room_id", "split")
OUTPUT_FIELDS = (
    "sample_id", "pair_id", "split", "label", "speaker_id",
    "speech_room_id", "ambience_room_id", "speech_rir",
    "ambience_rir", "mixture", "speech_stem", "ambience_stem",
    "dry_speech", "dry_ambience", "sha256_mix",
    "sha256_speech_rir", "sha256_ambience_rir", "snr_db",
)


@dataclass(frozen=True)
class SpeechSource:
    path: Path
    speaker_id: str
    split: str


@dataclass(frozen=True)
class RoomResponse:
    path: Path
    rir_id: str
    room_id: str
    split: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_file(root: Path, value: str, line: int) -> Path:
    raw = Path(value.strip())
    if not value.strip() or raw.is_absolute() or raw.drive:
        raise ValueError(f"Invalid relative asset path at line {line}.")
    resolved = (root / raw).resolve()
    if (not resolved.is_relative_to(root) or not resolved.is_file()
            or resolved.suffix.lower() not in (".wav", ".flac")):
        raise ValueError(f"Missing/unsafe WAV or FLAC input at line {line}.")
    return resolved


def _rows(root: Path, manifest: Path, required: tuple[str, ...]):
    seen_paths: set[Path] = set()
    seen_hashes: set[str] = set()
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"{manifest.name} requires columns: {','.join(required)}.")
        for line, row in enumerate(reader, 2):
            if any(not (row.get(field) or "").strip() for field in required):
                raise ValueError(f"Missing manifest value at line {line}.")
            if row["split"] not in SPLITS:
                raise ValueError(f"Unsupported split at line {line}.")
            path = _input_file(root, row["path"], line)
            digest = sha256_file(path)
            if path in seen_paths or digest in seen_hashes:
                raise ValueError(f"Duplicate asset or identical audio at line {line}.")
            seen_paths.add(path)
            seen_hashes.add(digest)
            yield row, path


def validate_source_manifests(root: Path, speech_csv: Path, rir_csv: Path):
    """Validate independent speaker and room split assignments before writing."""
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("Input root does not exist.")
    speech: dict[str, list[SpeechSource]] = {split: [] for split in SPLITS}
    rooms: dict[str, dict[str, list[RoomResponse]]] = {
        split: defaultdict(list) for split in SPLITS
    }
    speakers_seen: dict[str, str] = {}
    rooms_seen: dict[str, str] = {}
    response_ids: set[str] = set()
    for row, path in _rows(root, speech_csv, SPEECH_FIELDS):
        speaker, split = row["speaker_id"].strip(), row["split"]
        if speakers_seen.setdefault(speaker, split) != split:
            raise ValueError(f"Speaker leakage across splits: {speaker}.")
        speech[split].append(SpeechSource(path, speaker, split))
    for row, path in _rows(root, rir_csv, RIR_FIELDS):
        response, room, split = row["rir_id"].strip(), row["room_id"].strip(), row["split"]
        if response in response_ids:
            raise ValueError(f"Duplicate RIR ID: {response}.")
        response_ids.add(response)
        if rooms_seen.setdefault(room, split) != split:
            raise ValueError(f"Room leakage across splits: {room}.")
        rooms[split][room].append(RoomResponse(path, response, room, split))
    for split in SPLITS:
        if not speech[split]:
            raise ValueError(f"Missing speech inputs for {split}.")
        if len(rooms[split]) < 2:
            raise ValueError(f"At least two independent rooms required in {split}.")
        for room_id, responses in rooms[split].items():
            if len(responses) < 2:
                raise ValueError(f"Room {room_id} needs two distinct RIRs.")
    return speech, rooms


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))


def _loaded_speech(path: Path, seconds: float) -> np.ndarray:
    audio = load_audio(path)
    values = audio.samples[:round(SAMPLE_RATE * seconds)].astype(np.float64)
    if values.size < 1600 or _rms(values) <= 1e-6:
        raise ValueError(f"Speech source is too short or effectively silent: {path}")
    return values


def _loaded_rir(path: Path) -> np.ndarray:
    audio = load_audio(path)
    values = audio.samples.astype(np.float64)
    if not 256 <= values.size <= SAMPLE_RATE * 2:
        raise ValueError(f"RIR must be between 256 samples and 2 seconds: {path}")
    # Reuse the existing RIR analyzer to reject silent/invalid impulses.
    analyze_rir(values, SAMPLE_RATE)
    energy = float(np.linalg.norm(values))
    if energy < 1e-7:
        raise ValueError(f"RIR has insufficient energy: {path}")
    return values / energy


def _padded_convolution(source: np.ndarray, rir: np.ndarray, length: int) -> np.ndarray:
    convolved = fftconvolve(source, rir, mode="full")
    return np.pad(convolved, (0, length - convolved.size))


def _normalized(source: np.ndarray, target_rms: float) -> np.ndarray:
    level = _rms(source)
    if level < 1e-10:
        raise ValueError("Cannot synthesize from an effectively silent source.")
    return source * (target_rms / level)


def _save(path: Path, samples: np.ndarray):
    if not np.isfinite(samples).all() or np.max(np.abs(samples)) > 1.0:
        raise ValueError("Synthesized waveform is not finite or would clip.")
    sf.write(str(path), samples, SAMPLE_RATE, subtype="PCM_16")


def _stable_seed(seed: int, *tokens: str) -> int:
    digest = hashlib.sha256(
        ("|".join([str(seed), *tokens])).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little")


def build_dataset(input_root: Path, speech_csv: Path, rir_csv: Path, output: Path,
                  *, pairs_per_speech: int = 1, seconds: float = 3.0,
                  snr_db: float = 10.0, seed: int = 42) -> dict:
    """Build paired controls with identical dry speech/noise and output loudness.

    Does not overwrite an existing dataset. Writes to a fresh staging folder
    and only publishes the output after the manifest is complete.
    """
    if (not 1 <= pairs_per_speech <= 4 or not 0.25 <= seconds <= 10.0
            or not np.isfinite(snr_db) or not -10.0 <= snr_db <= 30.0
            or not isinstance(seed, int)):
        raise ValueError("Invalid generation settings.")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Dataset output already exists: {output}")
    speech, rooms = validate_source_manifests(input_root, speech_csv, rir_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with TemporaryDirectory(prefix="acousticspace_rir_stage_", dir=output.parent) as temp:
        stage = Path(temp)
        (stage / "mixtures").mkdir()
        (stage / "stems").mkdir()
        (stage / "references").mkdir()
        (stage / "rirs").mkdir()
        responses = {response.rir_id: response
                     for split_rooms in rooms.values()
                     for entries in split_rooms.values() for response in entries}
        # Opaque deterministic filenames are derived from hashes, not label names.
        rir_names = {}
        rir_signals = {}
        rir_hashes = {}
        for response_id, response in responses.items():
            values = _loaded_rir(response.path)
            opaque = hashlib.sha256(response_id.encode()).hexdigest()[:20]
            name = f"rirs/{opaque}.wav"
            _save(stage / name, values)
            rir_names[response_id] = name
            rir_hashes[response_id] = sha256_file(stage / name)
            rir_signals[response_id] = values

        entries_out = []
        for split in SPLITS:
            room_ids = sorted(rooms[split])
            for source_index, source in enumerate(speech[split]):
                dry_speech = _loaded_speech(source.path, seconds)
                for pair_index in range(pairs_per_speech):
                    primary_room = room_ids[(source_index + pair_index) % len(room_ids)]
                    other_room = room_ids[(source_index + pair_index + 1) % len(room_ids)]
                    primary_responses = sorted(rooms[split][primary_room], key=lambda r: r.rir_id)
                    other_responses = sorted(rooms[split][other_room], key=lambda r: r.rir_id)
                    source_rir = primary_responses[(source_index + pair_index) % len(primary_responses)]
                    same_room_rir = primary_responses[
                        (source_index + pair_index + 1) % len(primary_responses)
                    ]
                    different_rir = other_responses[(source_index + pair_index) % len(other_responses)]
                    pair_id = hashlib.sha256(
                        f"{seed}:{split}:{source.path}:{pair_index}".encode()
                    ).hexdigest()[:20]
                    rng = np.random.default_rng(_stable_seed(seed, pair_id))
                    dry_noise = rng.normal(size=dry_speech.size)
                    target_length = dry_speech.size + max(
                        len(rir_signals[x.rir_id])
                        for x in (source_rir, same_room_rir, different_rir)
                    ) - 1
                    speech_wet = _normalized(
                        _padded_convolution(
                            dry_speech, rir_signals[source_rir.rir_id], target_length
                        ), 0.055)
                    noise_target = 0.055 / (10 ** (snr_db / 20))
                    noise_same = _normalized(
                        _padded_convolution(
                            dry_noise, rir_signals[same_room_rir.rir_id], target_length
                        ), noise_target)
                    noise_different = _normalized(
                        _padded_convolution(
                            dry_noise, rir_signals[different_rir.rir_id], target_length
                        ), noise_target)
                    peak = max(float(np.max(np.abs(speech_wet + noise_same))),
                               float(np.max(np.abs(speech_wet + noise_different))))
                    shared_gain = min(1.0, 0.95 / max(peak, 1e-12))
                    dry_speech_name = f"references/{pair_id}_speech.wav"
                    dry_noise_name = f"references/{pair_id}_noise.wav"
                    # References are bounded to avoid clipping; any gain is
                    # absorbed into the estimated impulse-response amplitude.
                    _save(stage / dry_speech_name, _normalized(dry_speech, 0.055))
                    _save(stage / dry_noise_name, _normalized(dry_noise, 0.055))
                    for label, ambient, ambient_rir in (
                        ("consistent", noise_same, same_room_rir),
                        ("mismatched", noise_different, different_rir),
                    ):
                        sample_id = hashlib.sha256(
                            f"{pair_id}:{label}".encode()).hexdigest()[:20]
                        mixture = shared_gain * (speech_wet + ambient)
                        mix_name = f"mixtures/{sample_id}.wav"
                        speech_name = f"stems/{sample_id}_speech.wav"
                        ambience_name = f"stems/{sample_id}_ambience.wav"
                        _save(stage / mix_name, mixture)
                        _save(stage / speech_name, shared_gain * speech_wet)
                        _save(stage / ambience_name, shared_gain * ambient)
                        entries_out.append({
                            "sample_id": sample_id,
                            "pair_id": pair_id,
                            "split": split,
                            "label": label,
                            "speaker_id": source.speaker_id,
                            "speech_room_id": primary_room,
                            "ambience_room_id": ambient_rir.room_id,
                            "speech_rir": rir_names[source_rir.rir_id],
                            "ambience_rir": rir_names[ambient_rir.rir_id],
                            "mixture": mix_name,
                            "speech_stem": speech_name,
                            "ambience_stem": ambience_name,
                            "dry_speech": dry_speech_name,
                            "dry_ambience": dry_noise_name,
                            "sha256_mix": sha256_file(stage / mix_name),
                            "sha256_speech_rir": rir_hashes[source_rir.rir_id],
                            "sha256_ambience_rir": rir_hashes[ambient_rir.rir_id],
                            "snr_db": float(snr_db),
                        })
                        total += 1
        with (stage / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(entries_out)
        (stage / "metadata.json").write_text(json.dumps({
            "experiment": "controlled_known_rir_consistency_v1",
            "sample_rate": SAMPLE_RATE,
            "labels": ["consistent", "mismatched"],
            "pairs_per_speech": pairs_per_speech,
            "seed": seed,
            "snr_db": snr_db,
            "samples": total,
            "per_split": {split: sum(1 for x in entries_out if x["split"] == split)
                          for split in SPLITS},
            "warning": (
                "Labels identify synthetic same-room/different-room response assignment, "
                "NOT authentic versus deepfake speech. Stored stems and reference RIRs "
                "are oracle supervision; this is not blind room identification."
            ),
        }, indent=2), encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"Dataset output appeared during generation: {output}")
        stage.rename(output)
    return {"path": str(output), "samples": total, "per_split": {
        split: sum(1 for x in entries_out if x["split"] == split) for split in SPLITS
    }, "warning": "Controlled synthetic RIR consistency labels only; not deepfake evidence."}
