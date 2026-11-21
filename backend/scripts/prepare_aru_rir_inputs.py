r"""Prepare a *controlled* measured-RIR + anechoic-speech experiment input root.

Windows CMD from AcousticSpace/backend:
    .venv\Scripts\python.exe scripts\prepare_aru_rir_inputs.py ^
      datasets\speech_sources\ARU_Speech_Corpus_v1_0.zip ^
      datasets\rir_sources\prepared_rirs_v1 ^
      --output datasets\rir_sources\aru_measured_inputs_v1

Produces the schema expected by scripts/build_rir_dataset.py. This does not
create genuine in-room recordings, a blind RIR estimator, or real/fake labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

RATE = 16_000
SPLITS = ("train", "val", "test")
SPEAKER_PATTERN = re.compile(r"^(ID\d+)_ARU_", re.IGNORECASE)
MAX_ZIP_MEMBER_BYTES = 64 * 1024 * 1024


def _read_csv(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"{path.name} must contain {', '.join(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    for row in rows:
        if any(not (row.get(field) or "").strip() for field in required):
            raise ValueError(f"Missing value in {path.name}")
    return rows


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _safe_source(root: Path, raw: str) -> Path:
    rel = Path(raw)
    if not raw or rel.is_absolute() or rel.drive or ".." in rel.parts:
        raise ValueError(f"Unsafe RIR path: {raw}")
    path = (root / rel).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.suffix.lower() != ".wav":
        raise ValueError(f"Missing or invalid RIR: {raw}")
    return path


def _copy_rirs(source_root: Path, output: Path) -> tuple[list[dict], dict[str, int]]:
    rows = _read_csv(source_root / "responses.csv", ("path", "rir_id", "room_id", "split"))
    rooms = {split: defaultdict(int) for split in SPLITS}
    room_splits: dict[str, str] = {}
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    (output / "rirs").mkdir()
    prepared = []
    for index, row in enumerate(rows):
        split, room, rir_id = (row[key].strip() for key in ("split", "room_id", "rir_id"))
        if split not in SPLITS or rir_id in seen_ids:
            raise ValueError("Invalid split or duplicate RIR ID")
        if room in room_splits and room_splits[room] != split:
            raise ValueError(f"Room leakage across splits: {room}")
        seen_ids.add(rir_id)
        room_splits[room] = split
        src = _safe_source(source_root, row["path"].strip())
        info = sf.info(str(src))
        if info.samplerate != RATE or info.channels != 1 or not 256 <= info.frames <= 2 * RATE:
            raise ValueError(f"Prepared RIR must be 16-kHz mono, <= 2 s: {src.name}")
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
        if digest in seen_hashes:
            raise ValueError(f"Duplicate prepared RIR audio: {src.name}")
        seen_hashes.add(digest)
        rel = f"rirs/{index:03d}_{src.name}"
        shutil.copy2(src, output / rel)
        prepared.append({"path": rel, "rir_id": rir_id, "room_id": room, "split": split})
        rooms[split][room] += 1
    for split, split_rooms in rooms.items():
        if len(split_rooms) < 2 or any(n < 2 for n in split_rooms.values()):
            raise ValueError(f"Split {split} needs >=2 rooms, each with >=2 RIRs")
    _write_csv(output / "responses.csv", prepared, ("path", "rir_id", "room_id", "split"))
    return prepared, {split: sum(rooms[split].values()) for split in SPLITS}


def _decode_speech(archive: zipfile.ZipFile, entry: zipfile.ZipInfo, seconds: float,
                   channel: int) -> tuple[np.ndarray, int, int, int, str]:
    if entry.file_size > MAX_ZIP_MEMBER_BYTES:
        raise ValueError("ZIP member exceeds the bounded per-file read limit")
    with archive.open(entry, "r") as member:
        content = member.read(MAX_ZIP_MEMBER_BYTES + 1)
    if len(content) > MAX_ZIP_MEMBER_BYTES:
        raise ValueError("ZIP member exceeded the safe size limit")
    source_digest = hashlib.sha256(content).hexdigest()
    buffer = io.BytesIO(content)
    info = sf.info(buffer)
    if (info.samplerate < 8000 or info.samplerate > 192000 or
            info.channels < channel or info.channels > 16 or
            info.frames < math.ceil(info.samplerate * seconds)):
        raise ValueError("WAV is too short or has unsupported sample rate/channels")
    buffer.seek(0)
    data, rate = sf.read(buffer, dtype="float64", always_2d=True)
    mono = data[:, channel - 1]
    if not np.isfinite(mono).all():
        raise ValueError("WAV contains non-finite values")
    # Choose a high-energy window to avoid leading silence, without stitching
    # unrelated parts of the recording. This remains actual anechoic audio.
    n = math.ceil(rate * seconds)
    hop = max(1, rate // 4)
    starts = list(range(0, mono.size - n + 1, hop))
    if not starts or starts[-1] != mono.size - n:
        starts.append(mono.size - n)
    cs = np.concatenate(([0.0], np.cumsum(mono * mono, dtype=np.float64)))
    levels = [(cs[s + n] - cs[s]) / n for s in starts]
    selected_start = starts[int(np.argmax(levels))]
    selected = mono[selected_start:][:n]
    from math import gcd
    divisor = gcd(rate, RATE)
    converted = resample_poly(selected, RATE // divisor, rate // divisor)
    converted = converted[:round(RATE * seconds)]
    if converted.size < round(RATE * seconds) or not np.isfinite(converted).all():
        raise ValueError("Resampling produced insufficient or invalid samples")
    if float(np.sqrt(np.mean(converted * converted))) < 1e-5:
        raise ValueError("Selected speech segment is effectively silent")
    max_amplitude = float(np.max(np.abs(converted)))
    if max_amplitude > 0.95:
        converted = converted * (0.95 / max_amplitude)
    return converted.astype(np.float32), rate, info.channels, selected_start, source_digest


def prepare(archive_path: Path, prepared_rir_root: Path, output: Path,
            *, seconds: float = 2.0, files_per_speaker: int = 2,
            speech_channel: int = 1) -> dict:
    if not 0.25 <= seconds <= 10 or not 1 <= files_per_speaker <= 8 or not 1 <= speech_channel <= 16:
        raise ValueError("Invalid preparation settings")
    archive_path = archive_path.resolve()
    prepared_rir_root = prepared_rir_root.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    if not archive_path.is_file() or not prepared_rir_root.is_dir():
        raise ValueError("ARU ZIP or prepared RIR folder does not exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        candidates: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
        for entry in archive.infolist():
            filename = Path(entry.filename).name
            matched = SPEAKER_PATTERN.match(filename)
            if (entry.is_dir() or not matched or not filename.lower().endswith(".wav")
                    or "standard speech" not in filename.lower()):
                continue
            if entry.flag_bits & 1:
                continue  # Skip encrypted ZIP entries.
            candidates[matched.group(1).upper()].append(entry)
        speakers = sorted(candidates)
        if len(speakers) < 3:
            raise ValueError("Need at least three distinct ARU speaker IDs")
        for speaker in speakers:
            candidates[speaker].sort(key=lambda entry: entry.filename)
        with tempfile.TemporaryDirectory(prefix="acousticspace_aru_stage_", dir=output.parent) as directory:
            stage = Path(directory)
            _, rir_counts = _copy_rirs(prepared_rir_root, stage)
            (stage / "speech").mkdir()
            speech_rows: list[dict] = []
            speech_provenance: list[dict] = []
            rate_counts = Counter()
            channel_counts = Counter()
            skipped = 0
            for index, speaker in enumerate(speakers):
                split = SPLITS[index % len(SPLITS)]
                used = 0
                for entry in candidates[speaker]:
                    if used >= files_per_speaker:
                        break
                    try:
                        audio, original_rate, original_channels, selected_start, source_digest = _decode_speech(
                            archive, entry, seconds, speech_channel)
                    except (ValueError, OSError, RuntimeError, sf.LibsndfileError, EOFError):
                        skipped += 1
                        continue
                    relative = f"speech/{speaker}_{used + 1:02d}.wav"
                    sf.write(str(stage / relative), audio, RATE, subtype="PCM_16")
                    with (stage / relative).open("rb") as prepared_audio:
                        prepared_digest = hashlib.file_digest(prepared_audio, "sha256").hexdigest()
                    speech_provenance.append({
                        "prepared_path": relative,
                        "prepared_sha256": prepared_digest,
                        "source_archive_member": entry.filename,
                        "source_member_sha256": source_digest,
                        "source_sample_rate": original_rate,
                        "source_channels": original_channels,
                        "selected_channel_1_based": speech_channel,
                        "selected_start_frame": selected_start,
                        "selected_start_seconds": selected_start / original_rate,
                    })
                    rate_counts[str(original_rate)] += 1
                    channel_counts[str(original_channels)] += 1
                    speech_rows.append({"path": relative, "speaker_id": speaker, "split": split})
                    used += 1
                if used != files_per_speaker:
                    raise ValueError(f"Speaker {speaker} has only {used} usable recording(s); "
                                     f"need {files_per_speaker}. Try --seconds 1.")
            _write_csv(stage / "speech.csv", speech_rows, ("path", "speaker_id", "split"))
            samples_by_split = {split: sum(row["split"] == split for row in speech_rows)
                                for split in SPLITS}
            summary = {
                "prepared_speech_files": len(speech_rows),
                "speakers": len(speakers),
                "speech_files_per_split": samples_by_split,
                "prepared_rirs_per_split": rir_counts,
                "source_sample_rates": dict(rate_counts),
                "source_channel_counts": dict(channel_counts),
                "output_sample_rate": RATE,
                "output_channels": 1,
                "clip_seconds": seconds,
                "speech_channel_selected": speech_channel,
                "unusable_candidates_skipped": skipped,
                "speech_provenance": speech_provenance,
                "warning": "Controlled mixtures with real source speech and measured RIRs are still "
                           "synthetically mixed. This does not demonstrate blind room inference "
                           "or real/fake deepfake detection. Physical room/speaker provenance "
                           "must be checked independently.",
            }
            (stage / "preparation_metadata.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8")
            if output.exists():
                raise FileExistsError(f"Output appeared during preparation: {output}")
            stage.rename(output)
    return {"output": str(output), **{key: value for key, value in summary.items()
                                      if key != "speech_provenance"}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aru_zip", type=Path, help="Full ARU Speech Corpus ZIP file")
    parser.add_argument("prepared_rirs", type=Path, help="Folder containing responses.csv")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds", default=2.0, type=float)
    parser.add_argument("--files-per-speaker", default=2, type=int)
    parser.add_argument("--speech-channel", default=1, type=int, help="1-based channel index")
    args = parser.parse_args()
    try:
        result = prepare(args.aru_zip, args.prepared_rirs, args.output,
                         seconds=args.seconds, files_per_speaker=args.files_per_speaker,
                         speech_channel=args.speech_channel)
    except (OSError, ValueError, zipfile.BadZipFile, FileExistsError) as exc:
        print(f"ARU preparation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())