"""Load paired RIR-consistency experiments without feeding oracle RIRs to ML.

The CSV room IDs and actual RIR files are used only to validate pairing.
Model features come ONLY from four audio files:
dry_speech, speech_stem, dry_ambience, ambience_stem.
This is reference-aided; it cannot operate on a single arbitrary recording.
"""
import csv
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from app.rir_experiment import OUTPUT_FIELDS, SAMPLE_RATE, sha256_file
from app.rir_features import FEATURE_NAMES, reference_consistency_features

LABELS = {"consistent": 0, "mismatched": 1}
SPLITS = ("train", "val", "test")
MAX_FILES = 50_000
MAX_BYTES = 12 * SAMPLE_RATE * 8 + 4096


def _safe_audio_path(root: Path, relative: str) -> Path:
    path = Path(relative.strip())
    if not relative.strip() or path.is_absolute() or path.drive:
        raise ValueError("Dataset audio reference is not a safe relative path.")
    target = (root / path).resolve()
    if (not target.is_relative_to(root) or not target.is_file()
            or target.suffix.lower() != ".wav"):
        raise ValueError(f"Unsafe or missing dataset WAV reference: {relative}")
    if target.stat().st_size > MAX_BYTES:
        raise ValueError(f"Dataset WAV is too large: {relative}")
    return target


def _read_mono16(path: Path) -> np.ndarray:
    info = sf.info(path)
    if (info.channels != 1 or info.samplerate != SAMPLE_RATE
            or not 256 <= info.frames <= 12 * SAMPLE_RATE):
        raise ValueError(f"Dataset WAV requires bounded 16-kHz mono audio: {path.name}")
    audio, rate = sf.read(str(path), dtype="float64", always_2d=False)
    if rate != SAMPLE_RATE or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio values in {path.name}")
    return audio


def read_controlled_manifest(root: Path) -> dict[str, list[dict]]:
    """Verify pair integrity and all labels; never use rooms as model features."""
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("Controlled RIR dataset root does not exist.")
    grouped: dict[str, list[dict]] = {split: [] for split in SPLITS}
    pairs = defaultdict(list)
    ids, hashes = set(), set()
    with (root / "manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(OUTPUT_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("Controlled dataset manifest has missing columns.")
        for line, row in enumerate(reader, 2):
            if line > MAX_FILES + 1:
                raise ValueError("Controlled dataset exceeds the safe row limit.")
            if any(not (row.get(field) or "").strip() for field in OUTPUT_FIELDS):
                raise ValueError(f"Manifest row {line} has missing values.")
            if row["label"] not in LABELS or row["split"] not in SPLITS:
                raise ValueError(f"Invalid label or split on line {line}.")
            if row["sample_id"] in ids or row["sha256_mix"] in hashes:
                raise ValueError("Duplicate sample or identical mixture in manifest.")
            ids.add(row["sample_id"])
            hashes.add(row["sha256_mix"])
            if (row["speech_room_id"] == row["ambience_room_id"]) != (
                    row["label"] == "consistent"):
                raise ValueError("Consistency label disagrees with declared room assignment.")
            for column in ("mixture", "dry_speech", "dry_ambience",
                           "speech_stem", "ambience_stem"):
                _safe_audio_path(root, row[column])
            mix = _safe_audio_path(root, row["mixture"])
            if sha256_file(mix) != row["sha256_mix"]:
                raise ValueError(f"Mixture hash mismatch on line {line}.")
            pairs[row["pair_id"]].append(row)
            grouped[row["split"]].append(row)
    speakers, rooms = {}, {}
    for split, rows in grouped.items():
        for row in rows:
            speaker = row["speaker_id"]
            if speaker in speakers and speakers[speaker] != split:
                raise ValueError(f"Speaker leakage across splits: {speaker}.")
            speakers[speaker] = split
            for room in (row["speech_room_id"], row["ambience_room_id"]):
                if room in rooms and rooms[room] != split:
                    raise ValueError(f"Room leakage across splits: {room}.")
                rooms[room] = split
        if {r["label"] for r in rows} != set(LABELS):
            raise ValueError(f"Both consistency labels are required in {split}.")
    for pair_id, rows in pairs.items():
        if len(rows) != 2 or {r["label"] for r in rows} != set(LABELS):
            raise ValueError(f"Pair {pair_id} requires exactly one of each label.")
        a, b = rows
        if (any(a[key] != b[key] for key in
                ("split", "speaker_id", "speech_room_id",
                 "speech_rir", "dry_speech", "dry_ambience", "snr_db"))
                or a["ambience_rir"] == b["ambience_rir"]):
            raise ValueError(f"Paired control changed shared inputs: {pair_id}.")
        # Both wet speech stems must be byte-identical even though their names differ.
        pa = _safe_audio_path(root, a["speech_stem"])
        pb = _safe_audio_path(root, b["speech_stem"])
        if sha256_file(pa) != sha256_file(pb):
            raise ValueError(f"Paired control speech stem changed: {pair_id}.")
    return grouped


def extract_controlled_arrays(
    root: Path, *, include_test: bool = False
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build features from references/stems; leave test audio unread by default."""
    grouped = read_controlled_manifest(root)
    arrays = {}
    root = root.resolve()

    @lru_cache(maxsize=32)
    def cached_audio(path: Path):
        return _read_mono16(path)

    def audio(row, column):
        return cached_audio(_safe_audio_path(root, row[column]))

    selected_splits = SPLITS if include_test else ("train", "val")
    for split in selected_splits:
        rows = grouped[split]
        features, truth = [], []
        for row in rows:
            speech = audio(row, "speech_stem")
            ambient = audio(row, "ambience_stem")
            mix = audio(row, "mixture")
            if (speech.shape != ambient.shape or speech.shape != mix.shape
                    or np.max(np.abs(mix - speech - ambient)) > 3 / 32768):
                raise ValueError("Mixture/stem integrity check failed.")
            vector = reference_consistency_features(
                audio(row, "dry_speech"), speech,
                audio(row, "dry_ambience"), ambient,
            ).as_array()
            if vector.size != len(FEATURE_NAMES) or not np.isfinite(vector).all():
                raise ValueError("Invalid acoustic features.")
            features.append(vector)
            truth.append(LABELS[row["label"]])
        arrays[split] = (
            np.asarray(features, dtype=np.float64),
            np.asarray(truth, dtype=np.int64),
        )
    return arrays
