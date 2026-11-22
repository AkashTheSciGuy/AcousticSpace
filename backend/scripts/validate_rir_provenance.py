r"""Validate manually curated, split-safe RIR source provenance before experiments.

This does not infer or verify physical room identities from filenames. The
operator must supply records from documented measurement metadata.

Windows CMD, from backend:
  .venv\Scripts\python.exe scripts\validate_rir_provenance.py ^
    datasets\prepared_rirs datasets\prepared_rirs\curated_rooms.csv ^
    --output datasets\prepared_rirs\provenance_audit_v1.json

The tool validates CSV integrity, SHA-256, WAV technical compatibility, room-
and session-disjoint splits, sufficient positional diversity, and references.
It CANNOT verify that human-provided physical location claims are truthful.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf

SPLITS = ("train", "val", "test")
REQUIRED = (
    "path", "rir_id", "site_id", "physical_room_id",
    "session_id", "mic_device", "mic_position", "source_position",
    "provenance_reference", "split", "sha256",
)
MAX_ROWS = 10_000
MAX_WAV_BYTES = 4 * 1024 * 1024
MAX_RIR_FRAMES = 32_000
SAMPLE_RATE = 16_000
PLACEHOLDERS = {"", "unknown", "na", "n/a", "none", "null", "tbd", "todo", "?"}


def _nonplaceholder(text: str, field: str, line: int) -> str:
    value = text.strip()
    if value.lower() in PLACEHOLDERS:
        raise ValueError(f"Line {line}: {field} requires documented non-placeholder metadata.")
    return value


def _safe_rir(root: Path, relative: str, line: int) -> Path:
    if not relative or "\\" in relative or ":" in relative:
        raise ValueError(f"Line {line}: use a safe POSIX-style relative WAV path.")
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts or raw.suffix.lower() != ".wav":
        raise ValueError(f"Line {line}: invalid relative WAV filename.")
    target = (root / raw).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError(f"Line {line}: missing or unsafe WAV reference.")
    if not 0 < target.stat().st_size <= MAX_WAV_BYTES:
        raise ValueError(f"Line {line}: WAV is empty or larger than 4 MiB.")
    return target


def _digest(file: Path) -> str:
    h = hashlib.sha256()
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(root: Path, csv_path: Path) -> dict:
    """Validate declared RIR identities; never label the physical provenance verified."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("RIR dataset root does not exist.")
    seen_rir_ids: set[str] = set()
    seen_paths: set[Path] = set()
    seen_hashes: set[str] = set()
    rooms: dict[tuple[str, str], str] = {}
    sessions: dict[tuple[str, str], str] = {}
    room_positions: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    room_counts: Counter[tuple[str, str]] = Counter()
    counts: Counter[str] = Counter()
    sites: set[str] = set()
    references: set[str] = set()
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not set(REQUIRED).issubset(reader.fieldnames or []):
            raise ValueError("Curated metadata is missing required columns: " +
                             ", ".join(sorted(set(REQUIRED) - set(reader.fieldnames or []))))
        for line, record in enumerate(reader, 2):
            if line > MAX_ROWS + 1:
                raise ValueError("Manifest exceeds 10000 source records.")
            values = {field: _nonplaceholder(record.get(field) or "", field, line)
                      for field in REQUIRED}
            split = values["split"].lower()
            if split not in SPLITS:
                raise ValueError(f"Line {line}: split must be train, val or test.")
            digest = values["sha256"].lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError(f"Line {line}: malformed SHA-256 digest.")
            path = _safe_rir(root, values["path"], line)
            if values["rir_id"] in seen_rir_ids or path in seen_paths:
                raise ValueError(f"Line {line}: duplicated RIR identifier or file path.")
            if digest in seen_hashes:
                raise ValueError(f"Line {line}: same WAV content appears more than once.")
            if _digest(path) != digest:
                raise ValueError(f"Line {line}: WAV SHA-256 does not match metadata.")
            info = sf.info(str(path))
            if (info.format != "WAV" or info.samplerate != SAMPLE_RATE
                    or info.channels != 1 or not 256 <= info.frames <= MAX_RIR_FRAMES):
                raise ValueError(f"Line {line}: expected 16-kHz mono WAV, 256-32000 frames.")
            room_key = values["site_id"], values["physical_room_id"]
            session_key = values["site_id"], values["session_id"]
            if room_key in rooms and rooms[room_key] != split:
                raise ValueError(f"Physical room leakage across splits: {room_key}.")
            if session_key in sessions and sessions[session_key] != split:
                raise ValueError(f"Recording-session leakage across splits: {session_key}.")
            rooms[room_key] = split
            sessions[session_key] = split
            room_counts[room_key] += 1
            room_positions[room_key].add((values["mic_position"], values["source_position"]))
            seen_rir_ids.add(values["rir_id"])
            seen_paths.add(path)
            seen_hashes.add(digest)
            sites.add(values["site_id"])
            references.add(values["provenance_reference"])
            counts[split] += 1
    if not seen_rir_ids:
        raise ValueError("Curated provenance manifest is empty.")
    for split in SPLITS:
        split_rooms = {room for room, assigned in rooms.items() if assigned == split}
        if len(split_rooms) < 2:
            raise ValueError(f"{split}: at least two distinct curated physical rooms are required.")
    for room, count in room_counts.items():
        if count < 2 or len(room_positions[room]) < 2:
            raise ValueError(f"Room {room} needs two WAVs from distinct mic/source positions.")
    return {
        "status": "metadata_checks_passed",
        "records_checked": len(seen_rir_ids),
        "technical_audio_hashes_verified": True,
        "physical_provenance_independently_verified": False,
        "source_metadata_self_declared": True,
        "recordings_by_split": {split: counts[split] for split in SPLITS},
        "rooms_by_split": {
            split: sum(assigned == split for assigned in rooms.values()) for split in SPLITS
        },
        "unique_sites": len(sites),
        "provenance_references": len(references),
        "warnings": [
            "Physical room, source and microphone annotations are manually provided; this tool cannot independently verify their authenticity.",
            "Source documents and measurement layouts must be checked against the official dataset/provider records before using these identities to claim independent room splits.",
            "Non-overlapping physical rooms and sessions do not by themselves ensure independent speakers, audio sources, microphones or acoustic conditions.",
            "No model was trained, evaluated, or tested by this metadata check.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("curated_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite existing audit: {args.output}")
        result = validate(args.dataset_root, args.curated_manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
    except (OSError, RuntimeError, ValueError, FileExistsError) as exc:
        print(f"RIR provenance audit failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
