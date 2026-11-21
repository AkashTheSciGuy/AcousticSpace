r"""Prepare the 24 eight-channel RVB2014 type-1 RIRs for AcousticSpace.

Select exactly one microphone channel, preserve original recordings, and
write room-disjoint train/val/test RIR manifests. This script does not train
or evaluate a deepfake detector. Room groups are inferred from known names
and require independent provenance review.

From AcousticSpace/backend in Windows CMD:
    .venv\Scripts\python.exe scripts\prepare_measured_rirs.py ^
      datasets\rir_sources\measured_rirs ^
      --output datasets\rir_sources\prepared_rirs_v1 --channel 1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

EXPECTED_ROOMS = {
    "smallroom1": "train",
    "mediumroom1": "train",
    "smallroom2": "val",
    "largeroom1": "val",
    "mediumroom2": "test",
    "largeroom2": "test",
}
FILENAME = re.compile(
    r"^RVB2014_type1_rir_"
    r"(?P<room>(?:small|medium|large)room[12])_"
    r"(?:near|far)_angl[ab]\.wav$",
    flags=re.IGNORECASE,
)
SAMPLE_RATE = 16_000
EXPECTED_CHANNELS = 8
EXPECTED_PER_ROOM = 4


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(source: Path, output: Path, *, channel: int = 1) -> dict:
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise ValueError(f"Source folder does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if not 1 <= channel <= EXPECTED_CHANNELS:
        raise ValueError("--channel must be between 1 and 8")

    files = sorted(source.glob("*.wav"))
    parsed = []
    rooms = Counter()
    for path in files:
        match = FILENAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected RIR filename: {path.name}")
        room = match["room"].lower()
        if room not in EXPECTED_ROOMS:
            raise ValueError(f"Unexpected room name: {room}")
        rooms[room] += 1
        parsed.append((path, room))

    if len(files) != 24 or set(rooms) != set(EXPECTED_ROOMS) or any(
        rooms[r] != EXPECTED_PER_ROOM for r in EXPECTED_ROOMS
    ):
        raise ValueError(
            f"Expected 24 files across six rooms, four per room; found {dict(rooms)}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="acousticspace_rir_prepare_", dir=output.parent))
    try:
        (stage / "rirs").mkdir()
        rows = []
        audits = []
        mono_hashes = set()
        for path, room in parsed:
            info = sf.info(path)
            if info.samplerate != SAMPLE_RATE or info.channels != EXPECTED_CHANNELS:
                raise ValueError(f"Unexpected WAV format: {path.name}")
            if info.frames != SAMPLE_RATE:
                raise ValueError(f"Expected a one-second RIR: {path.name}")

            values, actual_rate = sf.read(path, dtype="float64", always_2d=True)
            if actual_rate != SAMPLE_RATE or values.shape != (SAMPLE_RATE, EXPECTED_CHANNELS):
                raise ValueError(f"Unexpected decoded dimensions: {path.name}")
            mono = values[:, channel - 1].copy()
            if not np.isfinite(mono).all():
                raise ValueError(f"Non-finite audio: {path.name}")
            peak = float(np.max(np.abs(mono)))
            rms = float(np.sqrt(np.mean(mono * mono)))
            if rms <= 1e-7 or peak <= 1e-6:
                raise ValueError(f"Selected microphone channel is silent: {path.name}")
            if peak > 1.0:
                raise ValueError(f"Selected channel exceeds normal audio bounds: {path.name}")

            relative = Path("rirs") / (path.stem + f"_ch{channel:02d}.wav")
            destination = stage / relative
            sf.write(destination, mono, SAMPLE_RATE, format="WAV", subtype="FLOAT")
            check = sf.info(destination)
            if check.channels != 1 or check.samplerate != SAMPLE_RATE or check.frames != SAMPLE_RATE:
                raise ValueError(f"Output audio verification failed: {relative}")
            digest = sha256(destination)
            if digest in mono_hashes:
                raise ValueError(f"Duplicate mono response found: {path.name}")
            mono_hashes.add(digest)
            rows.append({
                "path": relative.as_posix(),
                "rir_id": path.stem + f"_ch{channel:02d}",
                "room_id": room,
                "split": EXPECTED_ROOMS[room],
            })
            audits.append({
                "source_filename": path.name,
                "source_sha256": sha256(path),
                "prepared_filename": relative.as_posix(),
                "prepared_sha256": digest,
                "selected_channel_1_based": channel,
                "source_subtype": info.subtype,
                "peak_amplitude": peak,
                "rms_amplitude": rms,
            })

        with (stage / "responses.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("path", "rir_id", "room_id", "split"))
            writer.writeheader()
            writer.writerows(rows)
        (stage / "preparation_metadata.json").write_text(
            json.dumps({
                "dataset": "RVB2014_type1_rir_measured_candidates",
                "sample_rate": SAMPLE_RATE,
                "channel_selection": channel,
                "files": len(rows),
                "rooms_per_split": dict(Counter(EXPECTED_ROOMS.values())),
                "rirs_per_split": dict(Counter(x["split"] for x in rows)),
                "room_split_assignments": EXPECTED_ROOMS,
                "warnings": [
                    "Original 8-channel files are preserved unchanged.",
                    "Only one microphone channel was selected; channels were not averaged.",
                    "Room IDs are inferred from filenames and require provenance review.",
                    "Shared recording conditions may limit independence even with room-disjoint splits.",
                    "No speech sources are included. Not a real/fake or deepfake dataset.",
                ],
                "files_audit": audits,
            }, indent=2), encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"Output directory appeared during preparation: {output}")
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)

    return {
        "output": str(output),
        "prepared_files": len(rows),
        "channels": 1,
        "sample_rate": SAMPLE_RATE,
        "duration_seconds": 1,
        "selected_microphone_channel": channel,
        "rirs_per_split": dict(Counter(x["split"] for x in rows)),
        "manifest": str(output / "responses.csv"),
        "note": "RIR preparation only; speech references and provenance review still required.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Folder with 24 original eight-channel WAVs")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--channel", type=int, default=1, help="1-based microphone channel (1-8)")
    args = parser.parse_args()
    try:
        result = prepare(args.source, args.output, channel=args.channel)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"RIR preparation failed: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())