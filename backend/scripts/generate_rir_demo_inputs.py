"""Generate synthetic RIR experiment INPUTS for a pipeline smoke test.

The 'speech-like' excitations are synthetic harmonic signals, NOT voices.
No real speakers, measured rooms, spoofed voices, or model accuracy claims.
Never use this toy dataset as forensic validation.

From backend:
    .venv\Scripts\python.exe scripts\generate_rir_demo_inputs.py --output datasets\rir_demo_inputs
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.rir_experiment import SAMPLE_RATE  # noqa: E402

SPLITS = ("train", "val", "test")


def create_demo_inputs(output: Path):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Demo output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "speech").mkdir()
    (output / "rirs").mkdir()
    dry_rows = []
    rir_rows = []
    duration = 1.0
    time = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE
    for split_index, split in enumerate(SPLITS):
        for speaker_index in range(2):
            speaker = f"demo_speaker_{split_index}_{speaker_index}"
            rng = np.random.default_rng(100 + split_index * 10 + speaker_index)
            carrier = (
                0.23 * np.sin(2 * np.pi * (175 + 30 * speaker_index) * time)
                + 0.14 * np.sin(2 * np.pi * (360 + 20 * split_index) * time)
                + 0.04 * rng.normal(size=time.size)
            )
            envelope = (0.4 + 0.6 * np.sin(np.pi * 3 * time) ** 2) * (time > 0.04)
            track = np.clip(carrier * envelope, -0.9, 0.9)
            name = f"speech/{speaker}.wav"
            sf.write(str(output / name), track, SAMPLE_RATE, subtype="PCM_16")
            dry_rows.append({"path": name, "speaker_id": speaker, "split": split})
        for room_index in range(2):
            room = f"demo_room_{split_index}_{room_index}"
            for position in range(2):
                rate = 0.25 + room_index * 0.4 + split_index * .025 + position * .012
                sample_count = int(SAMPLE_RATE * 1.0)
                rt = np.arange(sample_count) / SAMPLE_RATE
                rng = np.random.default_rng(900 + 100 * split_index + 10 * room_index + position)
                signal = .1 * rng.normal(size=sample_count)
                signal *= np.exp(-np.log(1000) * rt / rate)
                signal[0] = 1.0
                signal += np.eye(1, sample_count, 320 + position * 120).ravel() * .06
                name = f"rirs/{room}_p{position}.wav"
                sf.write(str(output / name), signal, SAMPLE_RATE, subtype="PCM_16")
                rir_rows.append({"path": name, "rir_id": f"{room}_p{position}",
                                 "room_id": room, "split": split})
    for filename, rows, fields in (
        ("speech.csv", dry_rows, ("path", "speaker_id", "split")),
        ("responses.csv", rir_rows, ("path", "rir_id", "room_id", "split")),
    ):
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return {"output": str(output), "speech_files": len(dry_rows),
            "response_files": len(rir_rows),
            "warning": "Synthetic acoustic smoke-test inputs, NOT genuine dry speech or measured RIRs."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = create_demo_inputs(args.output)
    except (OSError, ValueError) as exc:
        print(f"Demo-input creation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
