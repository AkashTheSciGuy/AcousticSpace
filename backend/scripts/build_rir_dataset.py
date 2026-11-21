r"""Build a bounded synthetic same-room/different-room RIR benchmark.

Windows CMD (run from backend):
    .venv\Scripts\python.exe scripts\build_rir_dataset.py ^
      datasets\rir_demo_inputs datasets\rir_demo_inputs\speech.csv ^
      datasets\rir_demo_inputs\responses.csv --output datasets\rir_lab_v1

Inputs must have explicit speaker- and room-disjoint train/val/test splits.
No trained speech detector or ground-truth RIR extraction is produced.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.rir_experiment import build_dataset  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("speech_csv", type=Path)
    parser.add_argument("rir_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs-per-speech", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--snr-db", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        result = build_dataset(
            args.input_root, args.speech_csv, args.rir_csv, args.output,
            pairs_per_speech=args.pairs_per_speech,
            seconds=args.seconds, snr_db=args.snr_db, seed=args.seed,
        )
    except (OSError, ValueError, FileExistsError) as exc:
        print(f"Controlled RIR dataset failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
