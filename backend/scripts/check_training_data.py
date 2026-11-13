"""Check an AcousticSpace grouped training manifest without training a model.

From backend/ on Windows:
  .venv\\Scripts\\python.exe scripts\\check_training_data.py DATASET_ROOT MANIFEST_CSV

This reports declared group separation and validates every referenced SHA-256.
It cannot verify that speaker identities, source labels, or files are authentic.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Support direct invocation from backend/scripts on Windows.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.manifest_validation import validate_training_manifest  # noqa: E402


def check_dataset(dataset_root: Path, manifest: Path) -> dict:
    """Return the validated class, split, and source-group counts."""
    return validate_training_manifest(dataset_root, manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_root', type=Path)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args(argv)
    try:
        result = check_dataset(args.dataset_root, args.manifest)
    except (ValueError, OSError, csv.Error) as exc:
        print(f'Dataset preflight failed: {exc}', file=sys.stderr)
        return 2
    print(json.dumps({'status': 'ready_for_experimental_training', **result}, indent=2))
    print('Note: source labels and real-world generalization have not been independently verified.', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
