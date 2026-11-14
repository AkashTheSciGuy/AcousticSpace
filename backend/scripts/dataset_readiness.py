"""Audit class balance and declared group coverage before baseline training.

Windows CMD (run from backend/):
    .venv\\Scripts\\python.exe scripts\\dataset_readiness.py DATASET_ROOT MANIFEST_CSV

This checks file integrity using the existing training validator. Group values
are declarations supplied by the dataset, not independent identity verification.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.manifest_validation import validate_training_manifest  # noqa: E402

SPLITS = ('train', 'val', 'test')


def audit_dataset(dataset_root: Path, manifest: Path) -> dict:
    """Return read-only readiness statistics for a fully validated manifest."""
    verified = validate_training_manifest(dataset_root, manifest)
    groups = {split: set() for split in SPLITS}
    with manifest.open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            groups[row['split'].strip()].add(row['group'].strip())
    report = {}
    warnings = []
    for split in SPLITS:
        counts = verified['counts'][split]
        total = counts['real'] + counts['fake']
        fake_fraction = counts['fake'] / total
        report[split] = {
            'recordings': total,
            'real': counts['real'],
            'fake': counts['fake'],
            'fake_fraction': round(fake_fraction, 4),
            'groups': len(groups[split]),
        }
        if min(counts.values()) / total < 0.2:
            warnings.append(f'{split}: class imbalance (minority class below 20%).')
        if len(groups[split]) < 2:
            warnings.append(f'{split}: fewer than two independent declared source groups.')
    return {
        'status': 'valid_manifest',
        'recordings': verified['recordings'],
        'groups': verified['groups'],
        'splits': report,
        'warnings': warnings,
        'note': 'Valid grouping and file hashes do not establish real-world detection accuracy.',
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_root', type=Path)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit_dataset(args.dataset_root, args.manifest)
    except (OSError, ValueError, csv.Error) as exc:
        print(f'Dataset readiness failed: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
