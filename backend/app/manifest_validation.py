"""Validate an experimental train/val/test manifest before model training.

This verifies declared grouping and file identity; it cannot prove that source
metadata is truthful or that the dataset is representative of deployment audio.
"""

import csv
import hashlib
from collections import Counter
from pathlib import Path

SPLITS = ('train', 'val', 'test')
LABELS = ('real', 'fake')


def validate_training_manifest(dataset_root: Path, manifest: Path) -> dict:
    """Reject missing groups, leakage, and tampered/duplicate recordings.

    This deliberately requires a nonempty ``group`` column. Older clip-level
    manifests can still be created, but cannot be used for reported experiments.
    """
    root = dataset_root.resolve()
    if not root.is_dir():
        raise ValueError('Dataset root does not exist.')
    groups: dict[str, str] = {}
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    counts = Counter()
    total = 0

    with manifest.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        required = {'path', 'label', 'split', 'sha256', 'group'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Training requires a manifest with path,label,split,sha256,group columns.')
        for line_number, row in enumerate(reader, 2):
            relative = (row['path'] or '').strip()
            label = (row['label'] or '').strip()
            split = (row['split'] or '').strip()
            digest = (row['sha256'] or '').strip().lower()
            group = (row['group'] or '').strip()
            if not relative or not group or label not in LABELS or split not in SPLITS:
                raise ValueError(f'Invalid label, split, group or path on manifest line {line_number}.')
            if len(digest) != 64 or any(ch not in '0123456789abcdef' for ch in digest):
                raise ValueError(f'Invalid SHA-256 on manifest line {line_number}.')
            file_path = (root / relative).resolve()
            if not file_path.is_relative_to(root) or not file_path.is_file():
                raise ValueError(f'Missing or outside-root recording on line {line_number}.')
            path_key = file_path.relative_to(root).as_posix()
            if path_key in seen_paths:
                raise ValueError(f'Duplicate recording path on line {line_number}.')
            seen_paths.add(path_key)
            if digest in seen_hashes:
                raise ValueError(f'Duplicate audio hash on line {line_number}.')
            seen_hashes.add(digest)
            if group in groups and groups[group] != split:
                raise ValueError(f'Speaker/source group leakage across splits: {group}.')
            groups[group] = split
            hasher = hashlib.sha256()
            with file_path.open('rb') as audio:
                for block in iter(lambda: audio.read(1024 * 1024), b''):
                    hasher.update(block)
            if hasher.hexdigest() != digest:
                raise ValueError(f'Audio hash mismatch on line {line_number}.')
            counts[(split, label)] += 1
            total += 1

    for split in SPLITS:
        for label in LABELS:
            if not counts[(split, label)]:
                raise ValueError(f'Missing {label} recordings in {split} split.')
    return {'recordings': total, 'groups': len(groups),
            'counts': {split: {label: counts[(split, label)] for label in LABELS} for split in SPLITS}}
