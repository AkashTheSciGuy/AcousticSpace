"""Create deterministic train/validation/test manifest from labeled audio folders.

Expected layout:
  data_root/real/*.wav (or .mp3)
  data_root/fake/*.wav (or .mp3)

Run from backend/: python scripts/prepare_dataset.py PATH_TO_DATA --output dataset.csv
WARNING: Clip-level splitting does not guarantee speaker/source-independent splits.
"""

import argparse
import csv
import hashlib
import random
from pathlib import Path

EXTENSIONS = {'.wav', '.mp3'}
LABELS = ('real', 'fake')


def prepare_manifest(root: Path, output: Path, seed: int = 42) -> dict[str, int]:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f'Dataset root does not exist: {root}')
    rows: list[dict[str, str]] = []
    seen: dict[str, str] = {}
    counts: dict[str, int] = {}
    for label in LABELS:
        folder = root / label
        if not folder.is_dir():
            raise ValueError(f'Missing class directory: {folder}')
        unique: list[tuple[str, str]] = []
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f'Audio file escapes dataset root: {path}')
            sha256 = hashlib.sha256(resolved.read_bytes()).hexdigest()
            if sha256 in seen:
                if seen[sha256] != label:
                    raise ValueError('Identical audio files were labeled both real and fake.')
                continue  # avoid exact-duplicate leakage across splits
            seen[sha256] = label
            unique.append((resolved.relative_to(root).as_posix(), sha256))
        if len(unique) < 3:
            raise ValueError(f'Need at least 3 unique {label} files to create three splits.')
        random.Random(seed + LABELS.index(label)).shuffle(unique)
        n = len(unique)
        n_train = max(1, int(n * 0.7))
        n_val = max(1, int(n * 0.15))
        if n_train + n_val >= n:
            n_train = n - 2
            n_val = 1
        for i, (path, digest) in enumerate(unique):
            split = 'train' if i < n_train else 'val' if i < n_train + n_val else 'test'
            rows.append({'path': path, 'label': label, 'split': split, 'sha256': digest})
        counts[label] = n
    # Refuse overwriting a source recording or the existing manifest without explicit removal.
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'label', 'split', 'sha256'])
        writer.writeheader()
        writer.writerows(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path, help='Folder containing real/ and fake/ subdirectories')
    parser.add_argument('--output', type=Path, default=Path('dataset_manifest.csv'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(f'Created manifest: {prepare_manifest(args.dataset, args.output, args.seed)}')
    print('CAUTION: verify speaker and source independence before claiming generalization.')


if __name__ == '__main__':
    main()
