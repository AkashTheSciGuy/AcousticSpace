"""Build a deterministic manifest for labeled WAV/MP3 recordings.

Directory layout: DATA_ROOT/real/** and DATA_ROOT/fake/**.

For a speaker/source-independent split, provide --groups groups.csv with
columns path,group. Every unique clip must have a group. Example:
  path,group
  real/speaker_a/001.wav,speaker_a
  fake/speaker_a/002.wav,speaker_a

The group should identify the actual speaker or source, not the class or clip.
A group is assigned to exactly one of train/val/test, across both labels.

Without --groups, the legacy clip-level split is retained for compatibility,
but it is NOT evidence of speaker/source-independent evaluation.
"""

import argparse
import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path

EXTENSIONS = {'.wav', '.mp3'}
LABELS = ('real', 'fake')
SPLITS = ('train', 'val', 'test')


def _read_groups(root: Path, groups_file: Path) -> dict[str, str]:
    groups: dict[str, str] = {}
    with groups_file.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not {'path', 'group'}.issubset(reader.fieldnames or []):
            raise ValueError('Groups CSV must contain path and group columns.')
        for row in reader:
            relative = row['path'].strip().replace('\\', '/')
            group = row['group'].strip()
            path = (root / relative).resolve()
            if not relative or not group or not path.is_relative_to(root):
                raise ValueError('Groups CSV contains an invalid path or empty group.')
            key = path.relative_to(root).as_posix()
            if key in groups:
                raise ValueError(f'Duplicate group assignment for {key}.')
            groups[key] = group
    return groups


def _split_by_group(rows: list[dict[str, str]], seed: int) -> None:
    """Keep each source together and require both classes in all three splits.

    Randomized candidate search is deterministic for a given seed. It prioritizes
    feasible class coverage before attempting approximate split proportions.
    """
    members: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        members[row['group']].append(row)
    groups = list(members)
    for label in LABELS:
        if sum(any(row['label'] == label for row in items) for items in members.values()) < 3:
            raise ValueError(f'Need at least 3 independent groups containing {label} audio.')
    if len(groups) < 3:
        raise ValueError('Need at least 3 independent groups for three splits.')

    rng = random.Random(seed)
    for _ in range(1024):
        shuffled = groups.copy()
        rng.shuffle(shuffled)
        allocation: dict[str, str] = {}
        # First satisfy class coverage. A shared speaker containing both labels
        # can satisfy both requirements without leaking into another split.
        for split in ('val', 'test', 'train'):
            for label in LABELS:
                if any(key in allocation and allocation[key] == split and
                       any(row['label'] == label for row in members[key]) for key in groups):
                    continue
                candidates = [key for key in shuffled if key not in allocation and
                              any(row['label'] == label for row in members[key])]
                if not candidates:
                    break
                # Prefer a group containing both labels, when available.
                candidates.sort(key=lambda key: -len({r['label'] for r in members[key]}))
                allocation[candidates[0]] = split
            else:
                continue
            break
        else:
            totals = {label: sum(row['label'] == label for row in rows) for label in LABELS}
            used = {split: {label: 0 for label in LABELS} for split in SPLITS}
            for key, split in allocation.items():
                for row in members[key]:
                    used[split][row['label']] += 1
            # Allocate leftover groups according to class-level target deficits.
            for key in shuffled:
                if key in allocation:
                    continue
                def score(split: str) -> float:
                    targets = {'train': 0.70, 'val': 0.15, 'test': 0.15}
                    return sum((targets[split] * totals[label] - used[split][label]) /
                               max(totals[label], 1) for label in LABELS
                               if any(row['label'] == label for row in members[key]))
                selected = max(SPLITS, key=score)
                allocation[key] = selected
                for row in members[key]:
                    used[selected][row['label']] += 1
            for row in rows:
                row['split'] = allocation[row['group']]
            return
    raise ValueError('Unable to create speaker/source-independent train, val and test splits.')


def prepare_manifest(root: Path, output: Path, seed: int = 42,
                     groups_file: Path | None = None) -> dict[str, int]:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f'Dataset root does not exist: {root}')
    groups = _read_groups(root, groups_file.resolve()) if groups_file else None
    rows: list[dict[str, str]] = []
    seen: dict[str, str] = {}
    counts: dict[str, int] = {}
    for label in LABELS:
        folder = root / label
        if not folder.is_dir():
            raise ValueError(f'Missing class directory: {folder}')
        unique: list[dict[str, str]] = []
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f'Audio file escapes dataset root: {path}')
            relative = resolved.relative_to(root).as_posix()
            if groups is not None and relative not in groups:
                raise ValueError(f'No source group provided for: {relative}')
            digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
            if digest in seen:
                if seen[digest] != label:
                    raise ValueError('Identical audio files were labeled both real and fake.')
                continue
            seen[digest] = label
            item = {'path': relative, 'label': label, 'split': '', 'sha256': digest}
            if groups is not None:
                item['group'] = groups[relative]
            unique.append(item)
        if len(unique) < 3:
            raise ValueError(f'Need at least 3 unique {label} files to create three splits.')
        if groups is None:
            random.Random(seed + LABELS.index(label)).shuffle(unique)
            n = len(unique)
            n_train = max(1, int(n * .7))
            n_val = max(1, int(n * .15))
            if n_train + n_val >= n:
                n_train, n_val = n - 2, 1
            for i, row in enumerate(unique):
                row['split'] = 'train' if i < n_train else 'val' if i < n_train + n_val else 'test'
        rows.extend(unique)
        counts[label] = len(unique)

    if groups is not None:
        _split_by_group(rows, seed)
        known_paths = {row['path'] for row in rows}
        extra = set(groups) - known_paths
        if extra:
            raise ValueError(f'Groups CSV includes an unknown or duplicate-only recording: {sorted(extra)[0]}')
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ['path', 'label', 'split', 'sha256'] + (['group'] if groups is not None else [])
    with output.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', type=Path, default=Path('dataset_manifest.csv'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--groups', type=Path, help='CSV mapping each clip path to a speaker/source group')
    args = parser.parse_args()
    print(f'Created manifest: {prepare_manifest(args.dataset, args.output, args.seed, args.groups)}')
    if args.groups is None:
        print('CAUTION: clip-level splitting is not speaker/source independent.')
    else:
        print('Verified: each declared source group appears in only one split.')


if __name__ == '__main__':
    main()
