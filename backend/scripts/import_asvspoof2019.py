"""Build a speaker-disjoint AcousticSpace manifest from official ASVspoof 2019 LA.

Download and extract the official LA.zip from Edinburgh DataShare first.
The generated CSV works with scripts/train_baseline.py (FLAC supported by librosa).

Run from backend/:
    python scripts/import_asvspoof2019.py "D:/Datasets/ASVspoof2019/LA" \
        --output dataset_manifest.csv --max-per-speaker-class 10

The official LA *evaluation* partition is intentionally not used here.
This is a preliminary research baseline, not a benchmark submission.
"""

import argparse
import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

OUTPUT_COLUMNS = ('path', 'label', 'split', 'sha256', 'group')
PROTOCOL_NAMES = {
    'train': 'ASVspoof2019.LA.cm.train.trn.txt',
    'dev': 'ASVspoof2019.LA.cm.dev.trl.txt',
}


def _find_protocol(root: Path, name: str) -> Path:
    candidates = [root / 'ASVspoof2019_LA_cm_protocols' / name,
                  root / 'ASVspoof2019_LA_protocols' / name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f'Missing ASVspoof protocol {name}; expected in ASVspoof2019_LA_cm_protocols/.')


def _read_partition(root: Path, partition: str) -> dict[str, list[tuple[str, Path, str]]]:
    protocol = _find_protocol(root, PROTOCOL_NAMES[partition])
    audio_dir = root / f'ASVspoof2019_LA_{partition}' / 'flac'
    if not audio_dir.is_dir():
        raise ValueError(f'Missing FLAC directory: {audio_dir}')
    speakers: dict[str, list[tuple[str, Path, str]]] = defaultdict(list)
    seen_ids: set[str] = set()
    with protocol.open(encoding='utf-8') as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            parts = raw_line.split()
            if len(parts) != 5:
                raise ValueError(f'{protocol.name}:{line_number}: expected 5 protocol fields.')
            speaker, utterance, unused, attack, original_label = parts
            if not speaker.startswith('LA_') or not utterance.startswith(('LA_T_', 'LA_D_')):
                raise ValueError(f'{protocol.name}:{line_number}: invalid speaker/utterance ID.')
            if (partition == 'train' and not utterance.startswith('LA_T_')) or (
                partition == 'dev' and not utterance.startswith('LA_D_')
            ):
                raise ValueError(f'{protocol.name}:{line_number}: wrong partition utterance ID.')
            if unused != '-' or original_label not in {'bonafide', 'spoof'}:
                raise ValueError(f'{protocol.name}:{line_number}: invalid protocol fields.')
            if original_label == 'bonafide' and attack != '-':
                raise ValueError(f'{protocol.name}:{line_number}: invalid bonafide attack ID.')
            if utterance in seen_ids:
                raise ValueError(f'{protocol.name}:{line_number}: repeated utterance ID.')
            seen_ids.add(utterance)
            file = audio_dir / f'{utterance}.flac'
            if not file.is_file():
                raise ValueError(f'Audio file referenced by protocol is missing: {file}')
            label = 'real' if original_label == 'bonafide' else 'fake'
            speakers[speaker].append((label, file, utterance))
    if not speakers:
        raise ValueError(f'No examples in {protocol.name}.')
    return speakers


def make_manifest(root: Path, output: Path, *, seed: int = 42,
                  max_per_speaker_class: int = 10) -> dict:
    """Write a bounded, reproducible manifest; leave source data unmodified."""
    if max_per_speaker_class < 1:
        raise ValueError('--max-per-speaker-class must be at least 1.')
    root = root.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    train = _read_partition(root, 'train')
    dev = _read_partition(root, 'dev')
    if set(train).intersection(dev):
        raise ValueError('Official train and dev speakers overlap; refusing to continue.')
    if len(dev) < 2:
        raise ValueError('At least two independent development speakers are required.')
    dev_speakers = sorted(dev)
    random.Random(seed).shuffle(dev_speakers)
    midpoint = len(dev_speakers) // 2
    val_speakers = set(dev_speakers[:midpoint])
    test_speakers = set(dev_speakers[midpoint:])
    if not val_speakers or not test_speakers:
        raise ValueError('Cannot form speaker-disjoint validation and test sets.')

    rows = []
    hash_to_label = {}
    split_counts = Counter()
    split_label_counts = Counter()
    for partition, speakers in (('train', train), ('dev', dev)):
        for speaker, clips in sorted(speakers.items()):
            split = 'train' if partition == 'train' else ('val' if speaker in val_speakers else 'test')
            for label in ('real', 'fake'):
                candidates = sorted((clip for clip in clips if clip[0] == label), key=lambda x: x[2])
                rng = random.Random(f'{seed}:{partition}:{speaker}:{label}')
                rng.shuffle(candidates)
                for _, file, _ in candidates[:max_per_speaker_class]:
                    digest = hashlib.sha256(file.read_bytes()).hexdigest()
                    if digest in hash_to_label:
                        raise ValueError(f'Duplicate audio across records: {file}')
                    hash_to_label[digest] = label
                    relative = file.relative_to(root).as_posix()
                    rows.append({'path': relative, 'label': label, 'split': split,
                                 'sha256': digest, 'group': speaker})
                    split_counts[split] += 1
                    split_label_counts[(split, label)] += 1
    for split in ('train', 'val', 'test'):
        for label in ('real', 'fake'):
            if not split_label_counts[(split, label)]:
                raise ValueError(f'No {label} samples in {split}; choose a more complete dataset.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return {'total': len(rows), 'train': split_counts['train'],
            'val': split_counts['val'], 'test': split_counts['test'],
            'speakers': len(train) + len(dev)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('la_root', type=Path, help='Extracted ASVspoof2019/LA directory')
    parser.add_argument('--output', type=Path, default=Path('dataset_manifest.csv'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-per-speaker-class', type=int, default=10)
    args = parser.parse_args()
    print(make_manifest(args.la_root, args.output, seed=args.seed,
                        max_per_speaker_class=args.max_per_speaker_class))
    print('Source protocol: train -> train; dev speakers -> val/test. Original evaluation set untouched.')


if __name__ == '__main__':
    main()
