"""Dataset splitting should be deterministic and reject duplicate leakage."""

import csv
import importlib.util
from pathlib import Path

import pytest


def load_script(name: str):
    path = Path(__file__).resolve().parents[1] / 'scripts' / name
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_splits_and_deduplicates(tmp_path: Path) -> None:
    prepare = load_script('prepare_dataset.py').prepare_manifest
    root = tmp_path / 'data'
    for label in ('real', 'fake'):
        (root / label).mkdir(parents=True)
        for i in range(10):
            (root / label / f'{i}.wav').write_bytes(f'{label}-{i}'.encode())
    # Same-content duplicate within a class is skipped.
    (root / 'real' / 'duplicate.wav').write_bytes(b'real-0')
    output = tmp_path / 'manifest.csv'
    assert prepare(root, output) == {'real': 10, 'fake': 10}
    with output.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 20
    assert len({row['sha256'] for row in rows}) == 20
    for label in ('real', 'fake'):
        assert {row['split'] for row in rows if row['label'] == label} == {'train', 'val', 'test'}
    with pytest.raises(FileExistsError):
        prepare(root, output)


def test_rejects_conflicting_labels(tmp_path: Path) -> None:
    prepare = load_script('prepare_dataset.py').prepare_manifest
    root = tmp_path / 'data'
    for label in ('real', 'fake'):
        (root / label).mkdir(parents=True)
        for i in range(3):
            (root / label / f'{i}.wav').write_bytes(f'{label}-{i}'.encode())
    (root / 'fake' / 'contradiction.wav').write_bytes(b'real-0')
    with pytest.raises(ValueError, match='both real and fake'):
        prepare(root, tmp_path / 'manifest.csv')


def test_too_few_examples_rejected(tmp_path: Path) -> None:
    prepare = load_script('prepare_dataset.py').prepare_manifest
    root = tmp_path / 'data'
    for label in ('real', 'fake'):
        (root / label).mkdir(parents=True)
        (root / label / 'one.wav').write_bytes(label.encode())
    with pytest.raises(ValueError, match='at least 3'):
        prepare(root, tmp_path / 'manifest.csv')
