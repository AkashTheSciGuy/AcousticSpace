"""Offline fixture tests for the official ASVspoof LA protocol importer."""
import csv
import importlib.util
from pathlib import Path

import pytest


def script():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'import_asvspoof2019.py'
    spec = importlib.util.spec_from_file_location('asvspoof_import', path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def dataset(tmp_path, missing=False):
    root = tmp_path / 'LA'
    proto = root / 'ASVspoof2019_LA_cm_protocols'
    proto.mkdir(parents=True)
    for part, prefix, speakers in [('train', 'T', ['LA_0001', 'LA_0002']),
                                   ('dev', 'D', ['LA_0003', 'LA_0004', 'LA_0005', 'LA_0006'])]:
        folder = root / f'ASVspoof2019_LA_{part}' / 'flac'
        folder.mkdir(parents=True)
        lines = []
        for i, speaker in enumerate(speakers):
            for c, label in enumerate(('bonafide', 'spoof')):
                utterance = f'LA_{prefix}_{i:03d}{c:04d}'
                lines.append(f'{speaker} {utterance} - {"-" if c == 0 else "A01"} {label}\n')
                if not (missing and part == 'dev' and i == 0 and c == 0):
                    (folder / f'{utterance}.flac').write_bytes(f'{part}:{speaker}:{label}'.encode())
        name = f'ASVspoof2019.LA.cm.{"train.trn" if part == "train" else "dev.trl"}.txt'
        (proto / name).write_text(''.join(lines), encoding='utf-8')
    return root


def test_importer_is_deterministic_and_speaker_disjoint(tmp_path):
    module = script()
    root = dataset(tmp_path)
    first, second = tmp_path / 'first.csv', tmp_path / 'second.csv'
    stats = module.make_manifest(root, first)
    assert stats['total'] == 12
    module.make_manifest(root, second)
    assert first.read_bytes() == second.read_bytes()
    with first.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    groups = {}
    for row in rows:
        assert row['group'] not in groups or groups[row['group']] == row['split']
        groups[row['group']] = row['split']
    assert set(row['split'] for row in rows) == {'train', 'val', 'test'}
    assert {'real', 'fake'} == {row['label'] for row in rows if row['split'] == 'test'}
    with pytest.raises(FileExistsError):
        module.make_manifest(root, first)


def test_missing_recording_fails(tmp_path):
    root = dataset(tmp_path, missing=True)
    with pytest.raises(ValueError, match='missing'):
        script().make_manifest(root, tmp_path / 'out.csv')


def test_invalid_limit_fails(tmp_path):
    with pytest.raises(ValueError, match='at least 1'):
        script().make_manifest(tmp_path, tmp_path / 'out.csv', max_per_speaker_class=0)


def test_invalid_label_fails(tmp_path):
    root = dataset(tmp_path)
    p = root / 'ASVspoof2019_LA_cm_protocols' / 'ASVspoof2019.LA.cm.train.trn.txt'
    p.write_text(p.read_text().replace('bonafide', 'unknown', 1))
    with pytest.raises(ValueError, match='invalid protocol'):
        script().make_manifest(root, tmp_path / 'out.csv')
