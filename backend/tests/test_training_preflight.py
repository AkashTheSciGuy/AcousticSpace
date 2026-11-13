"""Tests for the read-only dataset preflight command."""
import csv
import hashlib
import importlib.util
from pathlib import Path


def preflight_module():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'check_training_data.py'
    spec = importlib.util.spec_from_file_location('check_training_data', path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_dataset(tmp_path):
    root = tmp_path / 'data'
    rows = []
    for split in ('train', 'val', 'test'):
        for label in ('real', 'fake'):
            path = root / f'{split}-{label}.wav'
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = f'{split}-{label}'.encode()
            path.write_bytes(payload)
            rows.append({'path': path.name, 'label': label, 'split': split,
                         'sha256': hashlib.sha256(payload).hexdigest(),
                         'group': split})
    manifest = tmp_path / 'manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'label', 'split', 'sha256', 'group'])
        writer.writeheader()
        writer.writerows(rows)
    return root, manifest


def test_preflight_reports_split_counts(tmp_path):
    root, manifest = create_dataset(tmp_path)
    result = preflight_module().check_dataset(root, manifest)
    assert result['recordings'] == 6
    assert result['groups'] == 3
    assert result['counts']['test'] == {'real': 1, 'fake': 1}


def test_preflight_cli_returns_zero_and_json(tmp_path, capsys):
    root, manifest = create_dataset(tmp_path)
    assert preflight_module().main([str(root), str(manifest)]) == 0
    assert 'ready_for_experimental_training' in capsys.readouterr().out


def test_preflight_rejects_modified_audio(tmp_path, capsys):
    root, manifest = create_dataset(tmp_path)
    (root / 'train-real.wav').write_bytes(b'changed')
    assert preflight_module().main([str(root), str(manifest)]) == 2
    assert 'hash mismatch' in capsys.readouterr().err.lower()


def test_preflight_rejects_cross_split_group(tmp_path, capsys):
    root, manifest = create_dataset(tmp_path)
    with manifest.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    rows[-1]['group'] = 'train'
    with manifest.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    assert preflight_module().main([str(root), str(manifest)]) == 2
    assert 'leakage' in capsys.readouterr().err.lower()
