"""Tests for the non-mutating training dataset readiness report."""
import csv
import hashlib
import importlib.util
from pathlib import Path


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'dataset_readiness.py'
    spec = importlib.util.spec_from_file_location('dataset_readiness', path)
    assert spec is not None and spec.loader is not None
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def fixture(tmp_path, imbalance=False):
    root = tmp_path / 'audio'
    root.mkdir()
    rows = []
    for split in ('train', 'val', 'test'):
        for group in ('a', 'b'):
            for label in ('real', 'fake'):
                for i in range(1 if not imbalance or label == 'fake' else 6):
                    name = f'{split}_{group}_{label}_{i}.wav'
                    content = name.encode()
                    (root / name).write_bytes(content)
                    rows.append({'path': name, 'label': label, 'split': split,
                                 'sha256': hashlib.sha256(content).hexdigest(),
                                 'group': f'{split}_{group}'})
    manifest = tmp_path / 'manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'label', 'split', 'sha256', 'group'])
        writer.writeheader()
        writer.writerows(rows)
    return root, manifest


def test_balanced_dataset_summary(tmp_path):
    root, manifest = fixture(tmp_path)
    result = module().audit_dataset(root, manifest)
    assert result['status'] == 'valid_manifest'
    assert result['recordings'] == 12
    assert result['groups'] == 6
    assert result['splits']['train'] == {'recordings': 4, 'real': 2, 'fake': 2,
                                         'fake_fraction': 0.5, 'groups': 2}
    assert result['warnings'] == []


def test_imbalanced_dataset_warns(tmp_path):
    root, manifest = fixture(tmp_path, imbalance=True)
    result = module().audit_dataset(root, manifest)
    assert len(result['warnings']) == 3
    assert all('class imbalance' in warning for warning in result['warnings'])


def test_cli_success(tmp_path, capsys):
    root, manifest = fixture(tmp_path)
    assert module().main([str(root), str(manifest)]) == 0
    assert 'fake_fraction' in capsys.readouterr().out


def test_rejects_tampered_audio(tmp_path, capsys):
    root, manifest = fixture(tmp_path)
    (root / 'train_a_real_0.wav').write_bytes(b'tampered')
    assert module().main([str(root), str(manifest)]) == 2
    assert 'hash mismatch' in capsys.readouterr().err.lower()
