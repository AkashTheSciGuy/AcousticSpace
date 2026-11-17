"""Offline tests for identical-split acoustic feature ablation."""
import csv
import hashlib
import importlib.util
import io
import wave
from pathlib import Path

import numpy as np


def comparison_module():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'compare_acoustic_models.py'
    spec = importlib.util.spec_from_file_location('compare_acoustic_models', path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def make_data(tmp_path):
    root = tmp_path / 'audio'
    root.mkdir()
    rows = []
    for split_index, split in enumerate(('train', 'val', 'test')):
        for label_index, label in enumerate(('real', 'fake')):
            for i in range(3):
                t = np.arange(3200) / 8000
                hz = 220 + 140 * label_index + 15 * split_index + 4 * i
                wave_signal = np.rint(10000 * np.sin(2 * np.pi * hz * t)).astype('<i2')
                output = io.BytesIO()
                with wave.open(output, 'wb') as wav:
                    wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(8000)
                    wav.writeframes(wave_signal.tobytes())
                payload = output.getvalue()
                name = f'{split}_{label}_{i}.wav'
                (root / name).write_bytes(payload)
                rows.append({'path': name, 'label': label, 'split': split,
                             'sha256': hashlib.sha256(payload).hexdigest(),
                             'group': f'{split}_speaker_{i}'})
    manifest = tmp_path / 'manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'label', 'split', 'sha256', 'group'])
        writer.writeheader(); writer.writerows(rows)
    return root, manifest


def test_comparison_has_two_models_same_split_counts(tmp_path):
    module = comparison_module()
    root, manifest = make_data(tmp_path)
    result = module.compare(root, manifest)
    assert result['recordings'] == {'train': 6, 'val': 6, 'test': 6}
    assert list(result['models']) == ['baseline_three_features', 'speech_acoustic_six_features']
    for model in result['models'].values():
        assert model['metrics']['val']['samples'] == 6
        assert model['metrics']['test']['samples'] == 6
        assert 0 <= model['metrics']['test']['f1_fake'] <= 1
        assert sum(map(sum, model['metrics']['test']['confusion_matrix_real_fake'])) == 6


def test_compare_rejects_group_leakage(tmp_path):
    module = comparison_module()
    root, manifest = make_data(tmp_path)
    with manifest.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    rows[-1]['group'] = rows[0]['group']
    with manifest.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    import pytest
    with pytest.raises(ValueError, match='leakage'):
        module.compare(root, manifest)


def test_main_refuses_overwriting_existing_report(tmp_path, capsys):
    module = comparison_module()
    root, manifest = make_data(tmp_path)
    target = tmp_path / 'report.json'
    target.write_text('keep this')
    assert module.main([str(root), str(manifest), '--output', str(target)]) == 2
    assert target.read_text() == 'keep this'
    assert 'already exists' in capsys.readouterr().err
