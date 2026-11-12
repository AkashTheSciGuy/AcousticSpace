"""Preflight tests use local fixture bytes, not fabricated detection scores."""

import csv
import hashlib

import pytest

from app.manifest_validation import validate_training_manifest


def make_manifest(tmp_path, *, leak=False, duplicate=False, bad_hash=False, no_groups=False):
    root = tmp_path / 'dataset'
    root.mkdir()
    rows = []
    for index, split in enumerate(('train', 'val', 'test')):
        for label in ('real', 'fake'):
            path = root / f'{split}_{label}.wav'
            data = f'{split}-{label}'.encode()
            path.write_bytes(data)
            rows.append({'path': path.name, 'label': label, 'split': split,
                         'sha256': hashlib.sha256(data).hexdigest(),
                         'group': 'shared' if leak else f'speaker-{index}'})
    if duplicate:
        rows[-1]['sha256'] = rows[0]['sha256']
    if bad_hash:
        rows[0]['sha256'] = '0' * 64
    path = tmp_path / 'manifest.csv'
    fields = ['path', 'label', 'split', 'sha256'] + ([] if no_groups else ['group'])
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in rows])
    return root, path


def test_valid_grouped_manifest(tmp_path):
    root, manifest = make_manifest(tmp_path)
    result = validate_training_manifest(root, manifest)
    assert result['recordings'] == 6
    assert result['groups'] == 3
    assert result['counts']['test']['fake'] == 1


def test_rejects_leakage(tmp_path):
    root, manifest = make_manifest(tmp_path, leak=True)
    with pytest.raises(ValueError, match='group leakage'):
        validate_training_manifest(root, manifest)


def test_rejects_duplicate_hash(tmp_path):
    root, manifest = make_manifest(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match='Duplicate audio hash'):
        validate_training_manifest(root, manifest)


def test_rejects_tampered_audio(tmp_path):
    root, manifest = make_manifest(tmp_path, bad_hash=True)
    with pytest.raises(ValueError, match='hash mismatch'):
        validate_training_manifest(root, manifest)


def test_rejects_clip_level_manifest(tmp_path):
    root, manifest = make_manifest(tmp_path, no_groups=True)
    with pytest.raises(ValueError, match='requires a manifest'):
        validate_training_manifest(root, manifest)
