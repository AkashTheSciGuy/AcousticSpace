"""Check that speaker/source-aware manifest creation prevents split leakage."""
import csv
import importlib.util
from pathlib import Path

import pytest


def load_prepare():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_dataset.py'
    spec = importlib.util.spec_from_file_location('prepare_dataset_grouped', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prepare_manifest


def make_data(tmp_path: Path, speakers: int = 6):
    root = tmp_path / 'data'
    rows = []
    for label in ('real', 'fake'):
        (root / label).mkdir(parents=True)
        for index in range(speakers):
            relative = f'{label}/{index}.wav'
            (root / relative).write_bytes(f'{label}-{index}'.encode())
            rows.append((relative, f'speaker_{index}'))
    groups = tmp_path / 'groups.csv'
    with groups.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['path', 'group'])
        writer.writerows(rows)
    return root, groups


def test_grouped_manifest_keeps_speakers_in_single_split(tmp_path):
    root, groups = make_data(tmp_path)
    prepare = load_prepare()
    assert prepare(root, tmp_path / 'a.csv', groups_file=groups) == {'real': 6, 'fake': 6}
    prepare(root, tmp_path / 'b.csv', groups_file=groups)
    assert (tmp_path / 'a.csv').read_bytes() == (tmp_path / 'b.csv').read_bytes()
    with (tmp_path / 'a.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    for group in {row['group'] for row in rows}:
        assert len({row['split'] for row in rows if row['group'] == group}) == 1
    for label in ('real', 'fake'):
        assert {row['split'] for row in rows if row['label'] == label} == {'train', 'val', 'test'}


def test_grouped_manifest_requires_all_mappings(tmp_path):
    root, groups = make_data(tmp_path)
    content = groups.read_text(encoding='utf-8').replace('real/0.wav,speaker_0\n', '')
    groups.write_text(content, encoding='utf-8')
    with pytest.raises(ValueError, match='No source group'):
        load_prepare()(root, tmp_path / 'out.csv', groups_file=groups)
    assert not (tmp_path / 'out.csv').exists()


def test_grouped_manifest_rejects_insufficient_independent_groups(tmp_path):
    root, groups = make_data(tmp_path)
    content = groups.read_text(encoding='utf-8')
    for index in range(2, 6):
        content = content.replace(f'speaker_{index}', 'speaker_1')
    groups.write_text(content, encoding='utf-8')
    with pytest.raises(ValueError, match='at least 3 independent groups'):
        load_prepare()(root, tmp_path / 'out.csv', groups_file=groups)


def test_grouped_manifest_rejects_unknown_recording(tmp_path):
    root, groups = make_data(tmp_path)
    with groups.open('a', encoding='utf-8') as stream:
        stream.write('real/nonexistent.wav,unknown_speaker\n')
    with pytest.raises(ValueError, match='unknown'):
        load_prepare()(root, tmp_path / 'out.csv', groups_file=groups)


def test_legacy_manifest_preserved_without_group_file(tmp_path):
    root, _ = make_data(tmp_path)
    prepare = load_prepare()
    assert prepare(root, tmp_path / 'out.csv') == {'real': 6, 'fake': 6}
    with (tmp_path / 'out.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert set(rows[0]) == {'path', 'label', 'split', 'sha256'}
