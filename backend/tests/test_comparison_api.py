"""Tests ensure comparison reports cannot show fabricated or malformed metrics."""
import json

from app.comparison_api import read_comparison


def valid_report():
    entry = {
        'samples': 2, 'accuracy': 0.5, 'precision_fake': 0.5,
        'recall_fake': 1.0, 'f1_fake': 2 / 3,
        'confusion_matrix_real_fake': [[0, 1], [0, 1]],
    }
    return {
        'experiment': 'speech_acoustics_feature_ablation_v1',
        'recordings': {'train': 4, 'val': 2, 'test': 2},
        'models': {name: {'metrics': {'val': entry, 'test': entry}}
                   for name in ('baseline_three_features', 'speech_acoustic_six_features')},
    }


def test_accepts_valid_report(tmp_path):
    path = tmp_path / 'comparison.json'
    path.write_text(json.dumps(valid_report()))
    report = read_comparison(path)
    assert report is not None
    assert report['models']['speech_acoustic_six_features']['metrics']['test']['f1_fake'] == 2 / 3


def test_rejects_missing_report(tmp_path):
    assert read_comparison(tmp_path / 'missing.json') is None


def test_rejects_inconsistent_matrix(tmp_path):
    data = valid_report()
    data['models']['speech_acoustic_six_features']['metrics']['test']['confusion_matrix_real_fake'] = [[1, 1], [1, 1]]
    path = tmp_path / 'comparison.json'
    path.write_text(json.dumps(data))
    assert read_comparison(path) is None


def test_rejects_out_of_range_score(tmp_path):
    data = valid_report()
    data['models']['baseline_three_features']['metrics']['val']['accuracy'] = 1.2
    path = tmp_path / 'comparison.json'
    path.write_text(json.dumps(data))
    assert read_comparison(path) is None
