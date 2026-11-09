"""Regression tests for optional experimental evaluation summaries."""

import json

from app.model_evaluation import read_evaluation


def valid_report():
    metrics = {'samples': 3, 'accuracy': 2 / 3, 'precision_fake': 0.5,
               'recall_fake': 1.0, 'f1_fake': 2 / 3,
               'confusion_matrix_real_fake': [[1, 1], [0, 1]]}
    return {'model_type': 'experimental_logistic_regression_audio_features',
            'metrics': {'val': dict(metrics), 'test': dict(metrics)}}


def test_missing_report_is_optional(tmp_path):
    assert read_evaluation(tmp_path / 'baseline_model.joblib') is None


def test_valid_training_report_is_summarized(tmp_path):
    path = tmp_path / 'baseline_model.joblib'
    path.with_suffix('.json').write_text(json.dumps(valid_report()), encoding='utf-8')
    result = read_evaluation(path)
    assert result is not None
    assert result['source'] == 'local_training_report'
    assert result['metrics']['test']['samples'] == 3
    assert result['metrics']['test']['accuracy'] == 2 / 3
    assert 'not been verified' in result['warning']


def test_invalid_metrics_are_not_exposed(tmp_path):
    path = tmp_path / 'baseline_model.joblib'
    report = valid_report()
    report['metrics']['test']['accuracy'] = 1.5
    path.with_suffix('.json').write_text(json.dumps(report), encoding='utf-8')
    assert read_evaluation(path) is None


def test_corrupt_or_inconsistent_reports_are_ignored(tmp_path):
    path = tmp_path / 'baseline_model.joblib'
    sidecar = path.with_suffix('.json')
    sidecar.write_text('{', encoding='utf-8')
    assert read_evaluation(path) is None
    report = valid_report()
    report['metrics']['val']['samples'] = 99
    sidecar.write_text(json.dumps(report), encoding='utf-8')
    assert read_evaluation(path) is None


def test_oversized_report_is_ignored(tmp_path):
    path = tmp_path / 'baseline_model.joblib'
    path.with_suffix('.json').write_text(' ' * (64 * 1024 + 1), encoding='utf-8')
    assert read_evaluation(path) is None
