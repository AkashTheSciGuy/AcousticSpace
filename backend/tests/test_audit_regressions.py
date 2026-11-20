"""Audit reproductions: malformed metadata and contradictory saved metrics."""
import json

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
import numpy as np

from app.main import app
from app.model_inference import EXPECTED_FEATURES
from app.model_evaluation import read_evaluation
from app.comparison_api import read_comparison
from test_model_evaluation import valid_report
from test_comparison_api import valid_report as valid_comparison


@pytest.mark.parametrize('field,value', [('features', None), ('labels', 1)])
def test_malformed_metadata_reports_unavailable(tmp_path, monkeypatch, field, value):
    path = tmp_path / 'bad.joblib'
    artifact = {'features': EXPECTED_FEATURES, 'labels': ('real', 'fake')}
    artifact[field] = value
    joblib.dump(artifact, path)
    monkeypatch.setenv('ACOUSTICSPACE_BASELINE_MODEL', str(path))
    response = TestClient(app).get('/api/model/status')
    assert response.status_code == 200
    assert response.json()['available'] is False


def test_wrong_pipeline_width_is_not_ready(tmp_path, monkeypatch):
    path = tmp_path / 'wrong.joblib'
    model = LogisticRegression().fit(np.array([[0, 1], [1, 0]]), [0, 1])
    joblib.dump({'features': EXPECTED_FEATURES, 'labels': ('real', 'fake'), 'pipeline': model}, path)
    monkeypatch.setenv('ACOUSTICSPACE_BASELINE_MODEL', str(path))
    assert TestClient(app).get('/api/model/status').json()['available'] is False


@pytest.mark.parametrize('score', ['accuracy', 'precision_fake', 'recall_fake', 'f1_fake'])
def test_contradictory_evaluation_is_rejected(tmp_path, score):
    path = tmp_path / 'model.joblib'
    data = valid_report()
    data['metrics']['test'][score] = 0.123
    path.with_suffix('.json').write_text(json.dumps(data))
    assert read_evaluation(path) is None


def test_contradictory_comparison_is_rejected(tmp_path):
    path = tmp_path / 'comparison.json'
    data = valid_comparison()
    data['models']['baseline_three_features']['metrics']['test']['accuracy'] = 1.0
    path.write_text(json.dumps(data))
    assert read_comparison(path) is None


def test_all_api_routes_with_synthetic_model_fixtures(tmp_path, monkeypatch):
    """Exercise every application endpoint; these are NOT research models."""
    from scripts.create_e2e_fixtures import create
    create(tmp_path)
    monkeypatch.setenv('ACOUSTICSPACE_BASELINE_MODEL', str(tmp_path / 'baseline.joblib'))
    monkeypatch.setenv('ACOUSTICSPACE_ACOUSTIC_MODEL', str(tmp_path / 'acoustic.joblib'))
    monkeypatch.setenv('ACOUSTICSPACE_COMPARISON_REPORT', str(tmp_path / 'comparison.json'))
    client = TestClient(app)
    for endpoint in ['/', '/health', '/docs', '/redoc', '/openapi.json', '/api/model/comparison']:
        assert client.get(endpoint).status_code == 200, endpoint
    content = (tmp_path / 'tone.wav').read_bytes()
    for endpoint in ['/api/audio/validate', '/api/audio/inspect']:
        response = client.post(endpoint, files={'file': ('tone.wav', content)})
        assert response.status_code == 200, response.text
    for variant in ['baseline', 'acoustic']:
        status = client.get('/api/model/status', params={'variant': variant})
        assert status.json()['available'] is True
        response = client.post('/api/model/predict', params={'variant': variant},
                               files={'file': ('tone.wav', content)})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result['model_version'] == status.json()['model_version']
        assert len(result['model_version']) == 71
        assert result['processing_time_ms'] >= 0
        assert result['fake_probability'] + result['real_probability'] == pytest.approx(1.0)
        assert 'Not RIR-based' in result['disclaimer']
    response = client.post('/api/rir/analyze', files={'file': ('rir.wav', (tmp_path / 'rir.wav').read_bytes())})
    assert response.status_code == 200, response.text
    assert response.json()['metrics']['rt60_t20_seconds'] == pytest.approx(0.5, abs=0.05)


def test_release_preflight_runs_both_models_on_supplied_audio(tmp_path, monkeypatch):
    from scripts.create_e2e_fixtures import create
    from scripts.verify_release import check_release
    create(tmp_path)
    monkeypatch.setenv('ACOUSTICSPACE_BASELINE_MODEL', str(tmp_path / 'baseline.joblib'))
    monkeypatch.setenv('ACOUSTICSPACE_ACOUSTIC_MODEL', str(tmp_path / 'acoustic.joblib'))
    monkeypatch.setenv('ACOUSTICSPACE_COMPARISON_REPORT', str(tmp_path / 'comparison.json'))
    (tmp_path / 'baseline.json').write_text(json.dumps(valid_report()))
    result = check_release(tmp_path / 'tone.wav')
    assert result['status'] == 'ready_for_local_demo'
    for variant in ('baseline', 'acoustic'):
        assert result['checks'][f'{variant}_inference_smoke'] is True
        assert result['inference_smoke'][variant]['model_version'].startswith('sha256:')


@pytest.mark.parametrize('variant', ['baseline', 'acoustic'])
def test_training_uses_same_normalization_as_manifest_validation(tmp_path, variant):
    """Whitespace accepted by preflight must not fail training or flip labels."""
    import csv
    import hashlib
    import soundfile as sf
    from scripts.train_baseline import train as train_baseline
    from scripts.train_acoustic_candidate import train as train_candidate
    rows = []
    pairs = [(split, label) for split in ('train', 'val', 'test') for label in ('real', 'fake')]
    for index, (split, label) in enumerate(pairs):
        path = tmp_path / f'{index}.wav'
        t = np.arange(4000) / 16000
        sf.write(path, 0.2 * np.sin(2 * np.pi * (200 + index * 100) * t), 16000)
        rows.append({'path': f' {path.name} ', 'label': f' {label} ', 'split': f' {split} ',
                     'group': split, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = tmp_path / 'manifest.csv'
    with manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / f'{variant}.joblib'
    (train_baseline if variant == 'baseline' else train_candidate)(tmp_path, manifest, output)
    artifact = joblib.load(output)
    assert artifact['pipeline'].classes_.tolist() == [0, 1]
    assert artifact['pipeline'].n_features_in_ == (3 if variant == 'baseline' else 6)
