"""Verify baseline status and prediction API without pretending a model exists."""

import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from app.main import app
from app.model_inference import model_status, predict_audio
from app.preprocessing import PreprocessedAudio

client = TestClient(app)


def wav() -> bytes:
    data = np.rint(12000 * np.sin(2 * np.pi * 440 * np.arange(8000) / 8000)).astype('<i2')
    out = io.BytesIO()
    with wave.open(out, 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(8000)
        f.writeframes(data.tobytes())
    return out.getvalue()


def test_unavailable_model_reports_503(tmp_path, monkeypatch):
    monkeypatch.setenv('ACOUSTICSPACE_BASELINE_MODEL', str(tmp_path / 'none.joblib'))
    assert client.get('/api/model/status').json()['available'] is False
    response = client.post('/api/model/predict', files={'file': ('clip.wav', wav())})
    assert response.status_code == 503
    assert 'No trained model' in response.json()['detail']


def test_real_training_artifact_used_for_prediction(tmp_path, monkeypatch):
    path = tmp_path / 'model.joblib'
    x = np.array([[0.1, .01, 200], [0.2, .02, 300], [0.3, .09, 1500], [0.4, .10, 1800]])
    y = np.array([0, 0, 1, 1])
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pipeline.fit(x, y)
    joblib.dump({'pipeline': pipeline, 'features': ('rms_amplitude', 'zero_crossing_rate', 'spectral_centroid_hz'), 'labels': ('real', 'fake')}, path)
    monkeypatch.setenv('ACOUSTICSPACE_BASELINE_MODEL', str(path))
    assert model_status()['available'] is True
    t = np.arange(16000) / 16000
    samples = np.sin(t * 440 * 2 * np.pi).astype(np.float32) * 0.2
    audio = PreprocessedAudio(samples=samples, sample_rate=16000, duration_seconds=1.0, original_sample_rate=16000, original_channels=1, rms_amplitude=0.1)
    result = predict_audio(audio)
    assert result['status'] == 'predicted'
    assert result['label'] in {'real', 'fake'}
    assert result['fake_probability'] + result['real_probability'] == pytest.approx(1.0, abs=1e-5)
    assert 'not forensic evidence' in result['disclaimer']
    response = client.post('/api/model/predict', files={'file': ('clip.wav', wav())})
    assert response.status_code == 200, response.text
    assert response.json()['label'] in {'real', 'fake'}


def test_rejects_unsupported_when_model_loaded(tmp_path, monkeypatch):
    # Invalid model artifact must not be mistaken for a usable model.
    path = tmp_path / 'bad.joblib'
    joblib.dump({'features': []}, path)
    monkeypatch.setenv('ACOUSTICSPACE_BASELINE_MODEL', str(path))
    assert client.get('/api/model/status').json()['available'] is False
