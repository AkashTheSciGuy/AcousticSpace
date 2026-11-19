"""Candidate is opt-in, separately trained, and does not alter baseline behavior."""
import io
import wave

import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.main import app
from app.model_inference import model_status, predict_audio
from app.preprocessing import PreprocessedAudio

client = TestClient(app)


def recording() -> bytes:
    samples = np.rint(10000 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype('<i2')
    stream = io.BytesIO()
    with wave.open(stream, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(samples.tobytes())
    return stream.getvalue()


def candidate_artifact(path):
    # The same six-column feature signature required by the candidate inference service.
    rng = np.random.default_rng(42)
    values = rng.normal(size=(12, 6))
    truth = np.array([0] * 6 + [1] * 6)
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pipeline.fit(values, truth)
    joblib.dump({'pipeline': pipeline,
                 'features': ('rms_amplitude', 'zero_crossing_rate', 'spectral_centroid_hz',
                              'mean_spectral_flatness', 'rms_coefficient_of_variation',
                              'mean_positive_spectral_flux'),
                 'labels': ('real', 'fake')}, path)


def test_candidate_unavailable_does_not_affect_baseline(tmp_path, monkeypatch):
    monkeypatch.setenv('ACOUSTICSPACE_BASELINE_MODEL', str(tmp_path / 'baseline_missing.joblib'))
    monkeypatch.setenv('ACOUSTICSPACE_ACOUSTIC_MODEL', str(tmp_path / 'candidate_missing.joblib'))
    assert model_status('acoustic')['available'] is False
    assert model_status()['variant'] == 'baseline'
    assert client.get('/api/model/status?variant=acoustic').json()['available'] is False
    response = client.post('/api/model/predict?variant=acoustic', files={'file': ('clip.wav', recording())})
    assert response.status_code == 503


def test_candidate_prediction_with_valid_local_artifact(tmp_path, monkeypatch):
    path = tmp_path / 'acoustic.joblib'
    candidate_artifact(path)
    monkeypatch.setenv('ACOUSTICSPACE_ACOUSTIC_MODEL', str(path))
    assert client.get('/api/model/status?variant=acoustic').json()['available'] is True
    response = client.post('/api/model/predict?variant=acoustic', files={'file': ('clip.wav', recording())})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['variant'] == 'acoustic'
    assert result['model_type'] == 'experimental_speech_acoustic_candidate'
    assert result['fake_probability'] + result['real_probability'] == 1.0 or abs(result['fake_probability'] + result['real_probability'] - 1.0) < 1e-5
    assert 'Not RIR-based' in result['disclaimer']


def test_invalid_variant_rejected_by_api():
    assert client.get('/api/model/status?variant=unsupported').status_code == 422


def test_three_feature_artifact_cannot_masquerade_as_candidate(tmp_path, monkeypatch):
    path = tmp_path / 'wrong.joblib'
    joblib.dump({'features': ('rms_amplitude', 'zero_crossing_rate', 'spectral_centroid_hz'),
                 'labels': ('real', 'fake')}, path)
    monkeypatch.setenv('ACOUSTICSPACE_ACOUSTIC_MODEL', str(path))
    assert client.get('/api/model/status?variant=acoustic').json()['available'] is False
