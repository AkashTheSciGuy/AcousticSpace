"""End-to-end tests for the inspection endpoint using real decodable audio."""

import io
import wave

import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def real_wav() -> bytes:
    t = np.arange(8000, dtype=np.float64) / 8000
    samples = np.rint(12000 * np.sin(2 * np.pi * 440 * t)).astype('<i2')
    output = io.BytesIO()
    with wave.open(output, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(samples.tobytes())
    return output.getvalue()


def post(filename: str, content: bytes):
    return client.post('/api/audio/inspect', files={'file': (filename, content)})


def test_inspects_real_audio() -> None:
    response = post('clip.wav', real_wav())
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['status'] == 'valid'
    assert data['filename'] == 'clip.wav'
    assert data['features']['sample_rate'] == 16000
    assert abs(data['features']['duration_seconds'] - 1.0) < 0.01
    assert data['features']['rms_amplitude'] > 0.1
    assert 250 < data['features']['spectral_centroid_hz'] < 650
    assert 'No deepfake classification' in data['message']


def test_rejects_unsupported_file() -> None:
    assert post('text.txt', b'hello').status_code == 415


def test_rejects_invalid_audio_header() -> None:
    assert post('fake.wav', b'hello world').status_code == 400


def test_rejects_truncated_wav() -> None:
    assert post('broken.wav', b'RIFF\x04\x00\x00\x00WAVE').status_code == 422


def test_rejects_oversize() -> None:
    response = post('huge.wav', real_wav() + b'\x00' * (20 * 1024 * 1024))
    assert response.status_code == 413
