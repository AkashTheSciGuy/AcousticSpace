"""HTTP integration checks using decodable WAV files."""

import io
import wave

import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def wav_bytes(samples: np.ndarray, rate: int = 16000) -> bytes:
    stream = io.BytesIO()
    pcm = np.clip(samples, -1, 1)
    with wave.open(stream, 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(np.rint(pcm * 32767).astype('<i2').tobytes())
    return stream.getvalue()


def post(name: str, content: bytes):
    return client.post('/api/rir/analyze', files={'file': (name, content)})


def test_analyzes_real_wav() -> None:
    t = np.arange(16000 * 2) / 16000
    rng = np.random.default_rng(123)
    sig = rng.normal(size=t.size) * np.exp(-np.log(1000) * t / 0.5)
    sig[0] = 2
    response = post('synthetic_rir.wav', wav_bytes((sig / 3).astype(np.float32)))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['status'] == 'analyzed'
    assert body['metrics']['sample_rate_hz'] == 16000
    assert body['metrics']['rt60_t20_seconds'] is not None
    assert 'does not determine' in body['message']


def test_rejects_mp3_extension() -> None:
    assert post('audio.mp3', b'ID3test').status_code == 415


def test_rejects_bogus_header() -> None:
    assert post('fake.wav', b'not wave').status_code == 400


def test_rejects_corrupt_wav() -> None:
    assert post('bad.wav', b'RIFF\x04\x00\x00\x00WAVE').status_code == 422


def test_rejects_oversized_upload() -> None:
    content = wav_bytes(np.zeros(1000, dtype=np.float32)) + b'\x00' * (20 * 1024 * 1024)
    assert post('large.wav', content).status_code == 413
