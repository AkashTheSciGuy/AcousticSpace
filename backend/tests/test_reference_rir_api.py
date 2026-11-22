"""Real HTTP tests for reference-aided RIR comparison with bounded WAV inputs."""
from __future__ import annotations

import io

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from scipy.signal import fftconvolve

from app.main import app

client = TestClient(app)


def _wav(audio, sample_rate=16000):
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(audio), sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _inputs():
    rng = np.random.default_rng(927)
    dry_speech = rng.normal(size=8000) * 0.05
    dry_ambience = rng.normal(size=8000) * 0.05
    speech_ir = np.zeros(2000)
    speech_ir[0], speech_ir[250], speech_ir[700] = 0.8, 0.22, 0.08
    ambience_ir = np.zeros(2000)
    ambience_ir[0], ambience_ir[450], ambience_ir[1000] = 0.8, 0.09, 0.03
    return {
        "dry_speech": _wav(dry_speech),
        "wet_speech": _wav(fftconvolve(dry_speech, speech_ir)),
        "dry_ambience": _wav(dry_ambience),
        "wet_ambience": _wav(fftconvolve(dry_ambience, ambience_ir)),
    }


def _post(inputs):
    return client.post(
        "/api/rir/reference-consistency",
        files={key: (key + ".wav", value, "audio/wav") for key, value in inputs.items()},
    )


def test_reference_aided_endpoint_returns_finite_descriptors_not_verdict():
    response = _post(_inputs())
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "analyzed"
    assert payload["sample_rate_hz"] == 16000
    assert payload["prediction"] is None
    assert "dry speech" in " ".join(payload["limitations"])
    assert len(payload["features"]) == 7
    assert set(payload["feature_names"]) == set(payload["features"])
    assert all(np.isfinite(list(payload["features"].values())))
    assert all(value >= 0 for value in payload["features"].values())


def test_reference_consistency_is_near_zero_for_same_known_transfer():
    rng = np.random.default_rng(402)
    dry_speech = rng.normal(size=8000) * 0.05
    dry_noise = rng.normal(size=8000) * 0.05
    rir = np.zeros(2000)
    rir[0], rir[100], rir[600] = 0.8, 0.3, 0.1
    response = _post({
        "dry_speech": _wav(dry_speech),
        "wet_speech": _wav(fftconvolve(dry_speech, rir)),
        "dry_ambience": _wav(dry_noise),
        "wet_ambience": _wav(fftconvolve(dry_noise, rir)),
    })
    assert response.status_code == 200, response.text
    features = response.json()["features"]
    # PCM quantization and source-dependent regularization introduce small
    # transfer-estimation differences, even with identical true responses.
    assert features["late_energy_difference"] < 0.15


def test_api_rejects_missing_isolated_stem():
    files = _inputs()
    files.pop("wet_ambience")
    assert _post(files).status_code == 422


def test_api_rejects_non_wav_and_oversized_payloads():
    data = _inputs()
    data["dry_speech"] = b"not audio"
    response = _post(data)
    assert response.status_code == 400, response.text

    data = _inputs()
    data["dry_speech"] = b"RIFF0000WAVE" + b"0" * (4 * 1024 * 1024)
    assert _post(data).status_code == 413


def test_api_rejects_wrong_channel_count_and_sample_rate():
    data = _inputs()
    data["dry_speech"] = _wav(np.ones((1000, 2)) * 0.01)
    assert _post(data).status_code == 422

    data = _inputs()
    data["dry_speech"] = _wav(np.ones(1000) * 0.01, sample_rate=8000)
    assert _post(data).status_code == 422


def test_api_rejects_silent_and_inconsistent_lengths():
    data = _inputs()
    data["dry_speech"] = _wav(np.zeros(8000))
    assert _post(data).status_code == 422

    data = _inputs()
    data["wet_ambience"] = _wav(np.ones(500) * 0.01)
    assert _post(data).status_code == 422


def test_openapi_exposes_reference_aided_route():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/api/rir/reference-consistency" in response.json()["paths"]
