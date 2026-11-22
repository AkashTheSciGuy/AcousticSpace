"""API handlers keep CPU-heavy audio operations outside the ASGI event loop.

The test traces asyncio.to_thread rather than using fragile wall-clock timing.
"""
import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import (
    audio_inspection,
    diagnostics_api,
    prediction_api,
    rir_api,
    segment_api,
)
from app.main import app
from scripts.final_acceptance import _rir_fixture, _speech_fixture


@pytest.mark.parametrize(
    "module,endpoint,analysis_func,is_model",
    [
        (audio_inspection, "/api/audio/inspect", "_analyze_recording", False),
        (diagnostics_api, "/api/audio/diagnostics", "_diagnose_recording", False),
        (rir_api, "/api/rir/analyze", "_measure_rir", False),
        (prediction_api, "/api/model/predict", "_predict_recording", True),
        (segment_api, "/api/model/segments", "_score_recording", True),
    ],
)
def test_audio_processing_uses_worker_threads(
    monkeypatch, module, endpoint, analysis_func, is_model,
):
    traced = []

    async def tracked_to_thread(func, *args, **kwargs):
        traced.append(func.__name__)
        return await asyncio.to_thread(func, *args, **kwargs)

    # Patching the module's asyncio reference does not patch the test runner.
    monkeypatch.setattr(module, "asyncio", SimpleNamespace(to_thread=tracked_to_thread))
    if is_model:
        def installed(_variant):
            return {"available": True}

        monkeypatch.setattr(module, "model_status", installed)
        # These are plumbing assertions; full trained-artifact HTTP inference
        # is covered by the separate prediction/segment API test modules.
        monkeypatch.setattr(
            module,
            analysis_func,
            lambda _path, _variant: {"status": "predicted" if module is prediction_api else "scored"},
        )

    audio = _rir_fixture() if module is rir_api else _speech_fixture()
    with TestClient(app) as client:
        response = client.post(
            endpoint,
            files={"file": ("fixture.wav", audio, "audio/wav")},
        )
    assert response.status_code == 200, response.text
    assert analysis_func in traced
    if is_model:
        assert "installed" in traced


@pytest.mark.parametrize(
    "module,endpoint",
    [
        (audio_inspection, "/api/audio/inspect"),
        (diagnostics_api, "/api/audio/diagnostics"),
        (rir_api, "/api/rir/analyze"),
    ],
)
def test_invalid_audio_still_uses_existing_client_error_contract(module, endpoint):
    with TestClient(app) as client:
        response = client.post(
            endpoint,
            files={"file": ("broken.wav", b"bad audio", "audio/wav")},
        )
    assert response.status_code == 400
