"""Malformed local model files must never crash the API status endpoint."""
import io
import wave
import numpy as np
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_corrupted_baseline_reports_unavailable(tmp_path, monkeypatch):
    path = tmp_path / "bad.joblib"
    path.write_bytes(b"\x80\x04.")
    monkeypatch.setenv("ACOUSTICSPACE_BASELINE_MODEL", str(path))
    response = client.get("/api/model/status?variant=baseline")
    assert response.status_code == 200
    assert response.json()["available"] is False
    assert "artifact" in response.json()["message"]

def test_corrupted_candidate_reports_unavailable_and_503(tmp_path, monkeypatch):
    path = tmp_path / "bad.joblib"
    path.write_bytes(b"\x80\x04.")
    monkeypatch.setenv("ACOUSTICSPACE_ACOUSTIC_MODEL", str(path))
    response = client.get("/api/model/status?variant=acoustic")
    assert response.status_code == 200
    assert response.json()["available"] is False
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.zeros(16000, dtype="<i2").tobytes())
    response = client.post("/api/model/predict?variant=acoustic",
                           files={"file": ("clip.wav", stream.getvalue())})
    assert response.status_code == 503
