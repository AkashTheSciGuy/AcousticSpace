"""Segment scores respect input duration and preserve default model behavior."""
import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.preprocessing import PreprocessedAudio
from app import segment_analysis

client = TestClient(app)


class FakeClassifier:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, matrix):
        assert matrix.ndim == 2 and matrix.shape[1] in (3, 6)
        fake = np.full(matrix.shape[0], 0.6)
        return np.column_stack((1 - fake, fake))


def make_audio(seconds=5.0):
    sr = 16000
    data = 0.2 * np.sin(2 * np.pi * 440 * np.arange(round(sr * seconds)) / sr)
    return PreprocessedAudio(samples=data.astype(np.float32), sample_rate=sr,
                             duration_seconds=seconds, original_sample_rate=sr,
                             original_channels=1, rms_amplitude=0.141)


@pytest.mark.parametrize("variant,width", [("baseline", 3), ("acoustic", 6)])
def test_segments_are_bounded_and_model_is_loaded_once(monkeypatch, variant, width):
    calls = []
    def load(path, variant):
        calls.append(variant)
        return FakeClassifier()
    monkeypatch.setattr(segment_analysis, "_load_model", load)
    result = segment_analysis.score_segments(make_audio(5), variant)
    assert len(calls) == 1
    assert result["segments"][0]["start_seconds"] == 0
    assert result["segments"][-1]["end_seconds"] == 5
    assert all(row["flagged"] is True for row in result["segments"])
    assert "NOT validated" in result["warning"]


def test_short_clip_produces_one_window(monkeypatch):
    monkeypatch.setattr(segment_analysis, "_load_model", lambda *args: FakeClassifier())
    result = segment_analysis.score_segments(make_audio(.75))
    assert len(result["segments"]) == 1
    assert result["segments"][0]["end_seconds"] == pytest.approx(.75)


def test_oversize_duration_is_rejected_before_model_loading(monkeypatch):
    monkeypatch.setattr(segment_analysis, "_load_model",
                        lambda *args: pytest.fail("Unexpected model load"))
    with pytest.raises(ValueError, match="60-second"):
        segment_analysis.score_segments(make_audio(61))


def test_segments_api_requires_a_trained_model(tmp_path, monkeypatch):
    monkeypatch.setenv("ACOUSTICSPACE_BASELINE_MODEL", str(tmp_path / "missing.joblib"))
    response = client.post("/api/model/segments",
                           files={"file": ("clip.wav", b"RIFFxxxxWAVE")})
    assert response.status_code == 503


def test_invalid_variant_is_rejected():
    response = client.post("/api/model/segments?variant=unknown",
                           files={"file": ("clip.wav", b"RIFFxxxxWAVE")})
    assert response.status_code == 422
