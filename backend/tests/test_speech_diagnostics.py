"""Speech spectrogram and cadence metadata are bounded, finite, and not claims."""
import io
import wave
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.preprocessing import PreprocessedAudio
from app.speech_diagnostics import analyze_speech_diagnostics

client = TestClient(app)


def audio(seconds=1, silent=False):
    rate = 16000
    values = (np.zeros(round(seconds * rate)) if silent else
              0.3 * np.sin(2 * np.pi * 220 * np.arange(round(seconds * rate)) / rate))
    return PreprocessedAudio(samples=values.astype(np.float32), sample_rate=rate,
        duration_seconds=seconds, original_sample_rate=rate, original_channels=1,
        rms_amplitude=float(np.sqrt(np.mean(values**2))))


@pytest.mark.parametrize("seconds,silent", [(0.3, False), (1.0, False), (1.0, True)])
def test_diagnostics_are_bounded_and_finite(seconds, silent):
    report = analyze_speech_diagnostics(audio(seconds, silent))
    assert report["mel_bins"] == 48
    assert 1 <= report["time_bins"] <= 100
    matrix = np.asarray(report["log_mel_db"])
    assert matrix.shape == (48, report["time_bins"])
    assert np.isfinite(matrix).all()
    assert (matrix <= 0.01).all() and (matrix >= -80.01).all()
    assert all(0 <= t <= seconds for t in report["energy_peak_seconds"])
    assert "NOT validated" in report["warning"]


def test_rejects_nonfinite_input():
    sample = audio()
    data = np.array(sample.samples, copy=True)
    data[100] = float("nan")
    from dataclasses import replace
    with pytest.raises(ValueError, match="finite"):
        analyze_speech_diagnostics(replace(sample, samples=data))


def test_api_returns_spectrogram_and_bounded_response():
    values = np.rint(10000 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)).astype("<i2")
    stream = io.BytesIO()
    with wave.open(stream, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(values.tobytes())
    response = client.post("/api/audio/diagnostics",
                           files={"file": ("clip.wav", stream.getvalue())})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "analyzed"
    assert len(result["log_mel_db"]) == 48
    assert len(result["log_mel_db"][0]) <= 100


def test_rejects_bad_file_extension():
    assert client.post("/api/audio/diagnostics",
                       files={"file": ("text.txt", b"bad")}).status_code == 415
