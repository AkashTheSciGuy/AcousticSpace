"""Tests for deterministic signal feature extraction."""

import numpy as np
import pytest

from app.features import extract_features
from app.preprocessing import PreprocessedAudio


def audio(samples: np.ndarray, rate: int = 16_000) -> PreprocessedAudio:
    return PreprocessedAudio(
        samples=samples, sample_rate=rate,
        duration_seconds=(len(samples) / rate if rate > 0 else 0.0),
        original_sample_rate=rate, original_channels=1,
        rms_amplitude=0.0,
    )


def test_sine_wave_has_expected_duration_and_rms() -> None:
    t = np.arange(16_000) / 16_000
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    features = extract_features(audio(tone))
    assert features.sample_rate == 16_000
    assert features.duration_seconds == pytest.approx(1.0)
    assert features.rms_amplitude == pytest.approx(0.5 / np.sqrt(2), abs=0.001)
    assert 0 < features.zero_crossing_rate < 1
    assert 250 < features.spectral_centroid_hz < 650


def test_silence_produces_zero_features() -> None:
    features = extract_features(audio(np.zeros(4096, dtype=np.float32)))
    assert features.rms_amplitude == 0
    assert features.zero_crossing_rate == 0
    assert features.spectral_centroid_hz == 0


def test_empty_audio_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        extract_features(audio(np.array([], dtype=np.float32)))


def test_nonfinite_samples_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        extract_features(audio(np.array([0.0, np.nan, 0.1], dtype=np.float32)))


def test_invalid_sample_rate_rejected() -> None:
    with pytest.raises(ValueError, match="Sample rate"):
        extract_features(audio(np.ones(4096, dtype=np.float32), rate=0))


def test_short_audio_supported() -> None:
    result = extract_features(audio(np.array([0.0, 0.1], dtype=np.float32)))
    assert np.isfinite(result.spectral_centroid_hz)
    assert result.rms_amplitude > 0
