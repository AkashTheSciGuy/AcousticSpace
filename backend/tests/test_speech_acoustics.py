"""Tests for experimental speech-acoustic descriptors (no RIR claims)."""

import numpy as np
import pytest

from app.preprocessing import PreprocessedAudio
from app.speech_acoustics import extract_speech_acoustics


def audio(samples, sr=16000):
    samples = np.asarray(samples, dtype=np.float32)
    return PreprocessedAudio(samples=samples, sample_rate=sr,
                             duration_seconds=len(samples) / sr,
                             original_sample_rate=sr, original_channels=1,
                             rms_amplitude=float(np.sqrt(np.mean(samples ** 2))))


def test_tone_is_less_flat_than_noise():
    time = np.arange(16000) / 16000
    tone = extract_speech_acoustics(audio(np.sin(2 * np.pi * 440 * time)))
    noise = extract_speech_acoustics(audio(np.random.default_rng(5).normal(size=16000)))
    assert 0 <= tone.mean_spectral_flatness < noise.mean_spectral_flatness <= 1
    assert tone.mean_positive_spectral_flux >= 0


def test_silence_is_safe_and_finite():
    features = extract_speech_acoustics(audio(np.zeros(16000)))
    assert all(value == 0 for value in features.to_dict().values())


def test_reproducible_and_handles_short_audio():
    samples = np.linspace(-1, 1, 40)
    first = extract_speech_acoustics(audio(samples))
    assert first == extract_speech_acoustics(audio(samples))
    assert extract_speech_acoustics(audio([0, 1])).mean_spectral_flatness == 0


def test_rejects_nonfinite_and_nonmono_audio():
    with pytest.raises(ValueError):
        extract_speech_acoustics(audio([0, float('nan')]))
    with pytest.raises(ValueError):
        extract_speech_acoustics(audio(np.zeros((32, 2))))
