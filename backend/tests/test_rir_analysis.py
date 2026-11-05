"""Tests for the RIR metric calculations, including unestimable cases."""

import numpy as np
import pytest

from app.rir_analysis import analyze_rir


def synthetic_decay(rate: int = 16000, rt60: float = 0.6) -> np.ndarray:
    rng = np.random.default_rng(42)
    t = np.arange(rate * 2) / rate
    # Amplitude falls 60 dB over rt60 seconds; noise excites the decay.
    envelope = np.exp(-np.log(1000.0) * t / rt60)
    noise = rng.normal(size=t.size)
    noise[0] = 10.0
    return (noise * envelope).astype(np.float32)


def test_estimates_rt60_from_synthetic_decay() -> None:
    result = analyze_rir(synthetic_decay(), 16000)
    assert result.rt60_t20_seconds == pytest.approx(0.6, abs=0.08)
    assert result.decay_fit_r_squared is not None
    assert result.decay_fit_r_squared >= 0.9
    assert result.clarity_c50_db is not None
    assert result.direct_to_reverberant_ratio_db is not None


def test_direct_impulse_does_not_invent_rt60() -> None:
    samples = np.zeros(16000, dtype=np.float32)
    samples[0] = 1
    result = analyze_rir(samples, 16000)
    assert result.rt60_t20_seconds is None
    assert result.clarity_c50_db is None


@pytest.mark.parametrize('samples,rate', [
    (np.zeros(16000), 16000),
    (np.array([1.0, 0.0]), 16000),
    (np.full(16000, np.nan), 16000),
    (np.ones(16000), 0),
])
def test_bad_rir_rejected(samples: np.ndarray, rate: int) -> None:
    with pytest.raises(ValueError):
        analyze_rir(samples, rate)


def test_delayed_onset() -> None:
    impulse = synthetic_decay()
    samples = np.concatenate([np.zeros(4000), impulse])
    result = analyze_rir(samples, 16000)
    assert result.onset_seconds == pytest.approx(0.25)
