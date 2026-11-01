"""Tests for deterministic audio decoding and normalization."""

import wave
from pathlib import Path

import numpy as np
import pytest

from app.preprocessing import load_audio


def make_wav(path: Path, *, rate: int = 8_000, channels: int = 2, seconds: float = 0.25) -> None:
    times = np.arange(int(seconds * rate), dtype=np.float64) / rate
    signal = np.rint(np.sin(2 * np.pi * 440 * times) * 8_000).astype("<i2")
    interleaved = np.repeat(signal[:, None], channels, axis=1)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(interleaved.tobytes())


def test_resamples_and_converts_stereo_to_mono(tmp_path: Path) -> None:
    source = tmp_path / "tone.wav"
    make_wav(source)
    result = load_audio(source)
    assert result.sample_rate == 16_000
    assert result.original_sample_rate == 8_000
    assert result.original_channels == 2
    assert result.samples.ndim == 1
    assert result.samples.dtype == np.float32
    assert result.duration_seconds == pytest.approx(0.25, abs=0.002)
    assert result.rms_amplitude > 0


def test_accepts_a_custom_target_rate(tmp_path: Path) -> None:
    source = tmp_path / "tone.wav"
    make_wav(source)
    result = load_audio(source, sample_rate=8_000)
    assert result.sample_rate == 8_000
    assert len(result.samples) == 2_000


def test_rejects_long_recordings_before_loading(tmp_path: Path) -> None:
    source = tmp_path / "long.wav"
    make_wav(source, seconds=0.3)
    with pytest.raises(ValueError, match="duration exceeds"):
        load_audio(source, max_duration_seconds=0.1)


def test_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="does not exist"):
        load_audio(tmp_path / "missing.wav")


def test_rejects_unsupported_extension(tmp_path: Path) -> None:
    source = tmp_path / "tone.txt"
    source.write_bytes(b"not a recording")
    with pytest.raises(ValueError, match="Only WAV and MP3"):
        load_audio(source)


def test_rejects_invalid_audio_contents(tmp_path: Path) -> None:
    source = tmp_path / "garbage.wav"
    source.write_bytes(b"not an audio file")
    with pytest.raises(OSError, match="could not be decoded"):
        load_audio(source)


def test_rejects_invalid_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        load_audio(tmp_path / "tone.wav", sample_rate=0)
