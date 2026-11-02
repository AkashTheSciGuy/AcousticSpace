"""Deterministic signal features for later AcousticSpace ML experiments.

These features describe an audio signal; they do not classify authenticity.
"""

from dataclasses import dataclass

import librosa
import numpy as np

from app.preprocessing import PreprocessedAudio


@dataclass(frozen=True)
class AudioFeatures:
    duration_seconds: float
    sample_rate: int
    rms_amplitude: float
    zero_crossing_rate: float
    spectral_centroid_hz: float


def extract_features(audio: PreprocessedAudio) -> AudioFeatures:
    """Extract global time-domain and frequency-domain features from mono audio."""
    samples = np.asarray(audio.samples, dtype=np.float32)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("Expected non-empty mono audio samples.")
    if audio.sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")
    if not np.isfinite(samples).all():
        raise ValueError("Audio samples must be finite.")

    # Avoid librosa's default FFT window (2048) on short recordings.
    frame_length = min(2048, max(2, samples.size))
    if frame_length % 2:
        frame_length -= 1
    hop_length = min(512, max(1, frame_length // 4))

    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(
        y=samples, frame_length=frame_length, hop_length=hop_length,
        center=False,
    ))) if samples.size >= frame_length else 0.0
    centroid = float(np.mean(librosa.feature.spectral_centroid(
        y=samples, sr=audio.sample_rate, n_fft=frame_length,
        hop_length=hop_length, center=False,
    ))) if samples.size >= frame_length else 0.0

    return AudioFeatures(
        duration_seconds=float(samples.size / audio.sample_rate),
        sample_rate=audio.sample_rate,
        rms_amplitude=rms,
        zero_crossing_rate=zcr,
        spectral_centroid_hz=centroid,
    )
