"""Exploratory speech-acoustic descriptors; these are NOT RIR estimates.

The descriptors reflect signal dynamics and spectral texture. Speech content,
recording equipment and compression can affect them independently of room
acoustics. Do not interpret any value as RT60, C50 or a deepfake verdict.
"""

from dataclasses import asdict, dataclass

import numpy as np

from app.preprocessing import PreprocessedAudio


@dataclass(frozen=True)
class SpeechAcousticFeatures:
    mean_spectral_flatness: float
    rms_coefficient_of_variation: float
    mean_positive_spectral_flux: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def extract_speech_acoustics(audio: PreprocessedAudio) -> SpeechAcousticFeatures:
    """Calculate bounded, deterministic spectral and envelope descriptors.

    Spectral flatness is in [0, 1]; RMS variation and normalized positive
    spectral flux are non-negative. These quantities are not room-only cues.
    """
    samples = np.asarray(audio.samples, dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError('Expected nonempty finite mono audio samples.')
    if audio.sample_rate <= 0:
        raise ValueError('Sample rate must be positive.')

    frame_size = min(1024, samples.size)
    if frame_size < 16:
        return SpeechAcousticFeatures(0.0, 0.0, 0.0)
    hop = max(1, frame_size // 2)
    frames = np.lib.stride_tricks.sliding_window_view(samples, frame_size)[::hop]
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    active = rms > max(1e-6, float(rms.max()) * 0.01)
    if not np.any(active):
        return SpeechAcousticFeatures(0.0, 0.0, 0.0)

    selected_rms = rms[active]
    variation = float(np.std(selected_rms) / max(float(np.mean(selected_rms)), 1e-12))
    window = np.hanning(frame_size)
    spectrum = np.abs(np.fft.rfft(frames[active] * window, axis=1))
    power = spectrum ** 2
    epsilon = 1e-12
    flatness = np.exp(np.mean(np.log(power + epsilon), axis=1)) / np.mean(power + epsilon, axis=1)

    # Relative positive spectral change between consecutive active frames.
    # Normalize each spectrum to avoid confusing overall volume with flux.
    normalized = spectrum / np.maximum(spectrum.sum(axis=1, keepdims=True), epsilon)
    flux = (np.maximum(np.diff(normalized, axis=0), 0).sum(axis=1)
            if len(normalized) > 1 else np.array([0.0]))
    return SpeechAcousticFeatures(
        mean_spectral_flatness=float(np.clip(np.mean(flatness), 0, 1)),
        rms_coefficient_of_variation=variation,
        mean_positive_spectral_flux=float(np.mean(flux)),
    )
