"""Exploratory speech diagnostics: spectrogram and energy-based timing cues.

This is NOT speech/environmental source separation, RIR extraction, phonetic
syllable alignment, or validated respiratory event detection.
"""
import librosa
import numpy as np
from scipy.signal import find_peaks

from app.preprocessing import PreprocessedAudio

MEL_BINS = 48
MAX_TIME_BINS = 100


def analyze_speech_diagnostics(audio: PreprocessedAudio) -> dict:
    samples = np.asarray(audio.samples, dtype=np.float32)
    rate = audio.sample_rate
    if samples.ndim != 1 or samples.size < 512 or rate < 1000 or not np.isfinite(samples).all():
        raise ValueError("Requires finite mono audio with at least 512 samples.")
    duration = samples.size / rate
    if duration > 60.001:
        raise ValueError("Audio exceeds the 60-second limit.")

    fft_size = 512
    hop = max(1, round(rate * 0.01))
    # Time is downsampled for a small bounded JSON response.
    power = librosa.feature.melspectrogram(
        y=samples, sr=rate, n_fft=fft_size, hop_length=hop,
        n_mels=MEL_BINS, center=True, power=2.0,
    )
    log_mel = librosa.power_to_db(power, ref=np.max, top_db=80.0)
    columns = min(MAX_TIME_BINS, log_mel.shape[1])
    indices = np.linspace(0, log_mel.shape[1] - 1, columns).astype(int)
    matrix = np.round(log_mel[:, indices].astype(np.float64), 1)

    envelope = librosa.feature.rms(y=samples, frame_length=fft_size,
                                   hop_length=hop, center=True)[0].astype(float)
    if not np.isfinite(envelope).all():
        raise ValueError("Non-finite short-time energy.")
    peak_value = float(np.max(envelope))
    activity_threshold = max(0.0001, 0.25 * float(np.quantile(envelope, 0.8)))
    peak_indices, _ = find_peaks(
        envelope, distance=max(1, round(0.12 * rate / hop)),
        prominence=max(1e-4, peak_value * 0.06),
        height=activity_threshold,
    )
    energy_peak_seconds = [round(min(float(i * hop / rate), duration), 3)
                           for i in peak_indices[:80]]

    # Quiet noisy intervals may contain breathing, but also equipment or room noise.
    flatness = librosa.feature.spectral_flatness(
        y=samples, n_fft=fft_size, hop_length=hop, center=True,
    )[0]
    length = min(len(flatness), len(envelope))
    quiet = envelope[:length] < activity_threshold
    boundaries = np.flatnonzero(np.diff(np.r_[False, quiet, False].astype(int)))
    candidates = []
    for start, end in zip(boundaries[::2], boundaries[1::2]):
        span_seconds = (end - start) * hop / rate
        if not (0.15 <= span_seconds <= 1.5):
            continue
        # Quiet but non-silent, noise-like energy: NOT proof of inhalation.
        mean_energy = float(np.mean(envelope[start:end]))
        mean_flatness = float(np.mean(flatness[start:end]))
        if mean_energy > 1e-4 and mean_flatness > 0.25:
            candidates.append({
                "start_seconds": round(start * hop / rate, 3),
                "end_seconds": round(min(end * hop / rate, duration), 3),
            })
            if len(candidates) >= 20:
                break

    spacing = (float(np.median(np.diff(peak_indices))) * hop / rate
               if len(peak_indices) >= 2 else None)
    return {
        "duration_seconds": round(duration, 3),
        "mel_bins": MEL_BINS,
        "time_bins": columns,
        "log_mel_db": matrix.tolist(),
        "energy_peak_seconds": energy_peak_seconds,
        "energy_peak_rate_hz": round(1 / spacing, 3) if spacing and spacing > 0 else None,
        "noisy_quiet_intervals": candidates,
        "warning": (
            "Log-mel spectrogram and energy peaks describe signal structure, not "
            "isolated environmental reverb, syllable timing or an estimated RIR. "
            "Noisy quiet intervals are only possible breathing candidates; "
            "they can be background noise and are NOT validated inhalations."
        ),
    }
