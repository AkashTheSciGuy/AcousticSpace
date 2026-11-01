"""Reusable audio decoding and normalization for future ML experiments.

This module is deliberately independent of the upload-validation endpoint.
It extracts basic signal metadata; it does not detect a deepfake or estimate RIR.
"""

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


TARGET_SAMPLE_RATE = 16_000
MAX_DURATION_SECONDS = 60.0
SUPPORTED_SUFFIXES = {".wav", ".mp3"}


@dataclass(frozen=True)
class PreprocessedAudio:
    samples: np.ndarray
    sample_rate: int
    duration_seconds: float
    original_sample_rate: int
    original_channels: int
    rms_amplitude: float


def load_audio(
    path: str | Path,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
    max_duration_seconds: float = MAX_DURATION_SECONDS,
) -> PreprocessedAudio:
    """Decode a WAV/MP3 recording and return 16-kHz mono float32 samples.

    Inspect metadata before decoding to reject excessive durations early.
    Only use paths to trusted, already validated audio files.

    Raises:
        ValueError: invalid settings, unsupported format, silent/empty input,
            or duration beyond the configured limit.
        OSError: file missing, unreadable, or not a decodable audio recording.
    """
    file_path = Path(path)
    if sample_rate <= 0 or max_duration_seconds <= 0:
        raise ValueError("Sample rate and maximum duration must be positive.")
    if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError("Only WAV and MP3 files are supported.")
    if not file_path.is_file():
        raise OSError(f"Audio file does not exist: {file_path}")

    try:
        info = sf.info(str(file_path))
    except (RuntimeError, OSError) as exc:
        raise OSError("Audio file could not be decoded.") from exc

    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise ValueError("Audio recording is empty or has invalid metadata.")

    duration = info.frames / info.samplerate
    if duration > max_duration_seconds:
        raise ValueError(
            f"Audio duration exceeds the {max_duration_seconds:g}-second limit."
        )

    try:
        samples, actual_sr = librosa.load(
            file_path,
            sr=sample_rate,
            mono=True,
            dtype=np.float32,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        raise OSError("Audio file could not be decoded.") from exc

    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("Audio contains no usable samples.")

    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
    return PreprocessedAudio(
        samples=samples,
        sample_rate=int(actual_sr),
        duration_seconds=float(samples.size / actual_sr),
        original_sample_rate=int(info.samplerate),
        original_channels=int(info.channels),
        rms_amplitude=rms,
    )
