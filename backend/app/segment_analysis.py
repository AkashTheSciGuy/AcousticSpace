"""Experimental overlapping-window scores from the existing local models.

A high fake-class score on a window is NOT validated tamper localization:
classification features may change with phonetics, silence, microphone or noise.
"""
from dataclasses import replace

import numpy as np

from app.features import extract_features
from app.model_inference import (
    EXTRA_FEATURES, EXPECTED_FEATURES, ModelUnavailableError,
    _load_model, model_path,
)
from app.preprocessing import PreprocessedAudio
from app.speech_acoustics import extract_speech_acoustics

WINDOW_SECONDS = 3.0
HOP_SECONDS = 1.5
MAX_WINDOWS = 60


def score_segments(audio: PreprocessedAudio, variant: str = "baseline") -> dict:
    """Score overlapping windows with ONE trained model load and a batched call."""
    samples = np.asarray(audio.samples, dtype=np.float32)
    sr = audio.sample_rate
    if samples.ndim != 1 or samples.size < 32 or sr <= 0 or not np.isfinite(samples).all():
        raise ValueError("Audio must contain finite mono samples.")
    if samples.size / sr > 60.001:
        raise ValueError("Audio exceeds the 60-second limit.")

    # Model load also rejects invalid variants and feature signatures.
    pipeline = _load_model(model_path(variant), variant)
    window = max(32, round(WINDOW_SECONDS * sr))
    hop = max(1, round(HOP_SECONDS * sr))
    ranges = []
    features = []
    for start in range(0, samples.size, hop):
        end = min(start + window, samples.size)
        current = samples[start:end]
        rms = float(np.sqrt(np.mean(current.astype(np.float64) ** 2)))
        clip = replace(audio, samples=current, duration_seconds=len(current) / sr,
                       rms_amplitude=rms)
        base = extract_features(clip)
        values = [float(getattr(base, name)) for name in EXPECTED_FEATURES]
        if variant == "acoustic":
            acoustics = extract_speech_acoustics(clip)
            values.extend(float(getattr(acoustics, name)) for name in EXTRA_FEATURES)
        features.append(values)
        ranges.append((start, end))
        if len(ranges) > MAX_WINDOWS:
            raise ValueError("Too many analysis windows.")
        if end == samples.size:
            break

    feature_matrix = np.asarray(features, dtype=np.float64)
    if not np.isfinite(feature_matrix).all():
        raise ValueError("Audio generated non-finite feature values.")
    try:
        probabilities = np.asarray(pipeline.predict_proba(feature_matrix), dtype=np.float64)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ModelUnavailableError("Model could not score audio windows.") from exc
    if (probabilities.shape != (len(ranges), 2)
            or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
        raise ModelUnavailableError("Model returned invalid window scores.")

    fake_class_index = list(pipeline.classes_).index(1)
    entries = []
    for (start, end), probability in zip(ranges, probabilities):
        fake = float(probability[fake_class_index])
        entries.append({
            "start_seconds": round(start / sr, 3),
            "end_seconds": round(end / sr, 3),
            "fake_score": round(fake, 6),
            "flagged": fake >= 0.5,
        })

    return {
        "variant": variant,
        "window_seconds": WINDOW_SECONDS,
        "hop_seconds": HOP_SECONDS,
        "segments": entries,
        "warning": (
            "Exploratory overlapping-window classifier scores, NOT validated "
            "deepfake localization or measured RIR. Scores are uncalibrated; "
            "a flagged window is not proof of tampering."
        ),
    }
