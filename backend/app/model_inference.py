"""Safe-by-default baseline inference service for a locally trained, trusted model.

This is an experimental audio-feature baseline, NOT the RIR/AST detector.
Do not load untrusted joblib/pickle artifacts: joblib deserialization runs code.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.features import extract_features
from app.preprocessing import PreprocessedAudio

EXPECTED_FEATURES = ('rms_amplitude', 'zero_crossing_rate', 'spectral_centroid_hz')
EXPECTED_LABELS = ('real', 'fake')
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'baseline_model.joblib'


class ModelUnavailableError(RuntimeError):
    """No compatible, trusted, locally trained baseline model is available."""


def model_path() -> Path:
    """An operator-controlled local path; never accepted from an HTTP request."""
    override = os.getenv('ACOUSTICSPACE_BASELINE_MODEL')
    return Path(override).expanduser().resolve() if override else DEFAULT_MODEL_PATH


def _load_model(path: Path):
    if not path.is_file():
        raise ModelUnavailableError('No trained baseline model is installed. Train on labeled audio first.')
    try:
        import joblib
        artifact = joblib.load(path)  # Load ONLY self-generated, trusted artifacts.
    except (ImportError, OSError, ValueError, EOFError, KeyError, TypeError) as exc:
        raise ModelUnavailableError('Baseline model dependencies or artifact are unavailable.') from exc
    if not isinstance(artifact, dict) or tuple(artifact.get('features', ())) != EXPECTED_FEATURES or tuple(artifact.get('labels', ())) != EXPECTED_LABELS:
        raise ModelUnavailableError('Incompatible baseline model artifact.')
    pipeline = artifact.get('pipeline')
    if not callable(getattr(pipeline, 'predict_proba', None)) or not hasattr(pipeline, 'classes_'):
        raise ModelUnavailableError('Model lacks calibrated class probability interface.')
    classes = np.asarray(pipeline.classes_)
    if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
        raise ModelUnavailableError('Model classes do not match real=0 and fake=1.')
    return pipeline


def model_status() -> dict:
    path = model_path()
    if not path.is_file():
        return {'available': False, 'model_type': 'experimental_audio_feature_baseline',
                'message': 'No trained model found. Dataset preparation and training are required.'}
    try:
        _load_model(path)
    except ModelUnavailableError as exc:
        return {'available': False, 'model_type': 'experimental_audio_feature_baseline', 'message': str(exc)}
    return {'available': True, 'model_type': 'experimental_audio_feature_baseline',
            'message': 'Experimental baseline installed; accuracy depends on separate dataset evaluation.'}


def predict_audio(audio: PreprocessedAudio) -> dict:
    pipeline = _load_model(model_path())
    f = extract_features(audio)
    values = np.array([[f.rms_amplitude, f.zero_crossing_rate, f.spectral_centroid_hz]], dtype=np.float64)
    try:
        probs = np.asarray(pipeline.predict_proba(values), dtype=float)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ModelUnavailableError('The installed baseline model could not process this recording.') from exc
    if probs.shape != (1, 2) or not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any() or not np.isclose(probs.sum(), 1, atol=1e-5):
        raise ModelUnavailableError('The installed model returned invalid probabilities.')
    classes = list(pipeline.classes_)
    fake_probability = float(probs[0, classes.index(1)])
    return {
        'status': 'predicted',
        'model_type': 'experimental_audio_feature_baseline',
        'label': 'fake' if fake_probability >= 0.5 else 'real',
        'fake_probability': round(fake_probability, 6),
        'real_probability': round(1.0 - fake_probability, 6),
        'disclaimer': ('Experimental baseline using RMS, zero-crossing rate and spectral centroid. '
                       'Not RIR-based, not verified accuracy, not forensic evidence.'),
    }
