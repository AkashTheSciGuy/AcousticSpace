"""Experimental model loading and inference for trusted local training artifacts.

The default baseline remains unchanged. The opt-in six-feature candidate is
speech-acoustic, not RIR-based; neither score is forensic evidence.
Never load joblib artifacts from untrusted sources.
"""
import os
from pathlib import Path

import numpy as np

from app.features import extract_features
from app.model_evaluation import read_evaluation
from app.preprocessing import PreprocessedAudio
from app.speech_acoustics import extract_speech_acoustics

EXPECTED_FEATURES = ('rms_amplitude', 'zero_crossing_rate', 'spectral_centroid_hz')
EXTRA_FEATURES = ('mean_spectral_flatness', 'rms_coefficient_of_variation', 'mean_positive_spectral_flux')
EXPECTED_LABELS = ('real', 'fake')
MODEL_DIR = Path(__file__).resolve().parents[1] / 'models'
DEFAULT_MODEL_PATH = MODEL_DIR / 'baseline_model.joblib'
CANDIDATE_MODEL_PATH = MODEL_DIR / 'acoustic_candidate.joblib'
VARIANTS = ('baseline', 'acoustic')


class ModelUnavailableError(RuntimeError):
    """No compatible, trusted, locally trained model is available."""


def _variant(variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError('Unsupported model variant.')
    return variant


def model_path(variant: str = 'baseline') -> Path:
    """Resolve an operator-configured local model file, never from upload data."""
    variant = _variant(variant)
    name = 'ACOUSTICSPACE_BASELINE_MODEL' if variant == 'baseline' else 'ACOUSTICSPACE_ACOUSTIC_MODEL'
    default = DEFAULT_MODEL_PATH if variant == 'baseline' else CANDIDATE_MODEL_PATH
    override = os.getenv(name)
    return Path(override).expanduser().resolve() if override else default


def _load_model(path: Path, variant: str = 'baseline'):
    variant = _variant(variant)
    if not path.is_file():
        raise ModelUnavailableError('No trained model found for the selected variant.')
    try:
        import joblib
        artifact = joblib.load(path)  # Only operator-created trusted files.
    except (ImportError, OSError, ValueError, EOFError, KeyError, TypeError) as exc:
        raise ModelUnavailableError('Model dependencies or artifact are unavailable.') from exc
    expected = EXPECTED_FEATURES if variant == 'baseline' else EXPECTED_FEATURES + EXTRA_FEATURES
    if (not isinstance(artifact, dict)
            or tuple(artifact.get('features', ())) != expected
            or tuple(artifact.get('labels', ())) != EXPECTED_LABELS):
        raise ModelUnavailableError('Incompatible model artifact for the selected variant.')
    pipeline = artifact.get('pipeline')
    if not callable(getattr(pipeline, 'predict_proba', None)) or not hasattr(pipeline, 'classes_'):
        raise ModelUnavailableError('Model lacks class probability interface.')
    classes = np.asarray(pipeline.classes_)
    if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
        raise ModelUnavailableError('Model classes do not match real=0 and fake=1.')
    return pipeline


def model_status(variant: str = 'baseline') -> dict:
    variant = _variant(variant)
    path = model_path(variant)
    model_type = ('experimental_audio_feature_baseline' if variant == 'baseline'
                  else 'experimental_speech_acoustic_candidate')
    try:
        _load_model(path, variant)
    except ModelUnavailableError as exc:
        return {'available': False, 'model_type': model_type, 'variant': variant, 'message': str(exc)}
    result = {'available': True, 'model_type': model_type, 'variant': variant,
              'message': 'Experimental model installed; performance is dataset-dependent.'}
    # The original baseline has its own exact sidecar report. Candidate results
    # are shown in the separately verified /api/model/comparison endpoint.
    if variant == 'baseline':
        report = read_evaluation(path)
        if report is not None:
            result['evaluation'] = report
    return result


def predict_audio(audio: PreprocessedAudio, variant: str = 'baseline') -> dict:
    variant = _variant(variant)
    pipeline = _load_model(model_path(variant), variant)
    features = extract_features(audio)
    values = [float(getattr(features, key)) for key in EXPECTED_FEATURES]
    if variant == 'acoustic':
        extra = extract_speech_acoustics(audio)
        values.extend(float(getattr(extra, key)) for key in EXTRA_FEATURES)
    array = np.asarray([values], dtype=np.float64)
    if not np.isfinite(array).all():
        raise ModelUnavailableError('Recording produced non-finite model features.')
    try:
        probabilities = np.asarray(pipeline.predict_proba(array), dtype=np.float64)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ModelUnavailableError('The installed model could not process this recording.') from exc
    if (probabilities.shape != (1, 2) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.isclose(probabilities.sum(), 1, atol=1e-5)):
        raise ModelUnavailableError('The installed model returned invalid probabilities.')
    fake = float(probabilities[0, list(pipeline.classes_).index(1)])
    description = ('RMS, zero-crossing rate and spectral centroid' if variant == 'baseline'
                   else 'three baseline and three exploratory speech-acoustic features')
    return {
        'status': 'predicted',
        'model_type': ('experimental_audio_feature_baseline' if variant == 'baseline'
                       else 'experimental_speech_acoustic_candidate'),
        'variant': variant,
        'label': 'fake' if fake >= 0.5 else 'real',
        'fake_probability': round(fake, 6),
        'real_probability': round(1.0 - fake, 6),
        'disclaimer': f'Experimental model using {description}. Not RIR-based, not verified accuracy, not forensic evidence.',
    }
