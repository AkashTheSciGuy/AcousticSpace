"""Optional local AST inference; never downloads weights during API requests.

Only operator-generated local safetensors models are accepted. This is a
speech spoofing experiment, not recovered RIR or a physics mismatch detector.
"""
import importlib.util
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.preprocessing import PreprocessedAudio

DEFAULT_AST_PATH = Path(__file__).resolve().parents[1] / "models" / "ast_model"


class ASTUnavailableError(RuntimeError):
    """A trained, compatible local AST checkpoint is not available."""


def ast_path() -> Path:
    override = os.getenv("ACOUSTICSPACE_AST_MODEL")
    return Path(override).expanduser().resolve() if override else DEFAULT_AST_PATH


def _ready_path(path: Path) -> bool:
    return (path.is_dir() and (path / "config.json").is_file()
            and (path / "preprocessor_config.json").is_file()
            and ((path / "model.safetensors").is_file()
                 or (path / "model.safetensors.index.json").is_file()))


@lru_cache(maxsize=2)
def _load_local_ast(path_string: str):
    try:
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        path = Path(path_string)
        extractor = ASTFeatureExtractor.from_pretrained(path, local_files_only=True,
                                                         trust_remote_code=False)
        model = ASTForAudioClassification.from_pretrained(
            path, local_files_only=True, use_safetensors=True, trust_remote_code=False)
        classes = model.config.id2label
        if str(classes.get(0, "")).lower() != "real" or str(classes.get(1, "")).lower() != "fake":
            raise ASTUnavailableError("AST labels must map 0=real and 1=fake.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        return torch, extractor, model, device
    except ASTUnavailableError:
        raise
    except (ImportError, OSError, RuntimeError, ValueError, AttributeError, KeyError) as exc:
        raise ASTUnavailableError("AST dependencies or trained local checkpoint are unavailable.") from exc


def ast_status() -> dict:
    path = ast_path()
    if not _ready_path(path):
        return {"available": False, "model_type": "optional_ast_experiment",
                "message": "No trained AST safetensors checkpoint is installed."}
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
        return {"available": False, "model_type": "optional_ast_experiment",
                "message": "AST dependencies are not installed (see requirements-ast.txt)."}
    try:
        _load_local_ast(str(path))
    except ASTUnavailableError as exc:
        return {"available": False, "model_type": "optional_ast_experiment",
                "message": str(exc)}
    return {"available": True, "model_type": "optional_ast_experiment",
            "message": "Trusted local AST checkpoint loaded. Scores are not calibrated."}


def predict_ast(audio: PreprocessedAudio) -> dict:
    if audio.sample_rate != 16000:
        raise ValueError("AST requires 16-kHz normalized audio.")
    torch, extractor, model, device = _load_local_ast(str(ast_path()))
    waveform = np.asarray(audio.samples, dtype=np.float32)
    if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("Expected nonempty finite mono audio.")
    # Match the 10-s maximum clip length used during training.
    clipped = waveform[:160000]
    features = extractor(clipped, sampling_rate=16000, return_tensors="pt")
    with torch.inference_mode():
        logits = model(input_features=features["input_features"].to(device)).logits
        scores = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
    if scores.shape != (2,) or not np.isfinite(scores).all():
        raise ASTUnavailableError("AST returned an invalid output.")
    fake = float(scores[1])
    return {"status": "predicted", "model_type": "optional_ast_experiment",
            "label": "fake" if fake >= .5 else "real",
            "fake_probability": round(fake, 6),
            "real_probability": round(1 - fake, 6),
            "analyzed_seconds": round(min(waveform.size, 160000) / 16000, 3),
            "disclaimer": (
                "Optional AST speech classifier; only the first 10 s were analyzed. "
                "Softmax scores are uncalibrated; not an RIR mismatch detector or "
                "forensic evidence."
            )}
