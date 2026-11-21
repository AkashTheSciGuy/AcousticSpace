"""Regression tests for the AST/Hugging Face feature extractor interface."""

import numpy as np
import pytest

from app.ast_inputs import ast_input_values, forward_ast


class FakeExtractor:
    def __call__(self, waveform, *, sampling_rate, return_tensors):
        assert sampling_rate == 16_000
        assert return_tensors == "pt"
        assert len(waveform) == 16_000
        return {"input_values": np.ones((1, 12, 128), dtype=np.float32)}


class StrictAST:
    """Keyword-only signature mimics ASTForAudioClassification.forward."""

    def __call__(self, *, input_values, labels=None):
        assert input_values.shape == (1, 12, 128)
        return {"labels": labels, "batch": input_values.shape[0]}


def test_extractor_and_model_use_documented_input_values_keyword():
    values = ast_input_values(FakeExtractor(), np.zeros(16000, dtype=np.float32))
    assert values.shape == (1, 12, 128)
    model = StrictAST()
    assert forward_ast(model, values)["batch"] == 1
    assert forward_ast(model, values, labels=np.array([1]))["labels"][0] == 1


def test_missing_input_values_produces_clear_error():
    extractor = lambda *_args, **_kwargs: {"input_features": np.zeros((1, 12, 128))}
    with pytest.raises(ValueError, match="did not return input_values"):
        ast_input_values(extractor, np.zeros(16000, dtype=np.float32))


def test_invalid_ast_batch_shape_is_rejected():
    extractor = lambda *_args, **_kwargs: {"input_values": np.zeros((12, 128))}
    with pytest.raises(ValueError, match="3-D spectrogram"):
        ast_input_values(extractor, np.zeros(16000, dtype=np.float32))


def test_real_ast_feature_extractor_contract_when_installed():
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    extractor = transformers.ASTFeatureExtractor()
    values = ast_input_values(extractor, np.zeros(16000, dtype=np.float32))
    assert tuple(values.shape) == (1, extractor.max_length, extractor.num_mel_bins)
