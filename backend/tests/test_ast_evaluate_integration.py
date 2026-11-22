"""Score existing AST checkpoints with a stubbed inference runtime (no GPU required)."""
from contextlib import nullcontext
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import evaluate_ast


class FakeTensor:
    def __init__(self, values):
        self.data = np.asarray(values)
    def to(self, _device):
        return self
    def detach(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return self.data


class FakeModel:
    def eval(self):
        return self


class FakeTorch:
    @staticmethod
    def inference_mode():
        return nullcontext()
    @staticmethod
    def softmax(logits, dim):
        assert dim == -1
        return logits


def test_evaluate_checkpoint_reads_only_requested_split(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("stubbed split_rows uses synthetic recordings", encoding="utf-8")
    rows = {"train": [("train.wav", 0)], "val": [("real.wav", 0), ("fake.wav", 1)],
            "test": [("test.wav", 1)]}
    visited = []
    monkeypatch.setattr(evaluate_ast, "split_rows", lambda *_: rows)
    monkeypatch.setattr(evaluate_ast, "_load_local_ast",
                        lambda *_: (FakeTorch, None, FakeModel(), "cpu"))
    def fake_load(path):
        visited.append(path)
        return SimpleNamespace(samples=np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(evaluate_ast, "load_audio", fake_load)
    monkeypatch.setattr(evaluate_ast, "ast_input_values",
                        lambda *_: FakeTensor(np.zeros((1, 1024, 128), dtype=np.float32)))
    score_iter = iter([.2, .9])
    def fake_forward(model, inputs):
        assert isinstance(model, FakeModel)
        fake = next(score_iter)
        return SimpleNamespace(logits=FakeTensor([[1 - fake, fake]]))
    monkeypatch.setattr(evaluate_ast, "forward_ast", fake_forward)

    report = evaluate_ast.evaluate_checkpoint("root", manifest, "model",
                                              split="val", select_threshold=True,
                                              bootstrap_samples=20, bootstrap_seed=17)
    assert report["dataset_manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert report["reported_training_mode"] == "unknown"
    assert report["uncertainty"]["bootstrap_samples"] == 20
    assert report["uncertainty"]["fixed_threshold"] == 0.5
    assert report["uncertainty"]["metrics"]["balanced_accuracy"]["point_estimate"] == 1.0
    assert visited == ["real.wav", "fake.wav"]
    assert report["metrics"]["accuracy"] == 1.0
    assert report["metrics"]["confusion_matrix_real_fake"] == [[1, 0], [0, 1]]
    assert report["selected_validation_threshold"]["threshold"] >= .2


def test_failed_model_probability_validation(monkeypatch):
    monkeypatch.setattr(evaluate_ast, "split_rows",
                        lambda *_: {"val": [("a", 0), ("b", 1)]})
    monkeypatch.setattr(evaluate_ast, "_load_local_ast",
                        lambda *_: (FakeTorch, None, FakeModel(), "cpu"))
    monkeypatch.setattr(evaluate_ast, "load_audio",
                        lambda path: SimpleNamespace(samples=np.zeros(16000, dtype=np.float32)))
    monkeypatch.setattr(evaluate_ast, "ast_input_values",
                        lambda *_: FakeTensor(np.zeros((1, 1024, 128), dtype=np.float32)))
    monkeypatch.setattr(evaluate_ast, "forward_ast",
                        lambda *_: SimpleNamespace(logits=FakeTensor([[.1, .1]])))
    with pytest.raises(ValueError, match="invalid class probabilities"):
        evaluate_ast.evaluate_checkpoint("root", "manifest", "model")
