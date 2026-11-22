"""Exercise AST freezing modes without downloading or loading GPU weights."""
from types import SimpleNamespace

import pytest

from scripts.train_ast import configure_trainable_ast


class FakeParameter:
    def __init__(self, elements):
        self.elements = elements
        self.requires_grad = True

    def requires_grad_(self, flag):
        self.requires_grad = bool(flag)
        return self

    def numel(self):
        return self.elements


class FakeModule:
    def __init__(self, *parameters):
        self.params = parameters

    def parameters(self):
        return iter(self.params)


class FakeModel:
    def __init__(self, layout):
        self.embedding = FakeModule(FakeParameter(50))
        self.classifier = FakeModule(FakeParameter(4))
        self.blocks = [
            FakeModule(FakeParameter(10)),
            FakeModule(FakeParameter(20)),
            FakeModule(FakeParameter(30)),
        ]
        self.norm = FakeModule(FakeParameter(2))
        if layout == "v4":
            self.audio_spectrogram_transformer = SimpleNamespace(
                encoder=SimpleNamespace(layer=self.blocks),
                layernorm=self.norm,
            )
        else:
            self.audio_spectrogram_transformer = SimpleNamespace(
                layers=self.blocks,
                layernorm=self.norm,
            )

    def parameters(self):
        yield from self.embedding.parameters()
        yield from self.classifier.parameters()
        for block in self.blocks:
            yield from block.parameters()
        yield from self.norm.parameters()


@pytest.mark.parametrize("layout", ["v4", "v5"])
def test_head_only_preserves_frozen_ast_encoder(layout):
    model = FakeModel(layout)
    result = configure_trainable_ast(model)
    assert result == {
        "mode": "head_only", "train_last_layers": 0,
        "trainable_parameters": 4, "total_parameters": 116,
    }
    assert model.classifier.params[0].requires_grad
    assert not model.embedding.params[0].requires_grad
    assert not model.norm.params[0].requires_grad
    assert not any(b.params[0].requires_grad for b in model.blocks)


@pytest.mark.parametrize("layout", ["v4", "v5"])
def test_partial_encoder_updates_only_requested_last_blocks(layout):
    model = FakeModel(layout)
    result = configure_trainable_ast(model, train_last_layers=2)
    assert result["mode"] == "partial_encoder_finetune"
    assert result["train_last_layers"] == 2
    assert result["trainable_parameters"] == 56
    assert not model.embedding.params[0].requires_grad
    assert not model.blocks[0].params[0].requires_grad
    assert model.blocks[1].params[0].requires_grad
    assert model.blocks[2].params[0].requires_grad
    assert model.norm.params[0].requires_grad


def test_full_encoder_mode_unfreezes_all_layers():
    model = FakeModel("v5")
    result = configure_trainable_ast(model, full_finetune=True)
    assert result["mode"] == "full_encoder_finetune"
    assert result["trainable_parameters"] == result["total_parameters"] == 116
    assert all(param.requires_grad for param in model.parameters())


@pytest.mark.parametrize("args", [
    {"train_last_layers": -1},
    {"train_last_layers": 1.5},
    {"train_last_layers": True},
    {"train_last_layers": 4},
    {"full_finetune": True, "train_last_layers": 1},
])
def test_invalid_modes_fail_closed(args):
    with pytest.raises(ValueError):
        configure_trainable_ast(FakeModel("v5"), **args)


def test_missing_encoder_layer_layout_does_not_fake_partial_training():
    model = FakeModel("v5")
    model.audio_spectrogram_transformer = SimpleNamespace(layernorm=model.norm)
    with pytest.raises(ValueError, match="unsupported layout"):
        configure_trainable_ast(model, train_last_layers=1)
