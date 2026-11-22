"""Read-only external AST evaluation tests using synthetic fixtures, no model downloads."""
from __future__ import annotations

import csv
import hashlib
import json
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from scripts import evaluate_ast_external as external

FIELDS = ("path", "label", "sha256", "group", "source_reference")


def _csv(path, rows, columns=FIELDS):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "external"
    root.mkdir()
    files = []
    for i, frequency in enumerate((220, 300)):
        path = root / f"recording_{i}.wav"
        t = np.arange(16000) / 16000
        sf.write(path, .15 * np.sin(t * 2 * np.pi * frequency), 16000, format="WAV")
        files.append(path)
    rows = [{
        "path": path.name,
        "label": "real" if i == 0 else "fake",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "group": f"source_group_{i}",
        "source_reference": "official_source_record_for_unit_test",
    } for i, path in enumerate(files)]
    labels = tmp_path / "external.csv"
    _csv(labels, rows)
    original = tmp_path / "training.csv"
    _csv(original, [{
        "path": "training.flac", "label": "fake", "sha256": "1" * 64,
        "group": "training_speaker", "source_reference": "training",
    }])
    return root, labels, original, rows


def test_manifest_validation_has_real_files_and_no_shared_audio_hash(corpus):
    root, manifest, original, _ = corpus
    rows, provenance = external.load_external_records(root, manifest, original)
    assert len(rows) == 2
    assert [label for _, label in rows] == [0, 1]
    assert provenance["classes"] == {"real": 1, "fake": 1}
    assert provenance["declared_groups"] == 2
    assert provenance["exact_audio_hash_overlap"] == 0
    assert len(provenance["external_manifest_sha256"]) == 64


def test_manifest_rejects_original_audio_hash_overlap(corpus):
    root, manifest, original, rows = corpus
    _csv(original, [{**rows[0], "path": "training.flac"}])
    with pytest.raises(ValueError, match="overlaps known training"):
        external.load_external_records(root, manifest, original)


def test_manifest_rejects_unverified_audio_bytes(corpus):
    root, manifest, original, rows = corpus
    rows[0]["sha256"] = "0" * 64
    _csv(manifest, rows)
    with pytest.raises(ValueError, match="checksum does not match"):
        external.load_external_records(root, manifest, original)


def test_manifest_rejects_outside_root_and_missing_source(corpus):
    root, manifest, original, rows = corpus
    rows[0]["path"] = "../other.wav"
    _csv(manifest, rows)
    with pytest.raises(ValueError, match="safe relative"):
        external.load_external_records(root, manifest, original)
    rows[0]["path"] = "recording_0.wav"
    rows[0]["source_reference"] = "unknown"
    _csv(manifest, rows)
    with pytest.raises(ValueError, match="source group and provenance"):
        external.load_external_records(root, manifest, original)


def test_manifest_rejects_duplicate_and_imbalanced_groups(corpus):
    root, manifest, original, rows = corpus
    rows[1]["sha256"] = rows[0]["sha256"]
    _csv(manifest, rows)
    with pytest.raises(ValueError, match="repeated file or SHA-256"):
        external.load_external_records(root, manifest, original)
    rows[1]["sha256"] = hashlib.sha256((root / "recording_1.wav").read_bytes()).hexdigest()
    rows[1]["group"] = rows[0]["group"]
    _csv(manifest, rows)
    with pytest.raises(ValueError, match="two documented source groups"):
        external.load_external_records(root, manifest, original)


def test_manifest_rejects_missing_class(corpus):
    root, manifest, original, rows = corpus
    rows[1]["label"] = "real"
    _csv(manifest, rows)
    with pytest.raises(ValueError, match="both independently labeled"):
        external.load_external_records(root, manifest, original)


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def to(self, _device):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeTorch:
    @staticmethod
    def inference_mode():
        return nullcontext()

    @staticmethod
    def softmax(logits, dim):
        assert dim == -1
        return logits


class FakeModel:
    def eval(self):
        return self


def test_external_inference_scores_only_validated_external_files(corpus, monkeypatch):
    root, manifest, original, _ = corpus
    visited = []
    monkeypatch.setattr(
        external, "_load_local_ast",
        lambda *_: (FakeTorch, object(), FakeModel(), "cpu"),
    )
    def load_audio(path):
        visited.append(path.name)
        return SimpleNamespace(samples=np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(external, "load_audio", load_audio)
    monkeypatch.setattr(
        external, "ast_input_values", lambda *_: FakeTensor(np.zeros((1, 2, 2))),
    )
    fake_scores = iter([0.2, 0.9])
    monkeypatch.setattr(
        external, "forward_ast",
        lambda *_: SimpleNamespace(
            logits=FakeTensor([[1 - (score := next(fake_scores)), score]])
        ),
    )
    result = external.evaluate_external(root, manifest, original, root / "model",
                                        bootstrap_samples=20, seed=4)
    assert result["status"] == "evaluated"
    assert result["metrics"]["accuracy"] == 1
    assert result["metrics"]["confusion_matrix_real_fake"] == [[1, 0], [0, 1]]
    assert result["uncertainty"]["bootstrap_samples"] == 20
    assert result["decision_threshold"] == .5
    assert visited == ["recording_0.wav", "recording_1.wav"]
    assert "does not exclude" in result["limitations"][1]


@pytest.mark.parametrize("settings", [
    {"threshold": .4}, {"threshold": True},
    {"bootstrap_samples": 0}, {"bootstrap_samples": 5000},
    {"bootstrap_samples": True}, {"seed": -1}, {"seed": 2 ** 63},
])
def test_invalid_fixed_evaluation_settings_fail_before_model_loading(corpus, settings):
    root, manifest, original, _ = corpus
    with pytest.raises(ValueError):
        external.evaluate_external(root, manifest, original, root / "model", **settings)


def test_output_does_not_overwrite_a_previous_report(corpus, tmp_path):
    root, manifest, original, _ = corpus
    target = tmp_path / "external_report.json"
    target.write_text(json.dumps({"existing": "untouched"}))
    with pytest.raises(SystemExit) as err:
        external.main([str(root), str(manifest), "--training-manifest", str(original),
                       "--model-dir", str(root / "model"), "--output", str(target)])
    assert err.value.code == 2
    assert json.loads(target.read_text()) == {"existing": "untouched"}
