"""Ensure optional AST uncertainty reports cannot silently claim stronger evidence."""
import json

import pytest

from scripts.evaluate_ast import evaluate_checkpoint, main, reported_training_mode


def test_checkpoint_training_metadata_unknown_without_proven_local_record(tmp_path):
    assert reported_training_mode(tmp_path) == "unknown"
    (tmp_path / "training_meta.json").write_text(
        json.dumps({"full_finetune": False}), encoding="utf-8"
    )
    assert reported_training_mode(tmp_path) == "head_only"
    (tmp_path / "training_meta.json").write_text(
        json.dumps({"full_finetune": True}), encoding="utf-8"
    )
    assert reported_training_mode(tmp_path) == "full_encoder_finetune"


@pytest.mark.parametrize("content", [
    '{"full_finetune": "false"}', '{"full_finetune": 1}',
    '{"source": "unknown"}', '{broken json',
])
def test_checkpoint_training_mode_does_not_infer_from_bad_metadata(tmp_path, content):
    (tmp_path / "training_meta.json").write_text(content, encoding="utf-8")
    assert reported_training_mode(tmp_path) == "unknown"


@pytest.mark.parametrize("samples", [-1, 1, 19, 2001, 1.2, True])
def test_ast_evaluation_rejects_invalid_bootstrap_before_model_load(samples):
    with pytest.raises(ValueError, match="bootstrap_samples"):
        evaluate_checkpoint("unused", "unused", "unused", bootstrap_samples=samples)


@pytest.mark.parametrize("seed", [-1, 2 ** 63, 1.2, True])
def test_ast_evaluation_rejects_invalid_random_seed_before_model_load(seed):
    with pytest.raises(ValueError, match="bootstrap_seed"):
        evaluate_checkpoint("unused", "unused", "unused", bootstrap_seed=seed)


def test_ast_cli_rejects_bad_bootstrap_options_before_expensive_model_load():
    with pytest.raises(SystemExit) as exc:
        main(["dataset", "manifest", "--bootstrap-samples", "19"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(["dataset", "manifest", "--bootstrap-seed", "-1"])
    assert exc.value.code == 2


def test_ast_cli_still_forbids_test_threshold_selection_with_bootstrap():
    with pytest.raises(SystemExit) as exc:
        main([
            "dataset", "manifest", "--split", "test",
            "--select-threshold", "--threshold", "0.5",
            "--bootstrap-samples", "20",
        ])
    assert exc.value.code == 2
