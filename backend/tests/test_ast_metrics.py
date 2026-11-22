"""Offline AST metrics: correct class labels, imbalanced data, no test tuning."""
import numpy as np
import pytest

from app.ast_metrics import (bootstrap_binary_uncertainty, classification_metrics,
                             select_validation_threshold)
from scripts.evaluate_ast import evaluate_checkpoint, main


def test_metrics_confusion_false_positives_and_auc():
    truth = [0, 0, 0, 1, 1]
    scores = [.2, .8, .9, .7, .95]
    result = classification_metrics(truth, scores)
    assert result["confusion_matrix_real_fake"] == [[1, 2], [0, 2]]
    assert result["real_false_positive_rate"] == pytest.approx(2 / 3)
    assert result["recall_fake"] == 1.0
    assert result["balanced_accuracy"] == pytest.approx((1 / 3 + 1) / 2)
    assert 0 <= result["roc_auc"] <= 1
    assert 0 <= result["average_precision_fake"] <= 1


def test_threshold_selection_prefers_lower_false_positive_rate():
    truth = [0, 0, 0, 1, 1]
    scores = [.25, .45, .52, .7, .9]
    selected = select_validation_threshold(truth, scores)
    assert selected["objective"] == "balanced_accuracy"
    assert selected["threshold"] == pytest.approx(.7)
    assert selected["metrics_on_selection_data"]["confusion_matrix_real_fake"] == [[3, 0], [0, 2]]
    assert "validation" in selected["warning"]


def test_constant_scores_and_majority_baseline_are_well_defined():
    result = classification_metrics([0, 0, 0, 1], [.4, .4, .4, .4])
    assert result["accuracy"] == .75
    assert result["balanced_accuracy"] == .5
    assert result["roc_auc"] == .5
    assert select_validation_threshold([0, 0, 0, 1], [.4] * 4)["threshold"] > .4


@pytest.mark.parametrize("truth,scores", [
    ([0, 0], [.1, .2]), ([0, 1], [.1, np.nan]), ([0, 1], [.1, 1.1]),
    ([0, 1], [.4]), ([0, 1], [[.2], [.4]]),
])
def test_invalid_inputs_fail_closed(truth, scores):
    with pytest.raises(ValueError):
        classification_metrics(truth, scores)


def test_test_split_never_selects_threshold():
    with pytest.raises(ValueError, match="only on validation"):
        evaluate_checkpoint("unused", "unused", "unused", split="test", select_threshold=True)


def test_cli_requires_precommitted_test_threshold():
    with pytest.raises(SystemExit) as error:
        main(["unused", "unused", "--split", "test"])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        main(["unused", "unused", "--split", "test", "--select-threshold"])
    assert error.value.code == 2

def test_existing_output_rejected_before_model_inference(tmp_path, monkeypatch, capsys):
    from scripts import evaluate_ast
    report = tmp_path / "existing.json"
    report.write_text('{"sentinel": true}', encoding="utf-8")

    def forbidden_inference(*_args, **_kwargs):
        raise AssertionError("Output preflight must run before expensive model inference")

    monkeypatch.setattr(evaluate_ast, "evaluate_checkpoint", forbidden_inference)
    with pytest.raises(SystemExit) as error:
        evaluate_ast.main(["dataset", "manifest", "--output", str(report)])
    assert error.value.code == 2
    assert "Output report already exists" in capsys.readouterr().err
    assert report.read_text(encoding="utf-8") == '{"sentinel": true}'


def test_bootstrap_intervals_are_seeded_bounded_and_class_balanced():
    labels = [0, 0, 0, 0, 1, 1]
    scores = [.1, .2, .3, .8, .6, .9]
    a = bootstrap_binary_uncertainty(labels, scores, samples=30, seed=9)
    b = bootstrap_binary_uncertainty(labels, scores, samples=30, seed=9)
    assert a == b
    assert a["confidence_level"] == .95
    assert a["real_count"] == 4
    assert a["fake_count"] == 2
    assert a["fixed_threshold"] == .5
    for values in a["metrics"].values():
        assert 0 <= values["ci95_lower"] <= values["ci95_upper"] <= 1
        assert 0 <= values["point_estimate"] <= 1
    assert "speaker/room/source correlations" in a["warning"]


def test_bootstrap_constant_fake_scores_does_not_fail_when_classes_are_retained():
    report = bootstrap_binary_uncertainty([0, 0, 1], [.4, .4, .4], samples=20)
    assert report["metrics"]["roc_auc"]["ci95_lower"] == pytest.approx(.5)
    assert report["metrics"]["roc_auc"]["ci95_upper"] == pytest.approx(.5)


@pytest.mark.parametrize("samples", [-1, 0, 1, 19, 2001, 5.3, "20", True])
def test_bootstrap_rejects_invalid_sample_count(samples):
    with pytest.raises(ValueError, match="Bootstrap samples"):
        bootstrap_binary_uncertainty([0, 1], [.2, .8], samples=samples)


@pytest.mark.parametrize("seed", [-1, 1.5, "42", True, 2 ** 63])
def test_bootstrap_rejects_invalid_seed(seed):
    with pytest.raises(ValueError, match="Bootstrap seed"):
        bootstrap_binary_uncertainty([0, 1], [.2, .8], samples=20, seed=seed)
