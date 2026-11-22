"""Offline AST classification metrics and validation-only threshold selection.

All labels are real=0, fake=1. Scores are uncalibrated model outputs.
Do not tune on test data or treat a selected validation score as unbiased.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _validated(labels, fake_scores):
    truth = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(fake_scores, dtype=np.float64)
    if (truth.ndim != 1 or scores.ndim != 1 or truth.size != scores.size
            or truth.size < 2 or set(truth.tolist()) != {0, 1}
            or not np.isfinite(scores).all()
            or np.any((scores < 0) | (scores > 1))):
        raise ValueError("Metrics require finite scores in [0, 1] and both real/fake labels.")
    return truth, scores


def _threshold(value):
    if not isinstance(value, (float, int)) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Decision threshold must be between 0 and 1.")
    return float(value)


def classification_metrics(labels, fake_scores, *, threshold=0.5):
    truth, scores = _validated(labels, fake_scores)
    chosen = _threshold(threshold)
    predictions = (scores >= chosen).astype(np.int64)
    matrix = confusion_matrix(truth, predictions, labels=[0, 1])
    tn, fp = [int(n) for n in matrix[0]]
    fn, tp = [int(n) for n in matrix[1]]
    return {
        "samples": int(truth.size),
        "real_count": int(np.sum(truth == 0)),
        "fake_count": int(np.sum(truth == 1)),
        "threshold": chosen,
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "precision_fake": float(precision_score(truth, predictions, zero_division=0)),
        "recall_fake": float(recall_score(truth, predictions, zero_division=0)),
        "f1_fake": float(f1_score(truth, predictions, zero_division=0)),
        "real_specificity": tn / (tn + fp),
        "real_false_positive_rate": fp / (tn + fp),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "average_precision_fake": float(average_precision_score(truth, scores)),
        "predicted_fake": int(np.sum(predictions)),
        "confusion_matrix_real_fake": matrix.tolist(),
    }


def select_validation_threshold(labels, fake_scores, *, objective="balanced_accuracy"):
    """Search unique validation scores; ties favor fewer real false positives.

    This selection is for *research*, never automatic live API calibration.
    The chosen validation metric is selection-biased and not a test result.
    """
    truth, scores = _validated(labels, fake_scores)
    if objective not in ("balanced_accuracy", "f1_fake"):
        raise ValueError("Unsupported threshold objective.")
    candidates = sorted({0.0, 0.5, 1.0, *(float(x) for x in scores)})
    results = [classification_metrics(truth, scores, threshold=t) for t in candidates]
    best = max(
        results,
        key=lambda item: (
            item[objective],
            -item["confusion_matrix_real_fake"][0][1],
            item["threshold"],
        ),
    )
    return {
        "objective": objective,
        "threshold": best["threshold"],
        "metrics_on_selection_data": best,
        "warning": (
            "Threshold selected on the validation split. Its validation metric "
            "is optimistic; do not interpret it as independent test accuracy. "
            "No deployed prediction threshold has been changed."
        ),
    }


def bootstrap_binary_uncertainty(labels, fake_scores, *, threshold=0.5,
                                 samples=500, seed=42):
    """Stratified 95% bootstrap intervals for a *fixed* decision threshold.

    Each class is resampled separately so imbalanced/small samples retain
    real and fake observations. Assumes independent examples *within class*;
    source/room-correlated recordings can make these intervals optimistic.
    Do not tune a threshold on test scores.
    """
    truth, scores = _validated(labels, fake_scores)
    chosen = _threshold(threshold)
    if type(samples) is not int or not 20 <= samples <= 2000:
        raise ValueError("Bootstrap samples must be an integer between 20 and 2000.")
    if type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise ValueError("Bootstrap seed must be a nonnegative bounded integer.")

    real_idx = np.flatnonzero(truth == 0)
    fake_idx = np.flatnonzero(truth == 1)
    rng = np.random.default_rng(seed)
    measured = ("balanced_accuracy", "real_false_positive_rate",
                "recall_fake", "f1_fake", "roc_auc")
    estimates = classification_metrics(truth, scores, threshold=chosen)
    resampled = {name: np.empty(samples, dtype=np.float64) for name in measured}
    for i in range(samples):
        selection = np.concatenate((
            rng.choice(real_idx, size=real_idx.size, replace=True),
            rng.choice(fake_idx, size=fake_idx.size, replace=True),
        ))
        metrics = classification_metrics(truth[selection], scores[selection],
                                         threshold=chosen)
        for name in measured:
            resampled[name][i] = metrics[name]
    return {
        "method": "stratified_nonparametric_bootstrap",
        "confidence_level": 0.95,
        "bootstrap_samples": samples,
        "seed": seed,
        "fixed_threshold": chosen,
        "real_count": int(real_idx.size),
        "fake_count": int(fake_idx.size),
        "metrics": {
            name: {
                "point_estimate": estimates[name],
                "ci95_lower": float(np.percentile(values, 2.5)),
                "ci95_upper": float(np.percentile(values, 97.5)),
            }
            for name, values in resampled.items()
        },
        "warning": (
            "Intervals assume independently sampled examples within each class, "
            "so speaker/room/source correlations and dataset shift are not "
            "accounted for. Bootstrap intervals do not validate generalization "
            "or calibrate fake-class scores. Selecting the threshold on this "
            "same dataset makes estimates optimistic."
        ),
    }
