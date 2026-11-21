r"""Train an exploratory reference-aided room-consistency classifier.

Requires the controlled dataset built by build_rir_dataset.py, including
isolated wet stems and dry source references. Ground-truth room ID/RIR files
are used only to verify manifest integrity, never as model features.

From backend in Windows CMD:
  .venv\Scripts\python.exe scripts\train_rir_detector.py datasets\rir_lab_v1 --output models\rir_consistency_toy.joblib

Training and validation use different speakers and rooms. Test evaluation
is explicitly opt-in; never use a test result for model/threshold selection.
This model is NOT a speech deepfake detector or deployable blind-RIR system.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.rir_consistency_dataset import extract_controlled_arrays  # noqa: E402
from app.rir_features import FEATURE_NAMES  # noqa: E402

MODEL_TYPE = "reference_aided_rir_consistency_logistic_experiment"


def evaluate(truth: np.ndarray, mismatch_score: np.ndarray) -> dict:
    """Class 0=consistent, class 1=mismatched; fixed uncalibrated 0.5 threshold."""
    if (truth.ndim != 1 or mismatch_score.shape != truth.shape
            or set(truth.tolist()) != {0, 1}
            or not np.isfinite(mismatch_score).all()
            or np.any((mismatch_score < 0) | (mismatch_score > 1))):
        raise ValueError("Invalid evaluation labels or mismatch-class scores.")
    predicted = (mismatch_score >= 0.5).astype(np.int64)
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    consistent_ok, consistent_wrong = (int(x) for x in matrix[0])
    mismatch_missed, mismatch_found = (int(x) for x in matrix[1])
    return {
        "samples": int(truth.size),
        "consistent_count": int(np.sum(truth == 0)),
        "mismatched_count": int(np.sum(truth == 1)),
        "decision_threshold": 0.5,
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "precision_mismatched": float(precision_score(truth, predicted, zero_division=0)),
        "recall_mismatched": float(recall_score(truth, predicted, zero_division=0)),
        "f1_mismatched": float(f1_score(truth, predicted, zero_division=0)),
        "false_mismatch_rate": consistent_wrong / (consistent_ok + consistent_wrong),
        "missed_mismatch_rate": mismatch_missed / (mismatch_missed + mismatch_found),
        "roc_auc": float(roc_auc_score(truth, mismatch_score)),
        "average_precision_mismatched": float(average_precision_score(truth, mismatch_score)),
        "confusion_matrix_consistent_mismatched": matrix.tolist(),
    }


def train(root: Path, output: Path, *, evaluate_test: bool = False) -> dict:
    output = output.resolve()
    report_file = output.with_suffix(".json")
    if output.exists() or report_file.exists():
        raise FileExistsError("RIR model/report exists; use a new --output path.")
    splits = extract_controlled_arrays(root)
    x_train, y_train = splits["train"]
    if x_train.ndim != 2 or x_train.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Invalid RIR training features.")
    if set(y_train.tolist()) != {0, 1}:
        raise ValueError("Both consistency classes are required in the training split.")
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    classifier.fit(x_train, y_train)
    metrics = {}
    for split in (("val", "test") if evaluate_test else ("val",)):
        samples, truth = splits[split]
        scores = classifier.predict_proba(samples)[:, 1]
        metrics[split] = evaluate(truth, scores)
    report = {
        "model_type": MODEL_TYPE,
        "features": list(FEATURE_NAMES),
        "label_map": {"consistent": 0, "mismatched": 1},
        "sample_counts": {split: int(len(splits[split][1])) for split in splits},
        "metrics": metrics,
        "warning": (
            "REFERENCE-AIDED, controlled synthetic room-consistency experiment. "
            "Requires isolated stems and dry sources, which ordinary speech uploads do not provide. "
            "Labels do not mean real/fake. The tiny toy dataset is a pipeline smoke test "
            "and cannot establish reliability, generalization or forensic accuracy. "
            "Scores are not calibrated. Test split must not guide tuning."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": classifier, "features": FEATURE_NAMES,
                 "labels": ("consistent", "mismatched"),
                 "model_type": MODEL_TYPE}, output)
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Only use once model and threshold are frozen; do not tune on test.")
    args = parser.parse_args(argv)
    try:
        report = train(args.dataset_root, args.output, evaluate_test=args.evaluate_test)
    except (OSError, ValueError) as exc:
        print(f"RIR training failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
