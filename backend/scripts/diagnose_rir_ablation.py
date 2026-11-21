r"""Read-only RIR feature ablation using train and validation splits only.

Windows CMD, from the AcousticSpace backend directory:
    .venv\Scripts\python.exe scripts\diagnose_rir_ablation.py datasets\rir_lab_measured_v1

This script deliberately does not load the held-out test audio, save a model,
or modify existing artifacts. Results are exploratory validation diagnostics,
not estimates of deployable deepfake-detection accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.rir_consistency_dataset import extract_controlled_arrays  # noqa: E402
from app.rir_features import FEATURE_NAMES  # noqa: E402


CANDIDATES = {
    "original_all_features": tuple(FEATURE_NAMES),
    "rt60_c50_late_energy": (
        "rt60_difference_seconds",
        "c50_difference_db",
        "late_energy_difference",
    ),
    "rt60_c50_only": (
        "rt60_difference_seconds",
        "c50_difference_db",
    ),
}


def run(dataset_root: Path) -> dict:
    # Existing loader decodes *only* train/val feature audio by default.
    splits = extract_controlled_arrays(dataset_root, include_test=False)
    x_train, y_train = splits["train"]
    x_val, y_val = splits["val"]
    expected_columns = len(FEATURE_NAMES)
    if (x_train.ndim != 2 or x_val.ndim != 2
            or x_train.shape[1] != expected_columns
            or x_val.shape[1] != expected_columns
            or set(y_train.tolist()) != {0, 1}
            or set(y_val.tolist()) != {0, 1}
            or not np.isfinite(x_train).all()
            or not np.isfinite(x_val).all()):
        raise ValueError("Invalid train/validation feature arrays.")

    models = {}
    for name, feature_names in CANDIDATES.items():
        indices = [FEATURE_NAMES.index(feature) for feature in feature_names]
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        )
        classifier.fit(x_train[:, indices], y_train)
        scores = classifier.predict_proba(x_val[:, indices])[:, 1]
        predicted = (scores >= 0.5).astype(int)
        models[name] = {
            "features": list(feature_names),
            "validation_accuracy": float(accuracy_score(y_val, predicted)),
            "validation_balanced_accuracy": float(balanced_accuracy_score(y_val, predicted)),
            "validation_f1_mismatched": float(f1_score(y_val, predicted, zero_division=0)),
            "validation_roc_auc": float(roc_auc_score(y_val, scores)),
            "confusion_matrix_consistent_mismatched": confusion_matrix(
                y_val, predicted, labels=[0, 1]
            ).tolist(),
        }

    per_feature = {}
    for idx, feature in enumerate(FEATURE_NAMES):
        train_values = x_train[:, idx]
        val_values = x_val[:, idx]
        if (np.unique(train_values).size < 2 or np.unique(val_values).size < 2):
            per_feature[feature] = {"validation_auc_higher_means_mismatch": None}
            continue
        per_feature[feature] = {
            "train_median_consistent": float(np.median(train_values[y_train == 0])),
            "train_median_mismatched": float(np.median(train_values[y_train == 1])),
            "val_median_consistent": float(np.median(val_values[y_val == 0])),
            "val_median_mismatched": float(np.median(val_values[y_val == 1])),
            "validation_auc_higher_means_mismatch": float(
                roc_auc_score(y_val, val_values)
            ),
        }

    return {
        "dataset_root": str(dataset_root.resolve()),
        "train_samples": int(y_train.size),
        "validation_samples": int(y_val.size),
        "test_split_evaluated": False,
        "feature_diagnostics": per_feature,
        "candidate_models": models,
        "warning": (
            "Exploratory comparison using train and validation only. "
            "The previously evaluated test split must not be reused for tuning. "
            "Future performance claims require a new independent held-out dataset; "
            "this is acoustic consistency under oracle-reference assumptions, "
            "not deepfake classification."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    try:
        report = run(args.dataset_root)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Ablation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())