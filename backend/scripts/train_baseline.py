"""Train the experimental three-feature baseline on a verified grouped manifest.

Run from backend/:
    python scripts/train_baseline.py DATASET_ROOT manifest.csv --output models/baseline_model.joblib

This model uses basic audio features, not RIR. Metrics are research-only.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.features import extract_features  # noqa: E402
from app.manifest_validation import validate_training_manifest  # noqa: E402
from app.preprocessing import load_audio  # noqa: E402

FEATURE_NAMES = ('rms_amplitude', 'zero_crossing_rate', 'spectral_centroid_hz')


def load_manifest_features(dataset: Path, manifest: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Validate the full manifest before decoding and extracting features."""
    validate_training_manifest(dataset, manifest)
    root = dataset.resolve()
    grouped: dict[str, list[tuple[list[float], int]]] = {'train': [], 'val': [], 'test': []}
    with manifest.open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            audio_path = (root / row['path']).resolve()
            audio = load_audio(audio_path)
            features = extract_features(audio)
            grouped[row['split']].append(([
                features.rms_amplitude,
                features.zero_crossing_rate,
                features.spectral_centroid_hz,
            ], int(row['label'] == 'fake')))
    return {
        split: (np.asarray([item[0] for item in values], dtype=np.float64),
                np.asarray([item[1] for item in values], dtype=np.int64))
        for split, values in grouped.items()
    }


def train(dataset: Path, manifest: Path, output: Path) -> dict:
    """Train only on grouped splits, then save the model and descriptive metrics."""
    output = output.resolve()
    report = output.with_suffix('.json')
    if output.exists() or report.exists():
        raise FileExistsError('Model or metrics output already exists; choose a new path.')
    splits = load_manifest_features(dataset, manifest)
    x_train, y_train = splits['train']
    if len(set(y_train.tolist())) < 2:
        raise ValueError('Both classes must be in the training split.')
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    )
    classifier.fit(x_train, y_train)
    metrics: dict[str, dict] = {}
    for split in ('val', 'test'):
        inputs, truth = splits[split]
        guesses = classifier.predict(inputs)
        metrics[split] = {
            'samples': int(truth.size),
            'accuracy': float(accuracy_score(truth, guesses)),
            'precision_fake': float(precision_score(truth, guesses, zero_division=0)),
            'recall_fake': float(recall_score(truth, guesses, zero_division=0)),
            'f1_fake': float(f1_score(truth, guesses, zero_division=0)),
            'confusion_matrix_real_fake': confusion_matrix(truth, guesses, labels=[0, 1]).tolist(),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'pipeline': classifier, 'features': FEATURE_NAMES,
                 'labels': ('real', 'fake')}, output)
    report.write_text(json.dumps({
        'model_type': 'experimental_logistic_regression_audio_features',
        'warning': 'Not RIR-based; grouping was checked but source metadata and external generalization are unverified.',
        'metrics': metrics,
    }, indent=2), encoding='utf-8')
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, default=Path('models/baseline_model.joblib'))
    args = parser.parse_args()
    print(json.dumps(train(args.dataset, args.manifest, args.output), indent=2))


if __name__ == '__main__':
    main()
