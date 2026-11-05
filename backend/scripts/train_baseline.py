"""Train an experimental audio-feature baseline using a labeled manifest.

Run from backend/:
  python -m pip install -r requirements-ml.txt
  python scripts/train_baseline.py DATASET_ROOT dataset_manifest.csv --output baseline_model.joblib

This baseline does not use RIR measurements or Audio Spectrogram Transformer.
Only use evaluation results on a speaker/source-independent held-out dataset.
"""

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Ensure backend/ is on PYTHONPATH when executed as a script from backend/.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.features import extract_features  # noqa: E402
from app.preprocessing import load_audio  # noqa: E402

FEATURE_NAMES = ('rms_amplitude', 'zero_crossing_rate', 'spectral_centroid_hz')


def load_manifest_features(dataset: Path, manifest: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    root = dataset.resolve()
    if not root.is_dir():
        raise ValueError('Dataset root does not exist.')
    grouped: dict[str, list[tuple[list[float], int]]] = {'train': [], 'val': [], 'test': []}
    seen: set[str] = set()
    with manifest.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if not {'path', 'label', 'split', 'sha256'}.issubset(reader.fieldnames or []):
            raise ValueError('Manifest is missing required columns.')
        for row in reader:
            split = row['split']
            if split not in grouped or row['label'] not in ('real', 'fake'):
                raise ValueError('Manifest has unsupported label or split.')
            if row['sha256'] in seen:
                raise ValueError('Manifest includes a duplicate audio hash.')
            seen.add(row['sha256'])
            audio_path = (root / row['path']).resolve()
            if not audio_path.is_relative_to(root) or not audio_path.is_file():
                raise ValueError('Manifest file is missing or outside dataset root.')
            import hashlib
            if hashlib.sha256(audio_path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError(f'Audio file content changed after manifest creation: {audio_path}')
            audio = load_audio(audio_path)
            f = extract_features(audio)
            grouped[split].append(([
                f.rms_amplitude, f.zero_crossing_rate, f.spectral_centroid_hz
            ], int(row['label'] == 'fake')))
    result = {}
    for split, values in grouped.items():
        if not values:
            raise ValueError(f'No recordings in {split} split.')
        result[split] = (np.asarray([v[0] for v in values]), np.asarray([v[1] for v in values]))
    return result


def train(dataset: Path, manifest: Path, output: Path) -> dict:
    dataset_splits = load_manifest_features(dataset, manifest)
    x_train, y_train = dataset_splits['train']
    if len(set(y_train.tolist())) < 2:
        raise ValueError('Both classes must be in the training split.')
    classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    classifier.fit(x_train, y_train)
    metrics: dict[str, dict] = {}
    for split in ('val', 'test'):
        inputs, truth = dataset_splits[split]
        guesses = classifier.predict(inputs)
        metrics[split] = {
            'samples': int(truth.size),
            'accuracy': float(accuracy_score(truth, guesses)),
            'precision_fake': float(precision_score(truth, guesses, zero_division=0)),
            'recall_fake': float(recall_score(truth, guesses, zero_division=0)),
            'f1_fake': float(f1_score(truth, guesses, zero_division=0)),
            'confusion_matrix_real_fake': confusion_matrix(truth, guesses, labels=[0, 1]).tolist(),
        }
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError('Model or metrics output already exists; choose a new path.')
    joblib.dump({'pipeline': classifier, 'features': FEATURE_NAMES, 'labels': ('real', 'fake')}, output)
    output.with_suffix('.json').write_text(json.dumps({
        'model_type': 'experimental_logistic_regression_audio_features',
        'warning': 'Not an RIR classifier; evaluate speaker/source separation before claiming validity.',
        'metrics': metrics,
    }, indent=2), encoding='utf-8')
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, default=Path('baseline_model.joblib'))
    args = parser.parse_args()
    print(json.dumps(train(args.dataset, args.manifest, args.output), indent=2))


if __name__ == '__main__':
    main()
