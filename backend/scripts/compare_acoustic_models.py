"""Compare the current three-feature baseline to a six-feature speech-acoustic candidate.

Run from backend/:
    .venv\\Scripts\\python.exe scripts\\compare_acoustic_models.py DATASET_ROOT MANIFEST_CSV \\
        --output models\\acoustic_comparison.json

Research experiment only: no RIR is estimated, and no deployed model is replaced.
The official ASVspoof evaluation partition must remain untouched.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.features import extract_features  # noqa: E402
from app.manifest_validation import validate_training_manifest  # noqa: E402
from app.preprocessing import load_audio  # noqa: E402
from app.speech_acoustics import extract_speech_acoustics  # noqa: E402

BASE_FEATURES = ('rms_amplitude', 'zero_crossing_rate', 'spectral_centroid_hz')
EXTRA_FEATURES = ('mean_spectral_flatness', 'rms_coefficient_of_variation', 'mean_positive_spectral_flux')
SPLITS = ('train', 'val', 'test')


def extract_split_arrays(dataset_root: Path, manifest: Path):
    """Verify entire manifest before decoding; use identical examples for both models."""
    validate_training_manifest(dataset_root, manifest)
    root = dataset_root.resolve()
    examples = {split: [] for split in SPLITS}
    with manifest.open('r', newline='', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            audio = load_audio((root / row['path'].strip()).resolve())
            base = extract_features(audio)
            acoustics = extract_speech_acoustics(audio)
            values = [float(getattr(base, name)) for name in BASE_FEATURES]
            values.extend(float(getattr(acoustics, name)) for name in EXTRA_FEATURES)
            if not np.isfinite(values).all():
                raise ValueError(f'Non-finite audio features in {row["path"]}')
            examples[row['split'].strip()].append((values, int(row['label'].strip() == 'fake')))
    return {
        split: (
            np.asarray([item[0] for item in entries], dtype=np.float64),
            np.asarray([item[1] for item in entries], dtype=np.int64),
        )
        for split, entries in examples.items()
    }


def evaluate(truth: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        'samples': int(truth.size),
        'accuracy': float(accuracy_score(truth, predictions)),
        'precision_fake': float(precision_score(truth, predictions, zero_division=0)),
        'recall_fake': float(recall_score(truth, predictions, zero_division=0)),
        'f1_fake': float(f1_score(truth, predictions, zero_division=0)),
        'confusion_matrix_real_fake': confusion_matrix(truth, predictions, labels=[0, 1]).tolist(),
    }


def compare(dataset_root: Path, manifest: Path) -> dict:
    arrays = extract_split_arrays(dataset_root, manifest)
    x_train, y_train = arrays['train']
    if len(set(y_train.tolist())) != 2:
        raise ValueError('Training split must contain both real and fake recordings.')
    results = {}
    for model_name, width, features in (
        ('baseline_three_features', 3, BASE_FEATURES),
        ('speech_acoustic_six_features', 6, BASE_FEATURES + EXTRA_FEATURES),
    ):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        )
        model.fit(x_train[:, :width], y_train)
        metrics = {}
        for split in ('val', 'test'):
            inputs, labels = arrays[split]
            metrics[split] = evaluate(labels, model.predict(inputs[:, :width]))
        results[model_name] = {'features': list(features), 'metrics': metrics}
    return {
        'experiment': 'speech_acoustics_feature_ablation_v1',
        'warning': ('Exploratory within-dataset comparison only. Speech-acoustic features are NOT measured RIR; '
                    'official evaluation partition not used. Repeated test-based iteration can bias results.'),
        'recordings': {name: int(arrays[name][1].size) for name in SPLITS},
        'models': results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_root', type=Path)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, default=Path('models/acoustic_comparison.json'))
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f'Report already exists: {args.output}; choose a new filename.')
        report = compare(args.dataset_root, args.manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2)
        print(json.dumps(report, indent=2))
        print(f'Comparison saved to {args.output}')
        return 0
    except (ValueError, OSError, csv.Error) as error:
        print(f'Comparison failed: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
