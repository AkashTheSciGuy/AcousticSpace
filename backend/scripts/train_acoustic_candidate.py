"""Train the separate six-feature speech-acoustic candidate; preserve baseline.

From backend/ (Windows CMD):
 .venv\\Scripts\\python.exe scripts\\train_acoustic_candidate.py DATASET_ROOT MANIFEST_CSV

This writes models/acoustic_candidate.joblib only; it never changes the baseline.
Only run with locally trusted, validated dataset recordings.
"""
import argparse
import sys
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_acoustic_models import BASE_FEATURES, EXTRA_FEATURES, extract_split_arrays  # noqa: E402


def train(dataset: Path, manifest: Path, output: Path) -> dict:
    """Validate input and train with the exact same feature ordering as inference."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f'Model already exists: {output}; refusing to overwrite.')
    splits = extract_split_arrays(dataset, manifest)
    features, labels = splits['train']
    if set(labels.tolist()) != {0, 1}:
        raise ValueError('Training set must include both real and fake classes.')
    model = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42,
    ))
    model.fit(features, labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'pipeline': model, 'features': BASE_FEATURES + EXTRA_FEATURES,
                 'labels': ('real', 'fake')}, output)
    return {'saved_to': str(output), 'train_samples': int(labels.size),
            'feature_count': len(BASE_FEATURES + EXTRA_FEATURES),
            'warning': 'Experimental speech-acoustic model, not RIR or forensic evidence.'}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, default=Path('models/acoustic_candidate.joblib'))
    args = parser.parse_args(argv)
    try:
        print(train(args.dataset, args.manifest, args.output))
    except (ValueError, OSError) as error:
        print(f'Candidate training failed: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
