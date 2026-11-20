"""Create synthetic browser-test inputs. NEVER use these models for research.

Run with --output pointing to an ignored temporary directory. Outputs are
strictly integration fixtures, not the user's trained ASVspoof models.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.model_inference import EXPECTED_FEATURES, EXTRA_FEATURES


def create(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    t = np.arange(32000) / 16000
    sf.write(output / 'tone.wav', 0.2 * np.sin(2 * np.pi * 440 * t), 16000)
    rir = rng.normal(size=t.size) * np.exp(-np.log(1000) * t / 0.5)
    rir[0] = 3
    sf.write(output / 'rir.wav', rir / 4, 16000)
    for name, features in [('baseline', EXPECTED_FEATURES), ('acoustic', EXPECTED_FEATURES + EXTRA_FEATURES)]:
        model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
        model.fit(rng.normal(size=(20, len(features))), [0, 1] * 10)
        joblib.dump({'pipeline': model, 'features': features, 'labels': ('real', 'fake')}, output / f'{name}.joblib')
    metric = {'samples': 2, 'accuracy': 0.5, 'precision_fake': 0.0,
              'recall_fake': 0.0, 'f1_fake': 0.0, 'confusion_matrix_real_fake': [[1, 0], [1, 0]]}
    # Made-up fixture counts only: never a research evaluation.
    (output / 'baseline.json').write_text(json.dumps({
        'model_type': 'experimental_logistic_regression_audio_features',
        'metrics': {'val': metric, 'test': metric},
    }))
    (output / 'comparison.json').write_text(json.dumps({
        'experiment': 'speech_acoustics_feature_ablation_v1',
        'recordings': {'train': 20, 'val': 2, 'test': 2},
        'models': {name: {'metrics': {'val': metric, 'test': metric}}
                   for name in ('baseline_three_features', 'speech_acoustic_six_features')},
    }))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    create(parser.parse_args().output)
