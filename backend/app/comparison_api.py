"""Read a local, self-generated acoustic feature ablation report; never execute model files."""

import json
import os
import math
from pathlib import Path

from app.metric_validation import scores_match_matrix

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix='/api/model', tags=['experimental-model'])
REPORT_PATH = Path(__file__).resolve().parents[1] / 'models' / 'acoustic_comparison.json'
FEATURE_MODELS = ('baseline_three_features', 'speech_acoustic_six_features')
SPLITS = ('val', 'test')
SCORES = ('accuracy', 'precision_fake', 'recall_fake', 'f1_fake')


def read_comparison(path: Path | None = None) -> dict | None:
    """Ignore missing/malformed/oversized reports rather than displaying misleading metrics."""
    path = path if path is not None else Path(os.getenv('ACOUSTICSPACE_COMPARISON_REPORT', str(REPORT_PATH)))
    try:
        if not path.is_file() or path.stat().st_size > 64 * 1024:
            return None
        with path.open('r', encoding='utf-8') as stream:
            data = json.load(stream)
    except (OSError, UnicodeError, ValueError):
        return None
    if not isinstance(data, dict) or data.get('experiment') != 'speech_acoustics_feature_ablation_v1':
        return None
    models = data.get('models')
    recordings = data.get('recordings')
    if not isinstance(models, dict) or not isinstance(recordings, dict):
        return None
    if any(type(recordings.get(split)) is not int or recordings[split] < 1 for split in ('train', 'val', 'test')):
        return None
    cleaned = {}
    for name in FEATURE_MODELS:
        item = models.get(name)
        if not isinstance(item, dict) or not isinstance(item.get('metrics'), dict):
            return None
        metrics = {}
        for split in SPLITS:
            entry = item['metrics'].get(split)
            if not isinstance(entry, dict) or type(entry.get('samples')) is not int or entry['samples'] != recordings[split]:
                return None
            scores = {}
            for score in SCORES:
                val = entry.get(score)
                if type(val) not in (float, int) or not math.isfinite(val) or not 0 <= val <= 1:
                    return None
                scores[score] = float(val)
            matrix = entry.get('confusion_matrix_real_fake')
            if (not isinstance(matrix, list) or len(matrix) != 2 or
                any(not isinstance(row, list) or len(row) != 2 or
                    any(type(cell) is not int or cell < 0 for cell in row) for row in matrix) or
                sum(sum(row) for row in matrix) != recordings[split]):
                return None
            if not scores_match_matrix(scores, matrix):
                return None
            metrics[split] = {'samples': recordings[split], **scores, 'confusion_matrix_real_fake': matrix}
        cleaned[name] = {'metrics': metrics}
    return {
        'experiment': data['experiment'],
        'recordings': {split: recordings[split] for split in ('train', 'val', 'test')},
        'warning': ('Research comparison from locally trained models. Neither feature set is a measured RIR; '
                    'the official ASVspoof evaluation partition was not used.'),
        'models': cleaned,
    }


@router.get('/comparison')
def get_comparison() -> dict:
    result = read_comparison()
    if result is None:
        raise HTTPException(status_code=404, detail='No valid acoustic comparison report installed yet.')
    return result
