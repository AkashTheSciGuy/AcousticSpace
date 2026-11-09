"""Read training-produced evaluation summaries without modifying model artifacts.

The adjacent .json file is created by scripts/train_baseline.py. These metrics
are descriptive experiment results, not a guarantee of generalization.
"""

import json
import math
from pathlib import Path

EXPECTED_MODEL_TYPE = 'experimental_logistic_regression_audio_features'
SPLITS = ('val', 'test')
SCORES = ('accuracy', 'precision_fake', 'recall_fake', 'f1_fake')


def read_evaluation(model_file: Path) -> dict | None:
    """Return a validated, JSON-compatible summary, or None if unavailable.

    Malformed or missing sidecar reports are treated as absent rather than
    preventing valid model inference. Never trust them as independent audits.
    """
    sidecar = model_file.with_suffix('.json')
    try:
        if not sidecar.is_file() or sidecar.stat().st_size > 64 * 1024:
            return None
        with sidecar.open('r', encoding='utf-8') as stream:
            report = json.load(stream)
    except (OSError, UnicodeError, ValueError):
        return None
    if not isinstance(report, dict) or report.get('model_type') != EXPECTED_MODEL_TYPE:
        return None
    metrics = report.get('metrics')
    if not isinstance(metrics, dict):
        return None

    cleaned = {}
    for split in SPLITS:
        item = metrics.get(split)
        if not isinstance(item, dict):
            return None
        n = item.get('samples')
        if type(n) is not int or n < 1:
            return None
        clean = {'samples': n}
        for key in SCORES:
            value = item.get(key)
            if type(value) not in (float, int) or not math.isfinite(value) or not (0 <= value <= 1):
                return None
            clean[key] = float(value)
        matrix = item.get('confusion_matrix_real_fake')
        if (not isinstance(matrix, list) or len(matrix) != 2 or
            any(not isinstance(row, list) or len(row) != 2 or
                any(type(cell) is not int or cell < 0 for cell in row) for row in matrix)):
            return None
        if sum(sum(row) for row in matrix) != n:
            return None
        clean['confusion_matrix_real_fake'] = matrix
        cleaned[split] = clean
    return {
        'source': 'local_training_report',
        'warning': 'Self-reported split metrics. Speaker/source independence has not been verified by this endpoint.',
        'metrics': cleaned,
    }
