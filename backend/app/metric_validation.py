"""Check saved classification scores against their confusion matrix."""
import math


def scores_match_matrix(scores: dict, matrix: list[list[int]]) -> bool:
    """Accept rounded reports, rejecting contradictory numerical claims.

    Rows are actual real/fake, columns are predicted real/fake. Undefined
    precision/recall/F1 use the training scripts' zero_division=0 convention.
    This checks internal consistency, never dataset provenance or accuracy.
    """
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    total = tn + fp + fn + tp
    expected = {
        'accuracy': (tn + tp) / total,
        'precision_fake': tp / (tp + fp) if tp + fp else 0.0,
        'recall_fake': tp / (tp + fn) if tp + fn else 0.0,
        'f1_fake': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }
    return all(math.isclose(scores[key], value, rel_tol=0, abs_tol=1e-4)
               for key, value in expected.items())
