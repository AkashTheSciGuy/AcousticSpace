"""Ground-truth event/interval metrics for AcousticSpace exploratory outputs.

These functions compare predictions with independently annotated timestamps.
They do NOT establish that the annotations are correct or that source/room
identities are independent.
"""
from __future__ import annotations

import math
from typing import Any


def _number(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Intervals and event times must be finite numbers.")
    return float(value)


def validate_intervals(raw: Any, duration: float, *, max_events: int = 400) -> list[tuple[float, float]]:
    """Validate intervals, require sorted non-overlap and clip bounds."""
    duration = _number(duration)
    if not 0 < duration <= 60.001:
        raise ValueError("Recording duration must be between 0 and 60 seconds.")
    if not isinstance(raw, list) or len(raw) > max_events:
        raise ValueError("Annotations must be a bounded list of intervals.")
    intervals: list[tuple[float, float]] = []
    last_end = 0.0
    for index, interval in enumerate(raw):
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError(f"Interval {index} must contain start and end.")
        start, end = (_number(item) for item in interval)
        if start < 0 or end <= start or end > duration + 1e-6:
            raise ValueError(f"Interval {index} has invalid recording bounds.")
        if start < last_end - 1e-9:
            raise ValueError("Intervals must be sorted, non-overlapping.")
        intervals.append((start, min(end, duration)))
        last_end = end
    return intervals


def union_intervals(raw: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Collapse overlapping prediction windows before measuring coverage."""
    ordered = sorted(raw)
    merged: list[tuple[float, float]] = []
    for start, end in ordered:
        if not (math.isfinite(start) and math.isfinite(end)) or start < 0 or end <= start:
            raise ValueError("Predicted interval bounds are invalid.")
        if merged and start <= merged[-1][1]:
            old_start, old_end = merged[-1]
            merged[-1] = (old_start, max(old_end, end))
        else:
            merged.append((start, end))
    return merged


def interval_coverage(
    prediction: list[tuple[float, float]],
    reference: list[tuple[float, float]],
    duration: float,
) -> dict:
    """Return duration-weighted coverage, not a clip-level deepfake verdict.

    Precision/recall are None when their denominator is zero. Silence and
    negative-only cases cannot be misrepresented as perfect recall.
    """
    duration = _number(duration)
    truth = validate_intervals([list(x) for x in reference], duration)
    predicted = union_intervals(prediction)
    if predicted and predicted[-1][1] > duration + 1e-6:
        raise ValueError("Predicted interval exceeds recording duration.")
    predicted = [(start, min(end, duration)) for start, end in predicted]

    def length(ranges: list[tuple[float, float]]) -> float:
        return sum(end - start for start, end in ranges)

    p_seconds = length(predicted)
    t_seconds = length(truth)
    overlap = 0.0
    i = j = 0
    while i < len(predicted) and j < len(truth):
        a, b = predicted[i]
        c, d = truth[j]
        overlap += max(0.0, min(b, d) - max(a, c))
        if b <= d:
            i += 1
        else:
            j += 1
    union = p_seconds + t_seconds - overlap
    return {
        "predicted_seconds": round(p_seconds, 6),
        "annotated_seconds": round(t_seconds, 6),
        "overlap_seconds": round(overlap, 6),
        "false_positive_seconds": round(max(0.0, p_seconds - overlap), 6),
        "missed_seconds": round(max(0.0, t_seconds - overlap), 6),
        "time_precision": round(overlap / p_seconds, 6) if p_seconds else None,
        "time_recall": round(overlap / t_seconds, 6) if t_seconds else None,
        "time_iou": round(overlap / union, 6) if union else None,
    }


def event_alignment(
    prediction_seconds: list[float],
    annotated_intervals: list[tuple[float, float]],
    duration: float,
    *,
    tolerance_seconds: float = 0.20,
) -> dict:
    """Maximum one-to-one event matching within a fixed time tolerance.

    Reports center-based timing matches for *proxy* peaks/candidates. A match
    does not establish that a peak represents an actual syllable or breath.
    """
    duration = _number(duration)
    references = validate_intervals([list(x) for x in annotated_intervals], duration)
    tolerance = _number(tolerance_seconds)
    if not 0 <= tolerance <= 1:
        raise ValueError("Tolerance must be between 0 and 1 second.")
    if not isinstance(prediction_seconds, list) or len(prediction_seconds) > 400:
        raise ValueError("Predictions must be a bounded list.")
    predicted = sorted(_number(x) for x in prediction_seconds)
    if any(x < 0 or x > duration + 1e-6 for x in predicted):
        raise ValueError("Event prediction lies outside recording.")
    centers = sorted((start + end) / 2 for start, end in references)
    # Left-to-right matching yields a maximum one-to-one match count for
    # ordered points under a uniform absolute-distance threshold.
    i = j = matched = 0
    while i < len(predicted) and j < len(centers):
        delta = predicted[i] - centers[j]
        if abs(delta) <= tolerance + 1e-9:
            matched += 1
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return {
        "predicted_events": len(predicted),
        "annotated_events": len(centers),
        "matched_events": matched,
        "false_positive_events": len(predicted) - matched,
        "missed_events": len(centers) - matched,
        "tolerance_seconds": tolerance,
        "precision": round(matched / len(predicted), 6) if predicted else None,
        "recall": round(matched / len(centers), 6) if centers else None,
        "f1": (
            round(2 * matched / (len(predicted) + len(centers)), 6)
            if predicted and centers else None
        ),
    }
