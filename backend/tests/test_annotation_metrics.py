"""Ground-truth scoring must fail closed and never invent event accuracy."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import soundfile as sf

from app.annotation_metrics import event_alignment, interval_coverage, validate_intervals
from scripts import evaluate_annotations


def test_overlapping_predicted_windows_are_counted_only_once():
    score = interval_coverage([(0, 2), (1, 3)], [(2, 4)], 5.0)
    assert score["predicted_seconds"] == 3.0
    assert score["annotated_seconds"] == 2.0
    assert score["overlap_seconds"] == 1.0
    assert score["false_positive_seconds"] == 2.0
    assert score["missed_seconds"] == 1.0
    assert score["time_precision"] == pytest.approx(1 / 3, abs=1e-6)
    assert score["time_recall"] == pytest.approx(.5)
    assert score["time_iou"] == pytest.approx(.25)


def test_negative_only_clips_never_report_perfect_precision_or_recall():
    score = interval_coverage([], [], 1.0)
    assert score["time_precision"] is None
    assert score["time_recall"] is None
    assert score["time_iou"] is None
    event = event_alignment([], [], 1.0)
    assert event["precision"] is None
    assert event["recall"] is None
    assert event["f1"] is None


def test_fixed_event_tolerance_enforces_one_to_one_matching():
    result = event_alignment([.15, .17, .45], [(.10, .20), (.40, .50)], 1.0, tolerance_seconds=.1)
    assert result["matched_events"] == 2
    assert result["false_positive_events"] == 1
    assert result["missed_events"] == 0
    assert result["precision"] == pytest.approx(2 / 3, abs=1e-6)
    assert result["recall"] == 1.0


@pytest.mark.parametrize("intervals", [
    [(0.0, .1), (.05, .2)],
    [(.3, .2)], [(0.0, 1.1)], [[0.0, None]],
    [[0.0, float("nan")]], [[False, .3]], "not intervals",
])
def test_invalid_annotation_intervals_are_rejected(intervals):
    with pytest.raises(ValueError):
        validate_intervals(intervals, 1.0)


def _manifest(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    recordings = []
    for i, frequency in enumerate((220, 350)):
        t = np.arange(16000, dtype=np.float64) / 16000
        signal = .2 * np.sin(2 * np.pi * frequency * t)
        file = corpus / f"sample{i}.wav"
        sf.write(file, signal, 16000, format="WAV", subtype="PCM_16")
        recordings.append({
            "recording_id": f"sample{i}",
            "path": file.name,
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            "speaker_id": f"speaker{i}",
            "session_id": f"session{i}",
            "annotation_source": f"manual_review_record_{i}",
            "annotations": {
                "syllables": [[.10, .20], [.35, .45]],
                "breaths": [[.55, .68]],
                "tamper": [[.20, .5]] if i == 0 else [],
            },
        })
    manifest = tmp_path / "labels.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "partition": "test",
        "recordings": recordings,
    }), encoding="utf-8")
    return corpus, manifest, recordings


def test_cadence_evaluation_scores_real_audio_without_claiming_breath_detection(tmp_path):
    root, manifest, _ = _manifest(tmp_path)
    report = evaluate_annotations.evaluate(root, manifest, task="cadence")
    assert report["status"] == "evaluated"
    assert report["partition"] == "test"
    assert report["recordings"] == 2
    assert len(report["manifest_sha256"]) == 64
    assert report["model_variant"] is None
    assert report["summary"]["syllables"]["annotated_events"] == 4
    assert report["summary"]["breaths"]["annotated_events"] == 2
    assert "NOT" not in report["warning"] or "not" in report["warning"].lower()
    assert "NOT been audited" in report["warning"]


def test_tamper_evaluation_scores_annotated_time_without_double_counting(
    tmp_path, monkeypatch,
):
    root, manifest, _ = _manifest(tmp_path)

    def fake_segments(_audio, variant):
        assert variant == "baseline"
        return {
            "segments": [
                {"start_seconds": 0, "end_seconds": .35, "flagged": True},
                {"start_seconds": .2, "end_seconds": .5, "flagged": True},
                {"start_seconds": .5, "end_seconds": 1., "flagged": False},
            ]
        }

    monkeypatch.setattr(evaluate_annotations, "score_segments", fake_segments)
    result = evaluate_annotations.evaluate(root, manifest, task="tamper")
    assert result["recordings"] == 2
    assert result["summary"]["tamper"]["predicted_seconds"] == 1.0
    assert result["summary"]["tamper"]["annotated_seconds"] == .3
    assert result["summary"]["tamper"]["overlap_seconds"] == .3
    assert result["summary"]["tamper"]["false_positive_seconds"] == .7
    assert result["summary"]["tamper"]["time_recall"] == 1.0
    assert result["summary"]["tamper"]["time_precision"] == .3


def test_cli_writes_versioned_report_and_refuses_to_overwrite(tmp_path, capsys):
    root, manifest, _ = _manifest(tmp_path)
    output = tmp_path / "heldout.json"
    args = [str(root), str(manifest), "--task", "cadence", "--output", str(output)]
    assert evaluate_annotations.main(args) == 0
    saved = output.read_bytes()
    assert evaluate_annotations.main(args) == 2
    assert output.read_bytes() == saved
    assert "Refusing to overwrite" in capsys.readouterr().err


def test_missing_annotation_is_rejected_before_scoring(tmp_path):
    root, manifest, recordings = _manifest(tmp_path)
    del recordings[0]["annotations"]["breaths"]
    manifest.write_text(json.dumps({
        "schema_version": 1, "partition": "test",
        "recordings": recordings,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing task-specific"):
        evaluate_annotations.evaluate(root, manifest, task="cadence")


def test_mismatched_sha_is_rejected(tmp_path):
    root, manifest, recordings = _manifest(tmp_path)
    recordings[0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps({
        "schema_version": 1, "partition": "test",
        "recordings": recordings,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatched audio SHA-256"):
        evaluate_annotations.evaluate(root, manifest, task="cadence")


def test_outside_root_paths_and_missing_provenance_are_rejected(tmp_path):
    root, manifest, recordings = _manifest(tmp_path)
    recordings[0]["path"] = "../other.wav"
    manifest.write_text(json.dumps({
        "schema_version": 1, "partition": "test", "recordings": recordings,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="relative WAV"):
        evaluate_annotations.evaluate(root, manifest, task="cadence")
    recordings[0]["path"] = "sample0.wav"
    recordings[0]["annotation_source"] = "unknown"
    manifest.write_text(json.dumps({
        "schema_version": 1, "partition": "test", "recordings": recordings,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="annotation_source"):
        evaluate_annotations.evaluate(root, manifest, task="cadence")
