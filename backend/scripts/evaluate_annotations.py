r"""Evaluate AcousticSpace timing proxies on independently annotated local clips.

Examples (Windows CMD, run from backend):
  .venv\Scripts\python.exe scripts\evaluate_annotations.py ^
    datasets\annotated_corpus datasets\annotated_corpus\events.json ^
    --task cadence --output models\cadence_heldout_v1.json

  .venv\Scripts\python.exe scripts\evaluate_annotations.py ^
    datasets\annotated_corpus datasets\annotated_corpus\events.json ^
    --task tamper --model baseline --output models\tamper_heldout_v1.json

No annotations, room/source identity, tamper labels or accuracy are generated.
This is observational performance measurement, not forensic validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.annotation_metrics import event_alignment, interval_coverage, validate_intervals  # noqa: E402
from app.preprocessing import load_audio  # noqa: E402
from app.segment_analysis import score_segments  # noqa: E402
from app.speech_diagnostics import analyze_speech_diagnostics  # noqa: E402

MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_AUDIO_BYTES = 20 * 1024 * 1024
MAX_RECORDINGS = 200
PLACEHOLDERS = {"", "unknown", "none", "tbd", "todo", "na", "n/a", "?"}


def _required_text(item: dict, key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or value.strip().lower() in PLACEHOLDERS:
        raise ValueError(f"Recording requires non-placeholder {key!r} metadata.")
    return value.strip()


def _audio_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("Audio filename must be a safe POSIX-style relative path.")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() not in (".wav", ".mp3"):
        raise ValueError("Audio filename must be a relative WAV or MP3 path.")
    full = (root / path).resolve()
    if not full.is_relative_to(root) or not full.is_file():
        raise ValueError("Audio file is missing or outside dataset root.")
    if not 0 < full.stat().st_size <= MAX_AUDIO_BYTES:
        raise ValueError("Audio file exceeds 20 MiB or is empty.")
    return full


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_annotation_manifest(root: Path, manifest: Path, task: str) -> tuple[list[dict], str, str]:
    """Preflight trusted local paths and user-supplied labels before decoding."""
    root = Path(root).resolve()
    manifest = Path(manifest)
    if not root.is_dir() or not manifest.is_file() or not 0 < manifest.stat().st_size <= MAX_MANIFEST_BYTES:
        raise ValueError("Missing corpus root or bounded annotation manifest.")
    if task not in ("cadence", "tamper"):
        raise ValueError("Task must be cadence or tamper.")
    raw = manifest.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, ValueError) as exc:
        raise ValueError("Annotation manifest must be valid UTF-8 JSON.") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Expected version-1 annotation JSON object.")
    split = payload.get("partition")
    if split not in ("val", "test"):
        raise ValueError("Manifest partition must be 'val' or 'test'.")
    items = payload.get("recordings")
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_RECORDINGS:
        raise ValueError("Manifest must contain between 1 and 200 recordings.")

    records = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    seen_hashes: set[str] = set()
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Recording {i} must be a JSON object.")
        recording_id = _required_text(item, "recording_id")
        speaker_id = _required_text(item, "speaker_id")
        session_id = _required_text(item, "session_id")
        annotation_source = _required_text(item, "annotation_source")
        file = _audio_file(root, item.get("path"))
        expected_hash = item.get("sha256")
        if (not isinstance(expected_hash, str) or len(expected_hash) != 64
                or any(ch not in "0123456789abcdefABCDEF" for ch in expected_hash)):
            raise ValueError(f"Recording {i} requires a valid SHA-256 checksum.")
        digest = _sha256(file)
        if expected_hash.lower() != digest:
            raise ValueError(f"Recording {i} has mismatched audio SHA-256.")
        if recording_id in seen_ids or file in seen_paths or digest in seen_hashes:
            raise ValueError("Duplicate recording ID, audio path, or audio bytes.")
        labels = item.get("annotations")
        if not isinstance(labels, dict):
            raise ValueError("Recording annotations must be a JSON object.")
        keys = ("breaths", "syllables") if task == "cadence" else ("tamper",)
        if any(key not in labels for key in keys):
            raise ValueError(f"Missing task-specific ground-truth intervals: {keys}.")
        seen_ids.add(recording_id)
        seen_paths.add(file)
        seen_hashes.add(digest)
        records.append({
            "recording_id": recording_id,
            "speaker_id": speaker_id,
            "session_id": session_id,
            "annotation_source": annotation_source,
            "file": file,
            "sha256": digest,
            "annotations": labels,
        })
    return records, split, hashlib.sha256(raw).hexdigest()


def _totals(rows: list[dict], field: str) -> dict:
    total_pred = sum(row[field]["predicted_events"] for row in rows)
    total_truth = sum(row[field]["annotated_events"] for row in rows)
    matched = sum(row[field]["matched_events"] for row in rows)
    return {
        "predicted_events": total_pred,
        "annotated_events": total_truth,
        "matched_events": matched,
        "false_positive_events": total_pred - matched,
        "missed_events": total_truth - matched,
        "precision": round(matched / total_pred, 6) if total_pred else None,
        "recall": round(matched / total_truth, 6) if total_truth else None,
        "f1": round(2 * matched / (total_pred + total_truth), 6)
              if total_pred and total_truth else None,
    }


def _aggregate_time(rows: list[dict]) -> dict:
    sums = {key: sum(float(row["tamper"][key]) for row in rows) for key in (
        "predicted_seconds", "annotated_seconds", "overlap_seconds",
        "false_positive_seconds", "missed_seconds"
    )}
    p, t, match = sums["predicted_seconds"], sums["annotated_seconds"], sums["overlap_seconds"]
    union = p + t - match
    return {
        **{key: round(val, 6) for key, val in sums.items()},
        "time_precision": round(match / p, 6) if p else None,
        "time_recall": round(match / t, 6) if t else None,
        "time_iou": round(match / union, 6) if union else None,
    }


def evaluate(root: Path, manifest: Path, *, task: str = "cadence", model: str = "baseline",
             tolerance_seconds: float = .20) -> dict:
    """Run actual signal processing and compare results to separately supplied annotations."""
    if task not in ("cadence", "tamper"):
        raise ValueError("Task must be cadence or tamper.")
    if model not in ("baseline", "acoustic"):
        raise ValueError("Model must be baseline or acoustic.")
    if type(tolerance_seconds) not in (float, int) or not 0 <= tolerance_seconds <= 1:
        raise ValueError("Tolerance must be a number between 0 and 1.")
    records, split, manifest_hash = load_annotation_manifest(root, manifest, task)
    results: list[dict] = []
    for record in records:
        audio = load_audio(record["file"])
        duration = float(audio.duration_seconds)
        keys = ("breaths", "syllables") if task == "cadence" else ("tamper",)
        truth = {
            key: validate_intervals(record["annotations"][key], duration)
            for key in keys
        }
        detail: dict[str, Any] = {
            "recording_id": record["recording_id"],
            "speaker_id": record["speaker_id"],
            "session_id": record["session_id"],
            "annotation_source": record["annotation_source"],
            "audio_sha256": record["sha256"],
            "duration_seconds": duration,
        }
        if task == "cadence":
            diagnostics = analyze_speech_diagnostics(audio)
            candidates = diagnostics["noisy_quiet_intervals"]
            mids = [(row["start_seconds"] + row["end_seconds"]) / 2
                    for row in candidates]
            detail["breaths"] = event_alignment(
                mids, truth["breaths"], duration, tolerance_seconds=tolerance_seconds
            )
            detail["syllables"] = event_alignment(
                diagnostics["energy_peak_seconds"], truth["syllables"], duration,
                tolerance_seconds=tolerance_seconds,
            )
        else:
            windows = score_segments(audio, variant=model)["segments"]
            flagged = [(float(w["start_seconds"]), float(w["end_seconds"]))
                       for w in windows if w["flagged"]]
            detail["tamper"] = interval_coverage(flagged, truth["tamper"], duration)
        results.append(detail)

    summary = (
        {"breaths": _totals(results, "breaths"),
         "syllables": _totals(results, "syllables")}
        if task == "cadence" else {"tamper": _aggregate_time(results)}
    )
    return {
        "status": "evaluated",
        "task": task,
        "partition": split,
        "recordings": len(results),
        "speakers": len({row["speaker_id"] for row in results}),
        "sessions": len({row["session_id"] for row in results}),
        "manifest_sha256": manifest_hash,
        "model_variant": model if task == "tamper" else None,
        "event_match_tolerance_seconds": tolerance_seconds if task == "cadence" else None,
        "summary": summary,
        "per_recording": results,
        "warning": (
            "These are comparisons against *user-supplied* timestamps whose correctness "
            "and independence have NOT been audited. Energy peaks are not established "
            "syllables; quiet noisy intervals are not established breaths; overlapping "
            "window scores are not validated tamper localization. Repeated test-set "
            "inspection or threshold tuning invalidates an untouched holdout claim. "
            "No blind RIR extraction or forensic deepfake accuracy is established."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("annotation_manifest", type=Path)
    parser.add_argument("--task", choices=("cadence", "tamper"), required=True)
    parser.add_argument("--model", choices=("baseline", "acoustic"), default="baseline")
    parser.add_argument("--tolerance-seconds", type=float, default=.20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite evaluation: {args.output}")
        report = evaluate(
            args.dataset_root, args.annotation_manifest,
            task=args.task, model=args.model,
            tolerance_seconds=args.tolerance_seconds,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    except (OSError, ValueError, RuntimeError, FileExistsError) as exc:
        print(f"Annotated evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
