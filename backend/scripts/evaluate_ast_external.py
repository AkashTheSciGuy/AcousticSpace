r"""Read-only external-corpus AST evaluation with exact-audio leakage checks.

Windows CMD, from backend:
  .venv\Scripts\python.exe scripts\evaluate_ast_external.py ^
    F:\Datasets\IndependentSpoofCorpus ^
    F:\Datasets\IndependentSpoofCorpus\eval_manifest.csv ^
    --training-manifest F:\Datasets\ASVspoof2019\manifest.csv ^
    --model-dir models\ast_partial_top1_v1 ^
    --output models\ast_external_v1.json

External CSV columns: path,label,sha256,group,source_reference.
Use real/fake labels (not 0/1). This program does not construct or invent labels.
It uses a pre-locked 0.5 threshold and never trains, chooses thresholds,
or revisits the validation data. A checksum disjointness check is necessary
but not sufficient to guarantee independent speakers or source recordings.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ast_inputs import ast_input_values, forward_ast  # noqa: E402
from app.ast_inference import _load_local_ast  # noqa: E402
from app.ast_metrics import bootstrap_binary_uncertainty, classification_metrics  # noqa: E402
from app.preprocessing import load_audio, SUPPORTED_SUFFIXES  # noqa: E402

REQUIRED_COLUMNS = frozenset(("path", "label", "sha256", "group", "source_reference"))
ALLOWED_LABELS = {"real": 0, "fake": 1}
MAX_ROWS = 1000
MAX_BYTES = 20 * 1024 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
BLANKS = {"", "unknown", "n/a", "na", "null", "none", "todo", "tbd", "?"}


def _file_hash(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _read_csv(path: Path, *, name: str):
    path = Path(path)
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_MANIFEST_BYTES:
        raise ValueError(f"{name} manifest missing or exceeds 2 MiB.")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        if not REQUIRED_COLUMNS.issubset(rows.fieldnames or []) and name == "external":
            raise ValueError("External CSV requires path,label,sha256,group,source_reference.")
        if name == "training" and not {"sha256", "label"}.issubset(rows.fieldnames or []):
            raise ValueError("Training manifest requires sha256 and label.")
        for index, row in enumerate(rows, 2):
            if index > MAX_ROWS + 1:
                raise ValueError(f"{name} manifest has more than {MAX_ROWS} records.")
            yield index, row


def _hash_value(value: str | None, *, line: int) -> str:
    text = (value or "").strip().lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"Invalid SHA-256 on line {line}.")
    return text


def _relative_audio(root: Path, name: str | None) -> Path:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("Expected a safe relative audio path.")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError("Expected a safe relative WAV, MP3 or FLAC audio path.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Audio path is missing or outside dataset root.")
    if not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError("External audio must be nonempty and no larger than 20 MiB.")
    return path


def load_external_records(root: Path, external_manifest: Path, training_manifest: Path) -> tuple[list, dict]:
    """Reject exact content overlap with every hash in the original training CSV."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("External dataset root does not exist.")
    reference_hashes = set()
    for line, row in _read_csv(training_manifest, name="training"):
        digest = _hash_value(row.get("sha256"), line=line)
        if row.get("label") not in ALLOWED_LABELS:
            raise ValueError(f"Training manifest label invalid on line {line}.")
        reference_hashes.add(digest)
    if not reference_hashes:
        raise ValueError("Original training manifest has no SHA-256 records.")

    entries = []
    seen_paths: set[Path] = set()
    seen_hashes: set[str] = set()
    classes: Counter[str] = Counter()
    groups: set[str] = set()
    sources: set[str] = set()
    for line, row in _read_csv(external_manifest, name="external"):
        label = (row.get("label") or "").strip()
        group = (row.get("group") or "").strip()
        source = (row.get("source_reference") or "").strip()
        if label not in ALLOWED_LABELS:
            raise ValueError(f"Line {line}: label must be real or fake.")
        if group.lower() in BLANKS or source.lower() in BLANKS:
            raise ValueError(f"Line {line}: real source group and provenance are required.")
        path = _relative_audio(root, row.get("path"))
        expected = _hash_value(row.get("sha256"), line=line)
        if path in seen_paths or expected in seen_hashes:
            raise ValueError(f"Line {line}: repeated file or SHA-256 in external evaluation.")
        if expected in reference_hashes:
            raise ValueError(f"Line {line}: exact source-audio SHA-256 overlaps known training/validation data.")
        if _file_hash(path) != expected:
            raise ValueError(f"Line {line}: audio checksum does not match external manifest.")
        seen_paths.add(path)
        seen_hashes.add(expected)
        entries.append((path, ALLOWED_LABELS[label]))
        classes[label] += 1
        groups.add(group)
        sources.add(source)
    if classes["real"] < 1 or classes["fake"] < 1:
        raise ValueError("External evaluation needs both independently labeled real and fake recordings.")
    if len(groups) < 2:
        raise ValueError("External evaluation needs at least two documented source groups.")
    return entries, {
        "recordings": len(entries),
        "classes": dict(classes),
        "declared_groups": len(groups),
        "declared_provenance_references": len(sources),
        "external_manifest_sha256": _file_hash(Path(external_manifest)),
        "original_manifest_sha256": _file_hash(Path(training_manifest)),
        "exact_audio_hash_overlap": 0,
    }


def evaluate_external(root: Path, manifest: Path, training_manifest: Path, checkpoint: Path,
                      *, threshold: float = 0.5, bootstrap_samples: int = 500,
                      seed: int = 42) -> dict:
    """Score a frozen checkpoint on held-out data; no weight or threshold changes."""
    if type(threshold) not in (float, int) or float(threshold) != 0.5:
        raise ValueError("External evaluation uses predeclared 0.5 threshold only.")
    if type(bootstrap_samples) is not int or not 20 <= bootstrap_samples <= 2000:
        raise ValueError("bootstrap_samples must be an integer between 20 and 2000.")
    if type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise ValueError("Bootstrap seed must be a nonnegative bounded integer.")
    records, provenance = load_external_records(root, manifest, training_manifest)
    torch, extractor, model, device = _load_local_ast(str(Path(checkpoint).resolve()))
    model.eval()
    labels, scores = [], []
    with torch.inference_mode():
        for path, label in records:
            samples = load_audio(path).samples[:160_000]
            values = ast_input_values(extractor, samples).to(device)
            output = forward_ast(model, values).logits
            probs = torch.softmax(output, dim=-1).detach().cpu().numpy()
            if (probs.shape != (1, 2) or not np.isfinite(probs).all()
                    or (probs < 0).any() or (probs > 1).any()
                    or not np.isclose(probs.sum(), 1, atol=1e-5)):
                raise ValueError("External AST inference returned invalid class probabilities.")
            labels.append(label)
            scores.append(float(probs[0, 1]))
    return {
        "status": "evaluated",
        "evaluation_kind": "externally_supplied_audio_no_retraining",
        "checkpoint_path": str(Path(checkpoint).resolve()),
        "decision_threshold": 0.5,
        "max_analyzed_seconds_per_clip": 10,
        "provenance": provenance,
        "metrics": classification_metrics(labels, scores, threshold=0.5),
        "uncertainty": bootstrap_binary_uncertainty(
            labels, scores, threshold=0.5, samples=bootstrap_samples, seed=seed),
        "limitations": [
            "The source identity, speaker-group information, annotation labels and independence of the external corpus are operator-declared and require original-source review.",
            "Exact SHA-256 disjointness alone does not exclude the same utterance transcoded, re-recorded, re-encoded, or shared speaker identity across corpora.",
            "The checkpoint was already selected on its original validation set; do not tune or select further models using these external evaluation labels.",
            "Performance on this external corpus does not measure RIR recovery, breath/syllable truth, manipulation localization, or forensic confidence.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("external_manifest", type=Path)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose a new report filename.")
    report = evaluate_external(
        args.dataset_root, args.external_manifest, args.training_manifest,
        args.model_dir, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
