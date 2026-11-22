r"""Evaluate one locally trained AST checkpoint without modifying its weights.

Examples (Windows CMD, run from backend):
    .venv\Scripts\python.exe scripts\evaluate_ast.py DATASET_ROOT MANIFEST --split val --select-threshold
    .venv\Scripts\python.exe scripts\evaluate_ast.py DATASET_ROOT MANIFEST --split test --threshold 0.5

Select a threshold ONLY on validation, then lock it before an optional one-time
test evaluation. Repeated test checks invalidate the claim of an untouched test.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ast_inputs import ast_input_values, forward_ast  # noqa: E402
from app.ast_metrics import (  # noqa: E402
    bootstrap_binary_uncertainty, classification_metrics, select_validation_threshold,
)
from app.ast_inference import _load_local_ast  # noqa: E402
from app.preprocessing import load_audio  # noqa: E402
from scripts.train_ast import split_rows  # noqa: E402



def reported_training_mode(checkpoint: Path) -> str:
    """Read small local training metadata without guessing training strategy."""
    file = Path(checkpoint) / "training_meta.json"
    try:
        if not file.is_file() or not 0 < file.stat().st_size <= 65536:
            return "unknown"
        with file.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
    except (OSError, UnicodeError, ValueError):
        return "unknown"
    if not isinstance(metadata, dict) or type(metadata.get("full_finetune")) is not bool:
        return "unknown"
    layers = metadata.get("train_last_layers", 0)
    if type(layers) is not int or layers < 0:
        return "unknown"
    if metadata["full_finetune"]:
        return "full_encoder_finetune" if layers == 0 else "unknown"
    return "partial_encoder_finetune" if layers else "head_only"


def evaluate_checkpoint(dataset_root, manifest, checkpoint, *, split="val",
                        threshold=0.5, select_threshold=False,
                        bootstrap_samples=0, bootstrap_seed=42):
    """Load only trusted local model weights, score the requested manifest split."""
    if split not in ("val", "test"):
        raise ValueError("Select only the validation or test split.")
    if select_threshold and split != "val":
        raise ValueError("Threshold selection is allowed only on validation.")
    if type(bootstrap_samples) is not int or (
            bootstrap_samples != 0 and not 20 <= bootstrap_samples <= 2000):
        raise ValueError("bootstrap_samples must be 0 (disabled) or 20 to 2000")
    if type(bootstrap_seed) is not int or not 0 <= bootstrap_seed < 2 ** 63:
        raise ValueError("bootstrap_seed must be a nonnegative bounded integer")
    # Verifies hashes and speaker/source-disjoint splits before loading audio.
    records = split_rows(Path(dataset_root), Path(manifest))
    torch, extractor, model, device = _load_local_ast(str(Path(checkpoint).resolve()))
    model.eval()
    truth, scores = [], []
    with torch.inference_mode():
        for path, label in records[split]:
            samples = load_audio(path).samples[:160_000]
            inputs = ast_input_values(extractor, samples).to(device)
            output = forward_ast(model, inputs).logits
            probabilities = torch.softmax(output, dim=-1).detach().cpu().numpy()
            if (probabilities.shape != (1, 2) or not np.isfinite(probabilities).all()
                    or np.any((probabilities < 0) | (probabilities > 1))
                    or not np.isclose(probabilities.sum(), 1, atol=1e-5)):
                raise ValueError("AST returned invalid class probabilities.")
            truth.append(label)
            scores.append(float(probabilities[0, 1]))
    report = {
        "model_type": "optional_ast_experiment",
        "dataset_split": split,
        "source": "local_asvspoof_manifest",
        "model_dir": str(Path(checkpoint).resolve()),
        "reported_training_mode": reported_training_mode(Path(checkpoint)),
        "dataset_manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        "metrics": classification_metrics(truth, scores, threshold=threshold),
        "warning": (
            "Exploratory within-dataset evaluation only. Scores are not calibrated; "
            "this does not measure RIR recovery or generalization. Do not select "
            "models or thresholds using the test split."
        ),
    }
    if select_threshold:
        report["selected_validation_threshold"] = select_validation_threshold(truth, scores)
    if bootstrap_samples:
        report["uncertainty"] = bootstrap_binary_uncertainty(
            truth, scores, threshold=threshold,
            samples=bootstrap_samples, seed=bootstrap_seed,
        )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path("models/ast_model"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--select-threshold", action="store_true",
                        help="Explore a balanced-accuracy threshold on validation ONLY.")
    parser.add_argument("--bootstrap-samples", type=int, default=0,
                        help="Optional stratified bootstrap iterations (0 off; 20-2000)")
    parser.add_argument("--bootstrap-seed", type=int, default=42,
                        help="Reproducible nonnegative bootstrap random seed")
    parser.add_argument("--output", type=Path, help="Write a new report without overwriting files.")
    args = parser.parse_args(argv)
    if args.split == "test" and args.threshold is None:
        parser.error("Explicit --threshold is required for test evaluation. Lock it on validation first.")
    if args.select_threshold and args.split != "val":
        parser.error("--select-threshold is permitted only with --split val.")
    if args.bootstrap_samples != 0 and not 20 <= args.bootstrap_samples <= 2000:
        parser.error("--bootstrap-samples must be 0 or an integer between 20 and 2000.")
    if args.bootstrap_seed < 0 or args.bootstrap_seed >= 2 ** 63:
        parser.error("--bootstrap-seed must be a nonnegative bounded integer.")
    if args.output is not None and args.output.exists():
        parser.error(
            f"Output report already exists: {args.output}. "
            "Choose a new filename, omit --output to print the report, "
            "or inspect the existing file."
        )
    report = evaluate_checkpoint(
        args.dataset_root, args.manifest, args.model_dir,
        split=args.split,
        threshold=args.threshold if args.threshold is not None else 0.5,
        select_threshold=args.select_threshold,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
