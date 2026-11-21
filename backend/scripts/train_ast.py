"""Optional Hugging Face AST audio-classification experiment.

This script is NOT a speech-RIR mismatch detector: labels are real/spoof.
The default freezes the AST encoder for a lower-memory head-only baseline;
pass --full-finetune to update the transformer weights if GPU memory permits.

Run from backend/ with trusted, validated manifest and optional dependencies.
No model or report is bundled in source control.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.manifest_validation import validate_training_manifest  # noqa: E402
from app.preprocessing import load_audio  # noqa: E402

PRETRAINED_AST = "MIT/ast-finetuned-audioset-10-10-0.4593"
LABEL_MAP = {"real": 0, "fake": 1}


def split_rows(dataset_root: Path, manifest: Path) -> dict:
    """Validate speaker separation and hashes before constructing AST datasets."""
    validate_training_manifest(dataset_root, manifest)
    root = dataset_root.resolve()
    grouped = {key: [] for key in ("train", "val", "test")}
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            grouped[row["split"]].append((root / row["path"], LABEL_MAP[row["label"]]))
    return grouped


def evaluate(model, loader, device, torch) -> dict:
    model.eval()
    truth, guesses = [], []
    with torch.inference_mode():
        for input_features, labels in loader:
            output = model(input_features=input_features.to(device)).logits
            truth.extend(labels.tolist())
            guesses.extend(output.argmax(dim=-1).cpu().tolist())
    return {
        "samples": len(truth),
        "accuracy": float(accuracy_score(truth, guesses)),
        "precision_fake": float(precision_score(truth, guesses, zero_division=0)),
        "recall_fake": float(recall_score(truth, guesses, zero_division=0)),
        "f1_fake": float(f1_score(truth, guesses, zero_division=0)),
        "confusion_matrix_real_fake": confusion_matrix(truth, guesses, labels=[0, 1]).tolist(),
    }


def train(args) -> dict:
    # Only import large optional ML dependencies when training is explicitly requested.
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
    except ImportError as exc:
        raise RuntimeError("Install requirements-ast.txt into the backend virtual environment.") from exc

    if args.epochs < 1 or args.batch_size < 1 or args.accumulate < 1:
        raise ValueError("Epochs, batch size and gradient accumulation must be positive.")
    if args.output.exists():
        raise FileExistsError("AST output directory already exists; choose a new output.")
    grouped = split_rows(args.dataset, args.manifest)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    extractor = ASTFeatureExtractor.from_pretrained(args.pretrained, trust_remote_code=False)
    model = ASTForAudioClassification.from_pretrained(
        args.pretrained, num_labels=2, ignore_mismatched_sizes=True,
        id2label={0: "real", 1: "fake"}, label2id=LABEL_MAP,
        trust_remote_code=False,
    )
    if not args.full_finetune:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("classifier.")
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("No trainable AST parameters found.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    class AudioDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, index):
            path, label = self.rows[index]
            waveform = load_audio(path).samples[:160000]  # 10 seconds at 16 kHz
            features = extractor(waveform, sampling_rate=16000, return_tensors="pt")
            return features["input_features"].squeeze(0), torch.tensor(label, dtype=torch.long)

    train_loader = DataLoader(AudioDataset(grouped["train"]), batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(AudioDataset(grouped["val"]), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_f1 = -1.0
    best_state = None
    best_metrics = None
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, (input_features, labels) in enumerate(train_loader, 1):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(input_features=input_features.to(device),
                             labels=labels.to(device)).loss / args.accumulate
            scaler.scale(loss).backward()
            if step % args.accumulate == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        metrics = evaluate(model, val_loader, device, torch)
        print(json.dumps({"epoch": epoch + 1, "validation": metrics}))
        if metrics["f1_fake"] > best_f1:
            best_f1 = metrics["f1_fake"]
            best_metrics = metrics
            best_state = {name: parameter.detach().cpu().clone()
                          for name, parameter in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Training did not produce a valid model.")
    model.load_state_dict(best_state)
    model.to(device)
    results = {"validation": best_metrics}
    if args.evaluate_test:
        loader = DataLoader(AudioDataset(grouped["test"]), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
        results["test"] = evaluate(model, loader, device, torch)

    args.output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(args.output, safe_serialization=True)
    extractor.save_pretrained(args.output)
    (args.output / "training_meta.json").write_text(json.dumps({
        "source": args.pretrained,
        "train_count": len(grouped["train"]),
        "val_count": len(grouped["val"]),
        "test_count": len(grouped["test"]),
        "full_finetune": args.full_finetune,
        "max_audio_seconds": 10,
        "warning": "Experimental real/fake AST classifier, not RIR recovery or cadence alignment.",
        "metrics": results,
    }, indent=2), encoding="utf-8")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/ast_model"))
    parser.add_argument("--pretrained", default=PRETRAINED_AST)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
