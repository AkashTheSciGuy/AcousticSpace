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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ast_inputs import ast_input_values, forward_ast  # noqa: E402
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
        for input_values, labels in loader:
            output = forward_ast(model, input_values.to(device)).logits
            truth.extend(labels.tolist())
            guesses.extend(output.argmax(dim=-1).cpu().tolist())
    return {
        "samples": len(truth),
        "accuracy": float(accuracy_score(truth, guesses)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, guesses)),
        "precision_fake": float(precision_score(truth, guesses, zero_division=0)),
        "recall_fake": float(recall_score(truth, guesses, zero_division=0)),
        "f1_fake": float(f1_score(truth, guesses, zero_division=0)),
        "confusion_matrix_real_fake": confusion_matrix(truth, guesses, labels=[0, 1]).tolist(),
    }


def accumulation_divisor(step: int, total: int, accumulate: int) -> int:
    """Scale the final accumulation group correctly when it is short."""
    if total < 1 or accumulate < 1 or step < 1 or step > total:
        raise ValueError("Invalid optimizer accumulation arguments.")
    group_first = ((step - 1) // accumulate) * accumulate
    return min(accumulate, total - group_first)



def configure_trainable_ast(model, *, train_last_layers: int = 0,
                            full_finetune: bool = False) -> dict:
    """Select head-only, partial-encoder, or full-encoder fine-tuning.

    Supports the 4.x AST encoder.layer and 5.x AST layers layouts.
    Fails rather than silently falling back to training only the head.
    """
    if type(train_last_layers) is not int or train_last_layers < 0:
        raise ValueError("train_last_layers must be a nonnegative integer.")
    if type(full_finetune) is not bool or (full_finetune and train_last_layers):
        raise ValueError("Choose either full_finetune or train_last_layers, not both.")
    for parameter in model.parameters():
        parameter.requires_grad_(full_finetune)
    if full_finetune:
        mode = "full_encoder_finetune"
    else:
        classifier = getattr(model, "classifier", None)
        backbone = getattr(model, "audio_spectrogram_transformer", None)
        if classifier is None or backbone is None:
            raise RuntimeError("Expected a compatible AST classifier and encoder.")
        for parameter in classifier.parameters():
            parameter.requires_grad_(True)
        mode = "head_only"
        if train_last_layers:
            old_encoder = getattr(backbone, "encoder", None)
            layers = getattr(old_encoder, "layer", None)
            if layers is None:
                layers = getattr(backbone, "layers", None)
            if layers is None or not 1 <= train_last_layers <= len(layers):
                raise ValueError("AST encoder has fewer layers than requested or unsupported layout.")
            for layer in list(layers)[-train_last_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
            final_norm = getattr(backbone, "layernorm", None)
            if final_norm is not None:
                for parameter in final_norm.parameters():
                    parameter.requires_grad_(True)
            mode = "partial_encoder_finetune"
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    if not trainable:
        raise RuntimeError("No trainable AST parameters found.")
    return {
        "mode": mode, "train_last_layers": train_last_layers,
        "trainable_parameters": trainable, "total_parameters": total,
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
    if args.patience < 0 or not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Patience must be nonnegative and learning rate must be positive.")
    if type(args.train_last_layers) is not int or args.train_last_layers < 0:
        raise ValueError("train_last_layers must be a nonnegative integer.")
    if args.full_finetune and args.train_last_layers:
        raise ValueError("Choose full_finetune OR train_last_layers.")
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
    trainable_config = configure_trainable_ast(
        model, train_last_layers=args.train_last_layers,
        full_finetune=args.full_finetune,
    )
    if args.gradient_checkpointing:
        if not getattr(model, "supports_gradient_checkpointing", False):
            raise RuntimeError("Installed AST model does not support gradient checkpointing.")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    print(json.dumps({"training_configuration": trainable_config,
                      "gradient_checkpointing": args.gradient_checkpointing}), flush=True)
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
            values = ast_input_values(extractor, waveform)
            return values.squeeze(0), torch.tensor(label, dtype=torch.long)

    train_loader = DataLoader(AudioDataset(grouped["train"]), batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(AudioDataset(grouped["val"]), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_score = -1.0
    best_state = None
    best_metrics = None
    best_epoch = 0
    history = []
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        model.train()
        # Prevent train-mode stochastic changes inside the *frozen* encoder.
        # The new classification head remains in training mode.
        if trainable_config["mode"] == "head_only":
            model.audio_spectrogram_transformer.eval()
        elif trainable_config["mode"] == "partial_encoder_finetune":
            backbone = model.audio_spectrogram_transformer
            backbone.eval()
            layers = getattr(getattr(backbone, "encoder", None), "layer", None)
            if layers is None:
                layers = backbone.layers
            for layer in list(layers)[-args.train_last_layers:]:
                layer.train()
            if getattr(backbone, "layernorm", None) is not None:
                backbone.layernorm.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss_sum = 0.0
        for step, (input_values, labels) in enumerate(train_loader, 1):
            # The final partial gradient-accumulation group needs its own
            # divisor, otherwise its update is unintentionally under-scaled.
            divisor = accumulation_divisor(step, len(train_loader), args.accumulate)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                raw_loss = forward_ast(model, input_values.to(device),
                                       labels=labels.to(device)).loss
                loss = raw_loss / divisor
            train_loss_sum += float(raw_loss.detach().item())
            scaler.scale(loss).backward()
            if step % args.accumulate == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        metrics = evaluate(model, val_loader, device, torch)
        train_loss = train_loss_sum / len(train_loader)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation": metrics})
        print(json.dumps({"epoch": epoch + 1, "train_loss": train_loss,
                          "selection_metric": args.selection_metric, "validation": metrics}),
              flush=True)
        score = metrics[args.selection_metric]
        if score > best_score:
            best_score = score
            best_metrics = metrics
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state = {name: parameter.detach().cpu().clone()
                          for name, parameter in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
        if args.patience and epochs_without_improvement >= args.patience:
            print(json.dumps({"early_stopped_after_epoch": epoch + 1,
                              "best_epoch": best_epoch}), flush=True)
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a valid model.")
    model.load_state_dict(best_state)
    model.to(device)
    results = {"validation": best_metrics, "best_epoch": best_epoch,
               "selection_metric": args.selection_metric, "epoch_history": history}
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
        "train_last_layers": args.train_last_layers,
        "training_mode": trainable_config["mode"],
        "trainable_parameters": trainable_config["trainable_parameters"],
        "total_parameters": trainable_config["total_parameters"],
        "gradient_checkpointing": args.gradient_checkpointing,
        "seed": args.seed,
        "learning_rate": args.lr,
        "gradient_accumulation": args.accumulate,
        "early_stopping_patience": args.patience,
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
    parser.add_argument("--selection-metric", choices=("f1_fake", "balanced_accuracy"),
                        default="f1_fake",
                        help="Choose checkpoint on validation only. Balanced accuracy penalizes real false positives.")
    parser.add_argument("--patience", type=int, default=0,
                        help="Stop after this many epochs without improvement (0 disables).")
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--train-last-layers", type=int, default=0,
                        help="Update last N encoder layers + classifier, instead of only classifier.")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Reduce activation memory during encoder training (slower).")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
