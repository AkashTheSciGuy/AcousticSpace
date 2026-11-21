"""Optional AST path is inert until a verified local model is present."""
import csv
import hashlib
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app
from app.ast_inference import ast_status
from scripts.train_ast import split_rows

client = TestClient(app)


def test_ast_status_is_unavailable_without_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("ACOUSTICSPACE_AST_MODEL", str(tmp_path / "missing"))
    response = client.get("/api/ast/status")
    assert response.status_code == 200
    assert response.json()["available"] is False
    assert ast_status()["available"] is False


def test_ast_predict_returns_503_without_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("ACOUSTICSPACE_AST_MODEL", str(tmp_path / "missing"))
    response = client.post("/api/ast/predict",
                           files={"file": ("clip.wav", b"RIFF0000WAVE")})
    assert response.status_code == 503


def test_ast_manifest_loader_respects_grouped_splits(tmp_path):
    root = tmp_path / "audio"
    root.mkdir()
    rows = []
    for split in ("train", "val", "test"):
        for label in ("real", "fake"):
            name = split + "_" + label + ".wav"
            payload = name.encode()
            (root / name).write_bytes(payload)
            rows.append({"path": name, "label": label, "split": split,
                         "group": split, "sha256": hashlib.sha256(payload).hexdigest()})
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    splits = split_rows(root, manifest)
    assert all(len(splits[key]) == 2 for key in splits)
    assert all(isinstance(path, Path) for path, _ in splits["train"])
