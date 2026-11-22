"""Official ASVspoof eval importer tests use generated FLACs, not benchmark audio."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.prepare_asvspoof2019_eval import create_manifest, main


def _read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture
def official_fixture(tmp_path):
    root = tmp_path / "LA"
    audio_dir = root / "ASVspoof2019_LA_eval" / "flac"
    protocol_dir = root / "ASVspoof2019_LA_cm_protocols"
    audio_dir.mkdir(parents=True)
    protocol_dir.mkdir()
    lines = []
    index = 0
    for label, attack in (("bonafide", "-"), ("spoof", "A07"), ("spoof", "A08")):
        for speaker in ("LA_0030", "LA_0031", "LA_0032"):
            for _ in range(3):
                index += 1
                utterance = f"LA_E_{index:07d}"
                audio = np.zeros(4096, dtype=np.float32)
                audio[20 + index] = 0.5
                sf.write(audio_dir / f"{utterance}.flac", audio, 16000, format="FLAC")
                lines.append(f"{speaker} {utterance} - {attack} {label}")
    protocol = protocol_dir / "ASVspoof2019.LA.cm.eval.trl.txt"
    protocol.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root, protocol


def test_fixed_eval_subset_is_reproducible_and_uses_official_labels(official_fixture, tmp_path):
    first = tmp_path / "fixed_a.csv"
    second = tmp_path / "fixed_b.csv"
    result = create_manifest(official_fixture[0], first, per_class=5, seed=42)
    again = create_manifest(official_fixture[0], second, per_class=5, seed=42)
    assert first.read_bytes() == second.read_bytes()
    assert result["manifest_sha256"] == again["manifest_sha256"]
    assert result["counts"] == {"real": 5, "fake": 5}
    assert result["speaker_groups"] >= 2
    assert result["partition"] == "ASVspoof2019_LA_official_eval_subset"
    rows = _read_rows(first)
    assert len(rows) == 10
    assert sorted(row["label"] for row in rows) == ["fake"] * 5 + ["real"] * 5
    assert all(row["path"].startswith("ASVspoof2019_LA_eval/flac/LA_E_") for row in rows)
    assert all(row["source_reference"].startswith("ASVspoof2019.LA.cm.eval.trl.txt:line=")
               for row in rows)
    assert all(len(row["sha256"]) == 64 for row in rows)
    assert len({row["path"] for row in rows}) == len(rows)
    assert len({row["sha256"] for row in rows}) == len(rows)
    assert all(hashlib.sha256((official_fixture[0] / row["path"]).read_bytes()).hexdigest() ==
               row["sha256"] for row in rows)


def test_eval_subset_changes_deterministically_with_explicit_seed(official_fixture, tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    create_manifest(official_fixture[0], first, per_class=5, seed=10)
    create_manifest(official_fixture[0], second, per_class=5, seed=11)
    assert _read_rows(first) != _read_rows(second)


@pytest.mark.parametrize("per_class", [0, 501, 1.5, "2", True])
def test_invalid_class_count_rejected_before_file_write(official_fixture, tmp_path, per_class):
    out = tmp_path / "invalid.csv"
    with pytest.raises(ValueError, match="per_class"):
        create_manifest(official_fixture[0], out, per_class=per_class)
    assert not out.exists()


@pytest.mark.parametrize("seed", [-1, 2 ** 63, 1.5, "42", True])
def test_invalid_seed_rejected(official_fixture, tmp_path, seed):
    with pytest.raises(ValueError, match="seed"):
        create_manifest(official_fixture[0], tmp_path / "invalid.csv",
                        per_class=2, seed=seed)


def test_refuses_to_overwrite_previously_fixed_subset(official_fixture, tmp_path):
    output = tmp_path / "fixed.csv"
    assert main([str(official_fixture[0]), "--per-class", "2", "--output", str(output)]) == 0
    before = output.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_manifest(official_fixture[0], output, per_class=2)
    assert output.read_bytes() == before


def test_protocol_bad_label_fails_closed(official_fixture, tmp_path):
    root, protocol = official_fixture
    data = protocol.read_text(encoding="utf-8")
    protocol.write_text(data.replace("bonafide", "UNLABELED", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported official label"):
        create_manifest(root, tmp_path / "bad.csv", per_class=2)
    assert not (tmp_path / "bad.csv").exists()


def test_duplicate_protocol_utterance_fails_closed(official_fixture, tmp_path):
    root, protocol = official_fixture
    data = protocol.read_text(encoding="utf-8")
    first = data.splitlines()[0]
    protocol.write_text(data + first + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate evaluation utterance"):
        create_manifest(root, tmp_path / "dupe.csv", per_class=2)


def test_incomplete_eval_source_refuses_manifest_write(official_fixture, tmp_path):
    root, protocol = official_fixture
    data = protocol.read_text(encoding="utf-8")
    real_id = data.splitlines()[0].split()[1]
    # Selecting all nine genuine recordings guarantees this missing source is checked.
    (root / "ASVspoof2019_LA_eval" / "flac" / (real_id + ".flac")).unlink()
    target = tmp_path / "incomplete.csv"
    with pytest.raises(ValueError, match="Selected official FLAC missing"):
        create_manifest(root, target, per_class=9)
    assert not target.exists()


def test_importer_requires_each_class_and_valid_protocol(official_fixture, tmp_path):
    root, protocol = official_fixture
    data = protocol.read_text(encoding="utf-8")
    without_fakes = "\n".join(line for line in data.splitlines()
                              if line.endswith("bonafide")) + "\n"
    protocol.write_text(without_fakes, encoding="utf-8")
    with pytest.raises(ValueError, match="Fewer than"):
        create_manifest(root, tmp_path / "oneclass.csv", per_class=2)


def test_unexpected_eval_utterance_prefix_is_rejected(official_fixture, tmp_path):
    root, protocol = official_fixture
    data = protocol.read_text(encoding="utf-8").replace("LA_E_", "LA_D_", 1)
    protocol.write_text(data, encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected evaluation speaker/utterance"):
        create_manifest(root, tmp_path / "bad_prefix.csv", per_class=2)


def test_new_manifest_is_accepted_by_external_evaluator(official_fixture, tmp_path):
    from scripts.evaluate_ast_external import load_external_records
    root, _ = official_fixture
    manifest = tmp_path / "eval.csv"
    create_manifest(root, manifest, per_class=4)
    training = tmp_path / "train.csv"
    training.write_text("path,label,sha256,group\n"
                        + "training.flac,real," + "0" * 64 + ",LA_9999\n",
                        encoding="utf-8")
    paths, provenance = load_external_records(root, manifest, training)
    assert len(paths) == 8
    assert provenance["classes"] == {"real": 4, "fake": 4}
    assert provenance["exact_audio_hash_overlap"] == 0
