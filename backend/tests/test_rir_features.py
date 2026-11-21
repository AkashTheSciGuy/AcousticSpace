"""Signal-derived RIR consistency research pipeline, not forensic verification."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import fftconvolve

from app import rir_consistency_dataset as inputs
from app.rir_features import (
    FEATURE_NAMES, estimate_reference_transfer, reference_consistency_features,
)
from app.rir_experiment import build_dataset
from scripts.generate_rir_demo_inputs import create_demo_inputs
from scripts.train_rir_detector import evaluate, train


@pytest.fixture
def controlled(tmp_path):
    root = tmp_path / "sources"
    output = tmp_path / "controlled"
    create_demo_inputs(root)
    build_dataset(root, root / "speech.csv", root / "responses.csv",
                  output, seconds=.5)
    return output


def test_estimates_transfer_from_known_broadband_reference():
    rng = np.random.default_rng(8)
    dry = rng.normal(size=8_000)
    impulse = np.zeros(4_000)
    impulse[0] = 1.0
    impulse[500] = .4
    impulse[1300] = -.2
    wet = fftconvolve(dry, impulse)
    estimate = estimate_reference_transfer(dry, wet)
    cosine = float(np.dot(estimate, impulse) / np.linalg.norm(impulse))
    assert cosine > .98


def test_acoustic_features_are_finite_and_named():
    rng = np.random.default_rng(42)
    dry_speech = rng.normal(size=8000)
    dry_noise = rng.normal(size=8000)
    r1 = np.zeros(3200)
    r1[0], r1[400], r1[1200] = 1, .4, .2
    r2 = np.zeros(3200)
    r2[0], r2[400], r2[1200] = 1, .1, .03
    features = reference_consistency_features(
        dry_speech, fftconvolve(dry_speech, r1),
        dry_noise, fftconvolve(dry_noise, r2),
    )
    assert len(FEATURE_NAMES) == 7
    assert features.as_array().shape == (7,)
    assert np.isfinite(features.as_array()).all()
    assert all(value >= 0 for name, value in features.to_dict().items()
               if name != "rt60_missing")


@pytest.mark.parametrize("dry,wet", [
    (np.zeros(1000), np.ones(2000)),
    (np.ones(1000), np.ones(1100)),
    (np.ones((1000, 1)), np.ones(2000)),
    (np.ones(1000), np.full(2000, np.nan)),
])
def test_invalid_reference_estimates_are_rejected(dry, wet):
    with pytest.raises(ValueError):
        estimate_reference_transfer(dry, wet)


def test_signal_features_exclude_oracle_rirs(controlled, monkeypatch):
    observed = []
    original_read = sf.read

    def inspected_read(path, *args, **kwargs):
        resolved = Path(path)
        assert "rirs" not in resolved.parts, "Ground-truth RIR used as a model input"
        observed.append(resolved)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(inputs.sf, "read", inspected_read)
    arrays = inputs.extract_controlled_arrays(controlled)
    assert observed
    assert set(arrays) == {"train", "val"}
    for split in ("train", "val"):
        features, labels = arrays[split]
        assert features.shape == (4, len(FEATURE_NAMES))
        assert sorted(labels.tolist()) == [0, 0, 1, 1]
        assert np.isfinite(features).all()


def test_test_waveforms_not_decoded_before_opt_in(controlled, monkeypatch):
    with (controlled / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    withheld = set()
    for row in rows:
        if row["split"] == "test":
            withheld.update((controlled / row[key]).resolve() for key in (
                "mixture", "speech_stem", "ambience_stem",
                "dry_speech", "dry_ambience"))
    original = inputs._read_mono16

    def protected_read(path):
        assert path not in withheld, "Test audio decoded before explicit opt-in"
        return original(path)

    monkeypatch.setattr(inputs, "_read_mono16", protected_read)
    arrays = inputs.extract_controlled_arrays(controlled)
    assert sorted(arrays) == ["train", "val"]


def test_training_preserves_split_boundary_and_saves_local_report(controlled, tmp_path):
    output = tmp_path / "rir_control_model.joblib"
    report = train(controlled, output)
    assert output.exists()
    assert output.with_suffix(".json").exists()
    assert set(report["metrics"]) == {"val"}
    assert "test" not in report["sample_counts"]
    assert report["features"] == list(FEATURE_NAMES)
    assert "REFERENCE-AIDED" in report["warning"]
    validation = report["metrics"]["val"]
    assert validation["samples"] == 4
    assert sum(sum(row) for row in validation["confusion_matrix_consistent_mismatched"]) == 4
    assert 0 <= validation["balanced_accuracy"] <= 1
    assert "real" not in report["label_map"]
    with pytest.raises(FileExistsError):
        train(controlled, output)


def test_fixed_threshold_evaluation_counting():
    m = evaluate(np.array([0, 0, 1, 1]), np.array([.1, .8, .7, .9]))
    assert m["confusion_matrix_consistent_mismatched"] == [[1, 1], [0, 2]]
    assert m["false_mismatch_rate"] == .5
    assert m["recall_mismatched"] == 1.0


def _mutate_csv(path, mutate):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    mutate(rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_rejects_tampered_label(controlled):
    manifest = controlled / "manifest.csv"

    def corrupt(rows):
        row = rows[0]
        row["label"] = "mismatched" if row["label"] == "consistent" else "consistent"

    _mutate_csv(manifest, corrupt)
    with pytest.raises(ValueError, match="label disagrees"):
        inputs.read_controlled_manifest(controlled)


def test_rejects_corrupted_mixture(controlled):
    with (controlled / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    path = controlled / row["mixture"]
    with path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="Mixture hash mismatch"):
        inputs.read_controlled_manifest(controlled)


def test_rejects_cross_split_speaker_reuse(controlled):
    manifest = controlled / "manifest.csv"

    def corrupt(rows):
        train_speaker = next(row["speaker_id"] for row in rows
                             if row["split"] == "train")
        for row in rows:
            if row["split"] == "val":
                row["speaker_id"] = train_speaker

    _mutate_csv(manifest, corrupt)
    with pytest.raises(ValueError, match="Speaker leakage"):
        inputs.read_controlled_manifest(controlled)
