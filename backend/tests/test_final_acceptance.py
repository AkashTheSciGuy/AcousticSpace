"""Regression tests for the final local demo acceptance tool."""
from __future__ import annotations

import json

from scripts.final_acceptance import check_endpoints, main, wav_bytes

import numpy as np
import pytest


def test_core_demo_acceptance_passes_without_trained_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("ACOUSTICSPACE_BASELINE_MODEL", str(tmp_path / "missing_base.joblib"))
    monkeypatch.setenv("ACOUSTICSPACE_ACOUSTIC_MODEL", str(tmp_path / "missing_candidate.joblib"))
    report = check_endpoints()
    assert report["status"] == "passed"
    assert report["checks"]["health"]["passed"]
    assert report["checks"]["audio_inspection"]["passed"]
    assert report["checks"]["spectrogram_diagnostics"]["passed"]
    assert report["checks"]["isolated_rir_analysis"]["passed"]
    assert report["checks"]["invalid_upload_rejection"]["passed"]
    assert report["checks"]["baseline_model"]["available"] is False
    assert report["checks"]["experimental_window_scoring"]["available"] is False


def test_model_required_mode_fails_when_models_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("ACOUSTICSPACE_BASELINE_MODEL", str(tmp_path / "missing_base.joblib"))
    monkeypatch.setenv("ACOUSTICSPACE_ACOUSTIC_MODEL", str(tmp_path / "missing_candidate.joblib"))
    report = check_endpoints(require_models=True)
    assert report["status"] == "failed"
    assert report["checks"]["baseline_model"]["passed"] is False
    assert report["checks"]["acoustic_model"]["passed"] is False


def test_cli_writes_report_once_and_rejects_overwrite(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ACOUSTICSPACE_BASELINE_MODEL", str(tmp_path / "missing_base.joblib"))
    monkeypatch.setenv("ACOUSTICSPACE_ACOUSTIC_MODEL", str(tmp_path / "missing_candidate.joblib"))
    target = tmp_path / "acceptance.json"
    assert main(["--output", str(target)]) == 0
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["status"] == "passed"
    assert main(["--output", str(target)]) == 2
    assert "Refusing to overwrite" in capsys.readouterr().err
    assert json.loads(target.read_text(encoding="utf-8")) == data


def test_fixture_rejects_invalid_sample_data():
    with pytest.raises(ValueError):
        wav_bytes(np.zeros(10))
    with pytest.raises(ValueError):
        wav_bytes(np.array([np.nan] * 512))
    with pytest.raises(ValueError):
        wav_bytes(np.full(512, 1.2))
