"""Synthetic-fixture tests for the non-destructive RIR signal audit."""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import soundfile as sf
import pytest

from scripts.audit_rir_signal_quality import (
    audit_signals, inspect_samples, main,
)

PREFIX = "RIRS_NOISES/real_rirs_isotropic_noises/"


def _wav(data: np.ndarray) -> bytes:
    stream = io.BytesIO()
    sf.write(stream, data, 16000, format="WAV", subtype="PCM_16")
    return stream.getvalue()


def test_signal_metrics_identify_short_silent_and_sustained_audio():
    impulse = np.zeros(16000, dtype=np.float32)
    impulse[4] = 0.8
    result = inspect_samples(impulse, 16000)
    assert result["flags"] == []
    assert result["peak_sample_index"] == 4
    assert result["last_10_percent_energy_fraction"] == 0

    short = np.zeros(1600, dtype=np.float32)
    short[4] = 0.5
    assert "shorter_than_250_ms" in inspect_samples(short, 16000)["flags"]

    silent = np.zeros(16000, dtype=np.float32)
    assert "near_silent_channel" in inspect_samples(silent, 16000)["flags"]
    assert inspect_samples(silent, 16000)["last_10_percent_energy_fraction"] is None

    constant = np.full(16000, 1.0, dtype=np.float32)
    flags = inspect_samples(constant, 16000)["flags"]
    assert "high_last_10_percent_energy" in flags
    assert "possible_full_scale_clipping" in flags

    with pytest.raises(ValueError):
        inspect_samples(np.array([np.nan]), 16000)


def test_archive_audit_uses_selected_channel_without_downmix(tmp_path):
    archive = tmp_path / "measured.zip"
    stereo = np.zeros((16000, 2), dtype=np.float32)
    stereo[4, 1] = 0.7
    multi = np.zeros((16000, 16), dtype=np.float32)
    multi[4, 0] = 0.8
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr(
            PREFIX + "air_type1_air_binaural_booth_0_1.wav", _wav(stereo),
        )
        writer.writestr(
            PREFIX + "RWCP_type1_rir_circle_e1b_imp001.wav", _wav(multi),
        )
    report1 = audit_signals(archive, channel=1)
    assert report1["candidate_files"] == 2
    assert report1["analyzed_files"] == 2
    assert report1["decode_errors"] == 0
    files1 = {item["family"]: item for item in report1["files"]}
    assert files1["AIR"]["source_channels"] == 2
    assert "near_silent_channel" in files1["AIR"]["flags"]
    assert files1["RWCP"]["flags"] == []
    report2 = audit_signals(archive, channel=2)
    files2 = {item["family"]: item for item in report2["files"]}
    assert files2["AIR"]["flags"] == []
    assert "near_silent_channel" in files2["RWCP"]["flags"]
    assert files1["AIR"]["selected_audio_sha256"] != files2["AIR"]["selected_audio_sha256"]


def test_cli_refuses_to_replace_existing_report(tmp_path):
    archive = tmp_path / "measured.zip"
    impulse = np.zeros(16000, dtype=np.float32)
    impulse[2] = 0.5
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr(
            PREFIX + "RVB2014_type1_rir_smallroom1_near_angla.wav",
            _wav(impulse),
        )
    output = tmp_path / "quality.json"
    assert main([str(archive), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["analyzed_files"] == 1
    assert main([str(archive), "--output", str(output)]) == 2
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_invalid_selected_channel_is_reported_not_silently_mixed(tmp_path):
    archive = tmp_path / "one.zip"
    mono = np.zeros(16000, dtype=np.float32)
    mono[4] = 0.5
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr(
            PREFIX + "RVB2014_type1_rir_smallroom1_near_angla.wav",
            _wav(mono),
        )
    report = audit_signals(archive, channel=2)
    assert report["analyzed_files"] == 0
    assert report["decode_errors"] == 1
    assert "exceeds" in report["errors"][0]["error"]
