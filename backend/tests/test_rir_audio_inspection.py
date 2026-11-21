"""Regression coverage for bounded, read-only measured RIR header auditing."""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import soundfile as sf

from scripts.inspect_rir_audio import inspect, main


def _wav(sample_rate: int, channels: int) -> bytes:
    audio = np.zeros((sample_rate, channels), dtype=np.float32)
    audio[8] = 0.75
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def test_archive_header_inspection_includes_multichannel_and_invalid_entries(tmp_path):
    archive = tmp_path / "rirs.zip"
    prefix = "RIRS_NOISES/real_rirs_isotropic_noises/"
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr(
            prefix + "RVB2014_type1_rir_smallroom1_near_angla.wav",
            _wav(16000, 8),
        )
        writer.writestr(
            prefix + "RWCP_type1_rir_circle_e1b_imp001.wav",
            _wav(8000, 1),
        )
        writer.writestr(
            prefix + "air_type1_air_binaural_booth_0_1.wav",
            b"not a WAV file",
        )
    report = inspect(archive)
    assert report["candidate_files"] == 3
    assert report["valid_wav_headers"] == 2
    assert report["invalid_wav_headers"] == 1
    assert report["valid_by_family"] == {"RVB2014": 1, "RWCP": 1}
    assert report["sample_rates_hz"] == {"8000": 1, "16000": 1}
    assert report["channel_counts"] == {"1": 1, "8": 1}
    assert report["duration_seconds_min"] == 1
    assert report["duration_seconds_max"] == 1
    assert "independent physical rooms" in report["note"]


def test_header_audit_writes_json_and_refuses_overwrite(tmp_path):
    archive = tmp_path / "rirs.zip"
    prefix = "RIRS_NOISES/real_rirs_isotropic_noises/"
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr(
            prefix + "RVB2014_type1_rir_smallroom1_near_angla.wav", _wav(16000, 1),
        )
    output = tmp_path / "report.json"
    assert main([str(archive), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["valid_wav_headers"] == 1
    assert main([str(archive), "--output", str(output)]) == 2
    assert json.loads(output.read_text(encoding="utf-8")) == report
