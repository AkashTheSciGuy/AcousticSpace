"""Offline reference-aided RIR CLI tests with generated WAVs only."""
from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import fftconvolve

from scripts.compare_reference_rir import compare, load_reference, main


def _sources(tmp_path):
    rng = np.random.default_rng(91)
    speech = rng.normal(size=8000) * 0.05
    ambience = rng.normal(size=8000) * 0.05
    rir = np.zeros(2000)
    rir[0], rir[250], rir[1000] = .8, .2, .05
    signals = {
        "dry_speech": speech,
        "wet_speech": fftconvolve(speech, rir),
        "dry_ambience": ambience,
        "wet_ambience": fftconvolve(ambience, rir),
    }
    paths = {}
    for key, audio in signals.items():
        paths[key] = tmp_path / f"{key}.wav"
        sf.write(paths[key], audio, 16000, format="WAV", subtype="PCM_16")
    return paths


def _args(paths, output):
    argv = []
    for role, path in paths.items():
        argv.extend(["--" + role.replace("_", "-"), str(path)])
    return argv + ["--output", str(output)]


def test_cli_produces_provenance_and_refuses_overwrite(tmp_path):
    paths = _sources(tmp_path)
    target = tmp_path / "report.json"
    assert main(_args(paths, target)) == 0
    report = json.loads(target.read_text(encoding="utf-8"))
    assert report["status"] == "analyzed"
    assert report["prediction"] is None
    assert len(report["input_sha256"]) == 4
    assert all(len(digest) == 64 for digest in report["input_sha256"].values())
    assert len(report["features"]) == 7
    assert all(np.isfinite(list(report["features"].values())))
    assert main(_args(paths, target)) == 2
    assert json.loads(target.read_text(encoding="utf-8")) == report


def test_cli_rejects_missing_ambience_source(tmp_path):
    paths = _sources(tmp_path)
    paths["dry_ambience"].unlink()
    with pytest.raises(ValueError, match="Missing WAV"):
        compare(paths)


def test_cli_rejects_multichannel_wav(tmp_path):
    paths = _sources(tmp_path)
    data = np.zeros((1000, 2))
    data[20, :] = .5
    sf.write(paths["wet_ambience"], data, 16000, format="WAV")
    with pytest.raises(ValueError, match="mono"):
        compare(paths)


def test_cli_rejects_wrong_sample_rate(tmp_path):
    paths = _sources(tmp_path)
    sf.write(paths["wet_ambience"], np.ones(8000) * .1, 8000, format="WAV")
    with pytest.raises(ValueError, match="16-kHz"):
        compare(paths)


def test_cli_rejects_silent_reference(tmp_path):
    paths = _sources(tmp_path)
    sf.write(paths["dry_speech"], np.zeros(8000), 16000, format="WAV")
    with pytest.raises(ValueError, match="silent"):
        load_reference(paths["dry_speech"])


def test_cli_rejects_wrong_number_of_signals(tmp_path):
    paths = _sources(tmp_path)
    del paths["wet_speech"]
    with pytest.raises(ValueError, match="four"):
        compare(paths)
