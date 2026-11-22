"""Integration checks for reproducible in-process API latency reports."""
import json

import pytest

from scripts.benchmark_api import main, run_benchmark


def test_benchmark_measures_real_http_routes_without_models(tmp_path, monkeypatch):
    monkeypatch.setenv("ACOUSTICSPACE_BASELINE_MODEL", str(tmp_path / "absent.joblib"))
    monkeypatch.setenv("ACOUSTICSPACE_ACOUSTIC_MODEL", str(tmp_path / "absent2.joblib"))
    result = run_benchmark(iterations=2)
    assert result["status"] == "measured"
    assert result["models_included"] is False
    assert result["iterations_per_endpoint"] == 2
    assert "reference_aided_rir" in result["routes"]
    assert "isolated_rir_metrics" in result["routes"]
    assert "baseline_prediction" not in result["routes"]
    for route in result["routes"].values():
        assert route["samples"] == 2
        assert 0 <= route["min_ms"] <= route["max_ms"]
        assert 0 <= route["median_ms"] <= route["max_ms"]
        assert 0 <= route["p95_ms"] <= route["max_ms"]


@pytest.mark.parametrize("iterations", [0, -1, 31, 1.0, "2", True])
def test_benchmark_rejects_invalid_iteration_count(iterations):
    with pytest.raises(ValueError, match="iterations"):
        run_benchmark(iterations)


def test_benchmark_optional_models_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("ACOUSTICSPACE_BASELINE_MODEL", str(tmp_path / "missing.joblib"))
    monkeypatch.setenv("ACOUSTICSPACE_ACOUSTIC_MODEL", str(tmp_path / "missing2.joblib"))
    with pytest.raises(RuntimeError, match="model is missing"):
        run_benchmark(1, include_models=True)


def test_benchmark_cli_creates_report_once(tmp_path, capsys):
    path = tmp_path / "api_latency.json"
    assert main(["--iterations", "1", "--output", str(path)]) == 0
    previous = path.read_bytes()
    report = json.loads(previous)
    assert report["method"] == "in_process_testclient_sequential_warm"
    assert report["routes"]["health"]["samples"] == 1
    assert main(["--iterations", "1", "--output", str(path)]) == 2
    assert path.read_bytes() == previous
    assert "Refusing to overwrite" in capsys.readouterr().err


def test_benchmark_cli_rejects_invalid_options(tmp_path):
    path = tmp_path / "invalid.json"
    assert main(["--iterations", "0", "--output", str(path)]) == 2
    assert not path.exists()
