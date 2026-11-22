"""Regression tests for bounded in-process concurrent audio processing."""
import json

import pytest

from scripts.benchmark_concurrency import main, run_benchmark


def test_parallel_inspection_requests_complete_and_report_measured_latency():
    report = run_benchmark(route="inspect", concurrency=2, batches=2, clip_seconds=1)
    assert report["status"] == "measured"
    assert report["route"] == "inspect"
    assert report["requests"] == 4
    assert report["concurrency"] == 2
    assert report["batches"] == 2
    assert report["warmup_requests_excluded"] == 1
    assert 0 < report["min_request_ms"] <= report["max_request_ms"]
    assert report["min_request_ms"] <= report["median_request_ms"] <= report["max_request_ms"]
    assert report["min_request_ms"] <= report["p95_request_ms"] <= report["max_request_ms"]
    assert report["throughput_requests_per_second"] > 0
    assert "without a real network" in report["warning"]


def test_parallel_spectrogram_requests_are_measured_without_model_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("ACOUSTICSPACE_BASELINE_MODEL", str(tmp_path / "missing.joblib"))
    report = run_benchmark(route="diagnostics", concurrency=2, batches=1, clip_seconds=1)
    assert report["requests"] == 2
    assert report["asgi_path"] == "/api/audio/diagnostics"


@pytest.mark.parametrize("options", [
    {"route": "not-a-route"},
    {"concurrency": 0}, {"concurrency": 9}, {"concurrency": 1.5},
    {"concurrency": True}, {"batches": 0}, {"batches": 21},
    {"clip_seconds": 0}, {"clip_seconds": 61},
    {"clip_seconds": True},
    {"concurrency": 8, "batches": 20, "clip_seconds": 60},
])
def test_benchmark_rejects_invalid_or_excessive_load(options):
    with pytest.raises(ValueError):
        run_benchmark(**options)


def test_concurrent_benchmark_cli_refuses_to_overwrite_report(tmp_path, capsys):
    target = tmp_path / "concurrent.json"
    argv = ["--concurrency", "2", "--batches", "1", "--output", str(target)]
    assert main(argv) == 0
    previous = target.read_bytes()
    result = json.loads(previous)
    assert result["requests"] == 2
    assert result["method"] == "single_process_httpx_asgi_transport_concurrent_batches"
    assert main(argv) == 2
    assert target.read_bytes() == previous
    assert "Refusing to overwrite" in capsys.readouterr().err
