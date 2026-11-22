r"""Benchmark AcousticSpace API processing with repeatable local audio fixtures.

From backend on Windows CMD:
  .venv\Scripts\python.exe scripts\benchmark_api.py --iterations 10 --output models\api_latency_v1.json

By default, tests non-model HTTP routes only. To require and benchmark the two
locally trained classifiers too, add --include-models.

This measures *in-process FastAPI TestClient* request durations after a warmup.
It is NOT a network, concurrency, deployment latency, or accuracy benchmark.
No audio is downloaded, and no result/model/dataset is overwritten.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
from fastapi.testclient import TestClient
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import app  # noqa: E402
from scripts.final_acceptance import _rir_fixture, _speech_fixture, wav_bytes  # noqa: E402

MAX_ITERATIONS = 30


def _post(client: TestClient, route: str, audio: bytes) -> dict:
    response = client.post(
        route, files={"file": ("fixture.wav", audio, "audio/wav")},
    )
    if response.status_code != 200:
        raise RuntimeError(f"{route} returned HTTP {response.status_code}: {response.text[:160]}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"{route} did not return a JSON object")
    return payload


def _cases(client: TestClient, include_models: bool) -> dict[str, Callable[[], object]]:
    audio = _speech_fixture()
    rir = _rir_fixture()
    rng = np.random.default_rng(563)
    speech = rng.normal(0.0, 0.06, size=8000)
    ambience = rng.normal(0.0, 0.06, size=8000)
    response = np.zeros(2000, dtype=np.float64)
    response[0] = 0.8
    response[500] = 0.2
    four_files = {
        "dry_speech": wav_bytes(speech),
        "wet_speech": wav_bytes(fftconvolve(speech, response)),
        "dry_ambience": wav_bytes(ambience),
        "wet_ambience": wav_bytes(fftconvolve(ambience, response)),
    }

    def health() -> None:
        result = client.get("/health")
        if result.status_code != 200 or result.json().get("status") != "ok":
            raise RuntimeError("Backend health endpoint failed")

    def validate() -> None:
        if _post(client, "/api/audio/validate", audio).get("status") != "valid":
            raise RuntimeError("Audio validation returned invalid status")

    def inspect() -> None:
        if _post(client, "/api/audio/inspect", audio).get("status") != "valid":
            raise RuntimeError("Audio inspection returned invalid status")

    def diagnostics() -> None:
        result = _post(client, "/api/audio/diagnostics", audio)
        if result.get("status") != "analyzed" or len(result.get("log_mel_db", [])) != 48:
            raise RuntimeError("Spectrogram diagnostics returned invalid result")

    def isolated_rir() -> None:
        result = _post(client, "/api/rir/analyze", rir)
        if result.get("status") != "analyzed" or not result.get("metrics"):
            raise RuntimeError("Isolated RIR analysis returned invalid result")

    def reference_rir() -> None:
        result = client.post(
            "/api/rir/reference-consistency",
            files={role: (role + ".wav", data, "audio/wav")
                   for role, data in four_files.items()},
        )
        if (result.status_code != 200 or result.json().get("prediction", "unexpected") is not None
                or len(result.json().get("features", {})) != 7):
            raise RuntimeError(f"Reference RIR comparison failed: HTTP {result.status_code}")

    cases = {
        "health": health,
        "validate_upload": validate,
        "inspect_audio": inspect,
        "spectrogram_diagnostics": diagnostics,
        "isolated_rir_metrics": isolated_rir,
        "reference_aided_rir": reference_rir,
    }
    if include_models:
        for variant in ("baseline", "acoustic"):
            status = client.get(f"/api/model/status?variant={variant}")
            if status.status_code != 200 or not status.json().get("available", False):
                raise RuntimeError(f"{variant} model is missing; cannot benchmark model inference")

            def model(variant: str = variant) -> None:
                result = _post(client, f"/api/model/predict?variant={variant}", audio)
                if result.get("status") != "predicted" or result.get("variant") != variant:
                    raise RuntimeError(f"{variant} model did not return expected prediction")

            cases[f"{variant}_prediction"] = model
    return cases


def run_benchmark(iterations: int = 5, *, include_models: bool = False) -> dict:
    """Time full in-process request/JSON handling after a one-call warmup."""
    if type(iterations) is not int or not 1 <= iterations <= MAX_ITERATIONS:
        raise ValueError(f"iterations must be an integer from 1 to {MAX_ITERATIONS}")
    if type(include_models) is not bool:
        raise ValueError("include_models must be true or false")
    measured: dict[str, dict] = {}
    with TestClient(app) as client:
        cases = _cases(client, include_models)
        for name, call in cases.items():
            call()  # one warmup per route, not counted
            elapsed = []
            for _ in range(iterations):
                start = time.perf_counter()
                call()
                elapsed.append((time.perf_counter() - start) * 1000)
            measured[name] = {
                "samples": iterations,
                "min_ms": round(min(elapsed), 3),
                "median_ms": round(statistics.median(elapsed), 3),
                "p95_ms": round(float(np.percentile(elapsed, 95)), 3),
                "max_ms": round(max(elapsed), 3),
            }
    return {
        "status": "measured",
        "method": "in_process_testclient_sequential_warm",
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "iterations_per_endpoint": iterations,
        "models_included": include_models,
        "fixtures": "Generated WAV test tones, random dry audio and simulated RIRs only",
        "routes": measured,
        "warning": (
            "Sequential in-process request processing only. This is not a true network "
            "latency, production load, throughput, concurrency or model accuracy test. "
            "Results must be measured on the intended deployment computer. "
            "Sample sizes below 20 are not reliable distribution estimates."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--include-models", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite existing benchmark: {args.output}")
        report = run_benchmark(args.iterations, include_models=args.include_models)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    except (ValueError, RuntimeError, OSError, FileExistsError) as exc:
        print(f"API benchmark failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
