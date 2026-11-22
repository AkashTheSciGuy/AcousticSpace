r"""Measure bounded simultaneous AcousticSpace ASGI audio requests.

Windows CMD (from backend, after installing requirements-dev.txt):
  .venv\Scripts\python.exe scripts\benchmark_concurrency.py --route inspect ^
    --concurrency 4 --batches 5 --clip-seconds 5 ^
    --output models\concurrency_local_v1.json

Uses httpx.AsyncClient and ASGITransport in one Python process. No socket
or deployed server is involved; this measures ASGI processing, not true
network end-to-end latency, GPU inference, or forensic accuracy.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import app  # noqa: E402
from scripts.benchmark_api import speech_fixture  # noqa: E402

ROUTES = {
    "inspect": ("/api/audio/inspect", "valid"),
    "diagnostics": ("/api/audio/diagnostics", "analyzed"),
}
MAX_TOTAL_CLIP_SECONDS = 1200


async def _post(client: httpx.AsyncClient, route: str, expected: str,
                audio: bytes) -> float:
    start = time.perf_counter()
    response = await client.post(
        route, files={"file": ("generated.wav", audio, "audio/wav")},
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    if response.status_code != 200:
        raise RuntimeError(f"{route} returned HTTP {response.status_code}: {response.text[:120]}")
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("status") != expected:
        raise RuntimeError(f"{route} did not return the expected analysis response")
    return elapsed_ms


def _validate(route: str, concurrency: int, batches: int, clip_seconds: int) -> None:
    if route not in ROUTES:
        raise ValueError("route must be inspect or diagnostics")
    if type(concurrency) is not int or not 1 <= concurrency <= 8:
        raise ValueError("concurrency must be an integer from 1 to 8")
    if type(batches) is not int or not 1 <= batches <= 20:
        raise ValueError("batches must be an integer from 1 to 20")
    if type(clip_seconds) is not int or not 1 <= clip_seconds <= 60:
        raise ValueError("clip_seconds must be an integer from 1 to 60")
    if concurrency * batches * clip_seconds > MAX_TOTAL_CLIP_SECONDS:
        raise ValueError("Combined workload exceeds 1200 generated clip-seconds")


async def measure(*, route: str = "inspect", concurrency: int = 2,
                  batches: int = 3, clip_seconds: int = 1) -> dict:
    """Benchmark sequential batches of concurrent, in-process ASGI requests."""
    _validate(route, concurrency, batches, clip_seconds)
    path, expected = ROUTES[route]
    audio = speech_fixture(clip_seconds)
    transport = httpx.ASGITransport(app=app)
    timings: list[float] = []
    async with httpx.AsyncClient(
        transport=transport, base_url="http://acousticspace.test",
        timeout=httpx.Timeout(120),
    ) as client:
        await _post(client, path, expected, audio)  # warmup (excluded)
        wall_start = time.perf_counter()
        for _ in range(batches):
            batch = await asyncio.gather(*(
                _post(client, path, expected, audio)
                for _ in range(concurrency)
            ))
            timings.extend(batch)
        wall_ms = (time.perf_counter() - wall_start) * 1000
    total = len(timings)
    return {
        "status": "measured",
        "method": "single_process_httpx_asgi_transport_concurrent_batches",
        "route": route,
        "asgi_path": path,
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "clip_seconds": clip_seconds,
        "concurrency": concurrency,
        "batches": batches,
        "requests": total,
        "warmup_requests_excluded": 1,
        "wall_time_ms": round(wall_ms, 3),
        "min_request_ms": round(min(timings), 3),
        "median_request_ms": round(statistics.median(timings), 3),
        "p95_request_ms": round(float(np.percentile(timings, 95)), 3),
        "max_request_ms": round(max(timings), 3),
        "throughput_requests_per_second": round(total * 1000 / wall_ms, 3),
        "warning": (
            "Generated test tones are not real speech. This benchmark measures "
            "concurrent processing in one ASGI process without a real network, "
            "HTTP socket, proxy, GPU model, or genuine spoof recordings. Results "
            "are hardware- and workload-dependent; they do not establish "
            "multi-worker production capacity, model accuracy, or forensic validity."
        ),
    }


def run_benchmark(*, route: str = "inspect", concurrency: int = 2,
                  batches: int = 3, clip_seconds: int = 1) -> dict:
    _validate(route, concurrency, batches, clip_seconds)
    return asyncio.run(measure(
        route=route, concurrency=concurrency,
        batches=batches, clip_seconds=clip_seconds,
    ))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=sorted(ROUTES), default="inspect")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--clip-seconds", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite existing report: {args.output}")
        report = run_benchmark(
            route=args.route, concurrency=args.concurrency,
            batches=args.batches, clip_seconds=args.clip_seconds,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    except (OSError, ValueError, RuntimeError, FileExistsError) as exc:
        print(f"Concurrent API benchmark failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
