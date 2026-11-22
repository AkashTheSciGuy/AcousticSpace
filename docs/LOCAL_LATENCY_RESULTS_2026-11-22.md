# AcousticSpace — Reported Windows API processing results

**Date:** 22 November 2026  
**Source:** User-provided output from a local AcousticSpace run on Windows with Python 3.14.7  
**Command:** scripts/benchmark_api.py with --iterations 20, --include-models and output models\api_latency_windows_v1.json  
**Method:** in_process_testclient_sequential_warm — one warmup request per endpoint, followed by 20 sequential samples per route

The local verification for the same revision reported **214 backend pytest tests passed**, **13 API acceptance checks passed**, the frontend production build succeeded, and frontend lint had 0 errors. Both experimental baseline artifacts were installed locally.

## Processing measurements

These values come from the **reported run**, not from an independently reproduced benchmark on a separate machine.

| Operation | Median (ms) | p95 (ms) | Maximum (ms) |
| --- | ---: | ---: | ---: |
| Health | 1.603 | 4.079 | 44.461 |
| Upload validation | 1.479 | 1.732 | 1.739 |
| Audio feature inspection | 10.564 | 11.145 | 11.209 |
| Log-mel spectrogram diagnostics | 12.259 | 14.632 | 14.807 |
| Isolated impulse-response metrics | 10.667 | 12.166 | 12.365 |
| Four-signal reference-aided RIR comparison | 3.836 | 4.139 | 4.173 |
| Three-feature baseline prediction | 12.015 | 13.029 | 13.408 |
| Six-feature acoustic candidate prediction | 13.976 | 15.093 | 15.146 |

The health route's 44.461 ms maximum compared with its 1.603 ms median suggests an isolated slow measurement, but no cause was established. Twenty samples per route are insufficient to characterize tail latency reliably.

## Interpretation and caveats

1. The audio used **generated test tones**, random references and simulated isolated room impulse responses. These are not recorded genuine or manipulated speech.
2. Audio-validation, inspection, diagnostics and classifier routes were exercised with **one-second input**; the RIR and four-reference test cases have other fixed lengths. Do not extrapolate these numbers to 60-second clips.
3. In-process TestClient timing excludes internet transfer, real TCP network overhead, frontend rendering, reverse proxies, concurrent users and production deployment effects.
4. No classification accuracy was measured. A fast incorrect prediction is still incorrect.
5. The baseline and candidate models were confirmed loadable, not independently validated. The reference-aided RIR feature comparison **does not** recover a room response from a single mixed user upload.

The original report remains local under backend/models and is not committed to GitHub. This summary preserves the explicitly supplied metrics for internship documentation.

## Repeatable extended measurements

After pulling the latest main, run from backend in Windows CMD:

~~~cmd
.venv\Scripts\python.exe scripts\benchmark_api.py --clip-seconds 10 --iterations 20 --include-models --output models\api_latency_windows_10s.json
~~~

To measure **in-process concurrent ASGI processing**, use the bounded workload tool:

~~~cmd
.venv\Scripts\python.exe scripts\benchmark_concurrency.py --route inspect --concurrency 4 --batches 5 --clip-seconds 5 --output models\concurrency_inspect_v1.json
.venv\Scripts\python.exe scripts\benchmark_concurrency.py --route diagnostics --concurrency 2 --batches 5 --clip-seconds 5 --output models\concurrency_diagnostics_v1.json
~~~

The concurrency tool uses httpx.AsyncClient and ASGITransport. It needs requirements-dev.txt (already installed for the existing pytest suite), does **not** launch a network server and does **not** represent real multi-worker production capacity. It measures request counts, per-request median/p95, total in-process wall time and approximate throughput for small, bounded runs.

Every output file must have a **new name**. The tools refuse to overwrite prior measurements so research evidence remains intact.

## Scientific acceptance still pending

This report closes an instrumentation gap but does not close the central research requirements: validating blind source-specific RIR inference, obtaining independently verified room metadata and new held-out performance, and testing AST and breath/phonetic consistency against independent labeled recordings. See docs/PROJECT1_REQUIREMENTS.md and GitHub issue #41.
