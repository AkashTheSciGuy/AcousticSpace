# AcousticSpace — In-process API latency benchmark

The original internship specification calls for a **low-latency API**. The backend now contains a bounded, repeatable measurement tool. It does not invent timings; it produces a report when executed on the actual target computer.

## Run on Windows CMD

In the existing local Python virtual environment, from the backend folder:

~~~cmd
cd /d F:\Projects\AcousticSpace\backend
.venv\Scripts\python.exe scripts\benchmark_api.py --iterations 20 --output models\api_latency_windows_v1.json
~~~

The tool warms each route once and then measures 20 sequential request/JSON cycles using FastAPI's *in-process* TestClient. It measures health, WAV validation, audio inspection, log-mel diagnostics, isolated-RIR metrics and the controlled four-signal reference-aided RIR comparison. All audio is **generated** in memory. No datasets or recordings are downloaded.

For locally trained three-/six-feature classifier timing, run under a **different output filename**:

~~~cmd
.venv\Scripts\python.exe scripts\benchmark_api.py --iterations 20 --include-models --output models\api_latency_models_v1.json
~~~

This second command fails if either classifier is unavailable; it never silently reports untrained models as scored.

The JSON includes Python version, platform, route names, sample counts and minimum/median/95th-percentile/maximum processing milliseconds. Results are specific to the installed machine and the provided fixture lengths. This tool refuses to overwrite prior reports.

## What the benchmark does NOT prove

- It does **not** include live network transmission, reverse-proxy latency, browser rendering, or internet upload time.
- It does **not** measure parallel throughput, concurrent-user capacity, stress load or GPU AST inference.
- It does **not** measure accuracy, source separation, blind-RIR recovery, breath detection or segment-localization quality.
- Generated 1-s audio fixtures may process differently from actual 60-s real-world recordings.
- A fixed pass/fail service-level objective cannot be claimed unless the internship's accepted hardware, clip durations, traffic and latency target are defined in advance.

The small default 5-call run is a smoke measurement, not a stable latency distribution. Use at least 20 iterations and preserve hardware information separately for presentation.

The correct internship status is **measurement instrumentation implemented; target-machine benchmark result pending** until the tool has been executed locally and the actual JSON has been reviewed.
