# Verification evidence — 20 November 2026

**Verified code revision:** `88343950fb3267c66f2ed6b50268ba4ce5f2e2f3` on PR #28, branch `audit/workflow-completion`.

[GitHub Actions run 44786956556](https://github.com/AkashTheSciGuy/AcousticSpace/actions/runs/44786956556) completed successfully for this exact code head. A later documentation-only update records these results; it does not change application or test code.

| Check | Actual result and scope |
| --- | --- |
| Backend pytest | **116 passed**, locally and on GitHub Actions; CI duration 5.48 s; one upstream deprecation warning |
| Frontend production build | **Passed**, locally and in CI |
| Frontend lint | **Passed**, no application source warnings |
| Chromium browser suite | **4 passed** in CI: inspection; playback/pause/seek/remove; both classifier variants; saved metrics; RIR results; comparison; history clear/reload; invalid uploads; network failure/retry; malformed status recovery; pending prediction cancellation; mobile layout |
| Docker | **Passed**: multi-stage image build; built UI at `/`; `/health`; model status correctly unavailable without bundled model files |
| Same-origin production serving | **Passed** via local FastAPI TestClient: built HTML, JS/CSS assets, health/OpenAPI and model status |
| Application endpoint integration | **Passed**: `/`, `/health`, docs/Redoc/OpenAPI, validate, inspect, both model status/prediction variants, comparison and RIR; existing suite also covers invalid inputs and unavailable artifacts |
| Real MP3 encoding/decoding smoke | **Passed locally**, FFmpeg-encoded synthetic tone: validate/inspect and both model predictions returned 200; RIR correctly rejected MP3 with 415 |
| Training regression | **Passed**: both actual training entry points produce fitted three-/six-feature artifacts from tiny synthetic grouped manifests with padded CSV fields |
| Git artifact exclusion | No datasets, CSV manifests, generated WAV/MP3/FLAC recordings or joblib model binaries tracked or added |
| Original artifact release preflight | **Incomplete, exit 2**: neither original model, original comparison JSON nor original baseline evaluation JSON was available |

## PR #27 review

Base reviewed: `e352f104fac33a5531193c7908fe8fa520f29bc5` (main). PR #27 was **already merged** when this audit began; no second merge was attempted. Its exact-head GitHub Actions run 44781557342 concluded success. Its diff addressed corrupt deserialization, stale UI errors, ignored model artifacts, CI and truthful setup guidance. This audit found and fixed additional metadata, report-consistency, training-normalization and frontend robustness issues; see [workflow audit](WORKFLOW_AUDIT.md).

## What model verification means here

The original two trained models, dataset, manifest and saved comparison/evaluation JSON are deliberately absent from Git and were **not supplied** in this environment. A default `scripts/verify_release.py` run exits 2 and reports `incomplete` for both model loads and reports. The historical 70.63%/68.75% test accuracies were **not independently reproduced**. A clean checkout is operational for inspection/RIR analysis but is not ready for a demo using the original classifier artifacts until those are installed.

Backend and browser integration tests generate tiny trusted synthetic logistic-regression artifacts. They test actual loading, ordered three/six-feature inference, API scores and UI behavior. Their predictions and fixture reports are not research results and never replace the original trained models. Synthetic RIR decay tests validate numerical behavior, not accuracy on real rooms or fake speech.

Runtime used locally: Python 3.12, Node 24.22.3, FastAPI 0.165.0, librosa 1.0.1, NumPy 2.5.5, scikit-learn 1.9.1, joblib 1.6.0. The backend has one upstream Starlette/httpx deprecation warning, with no accompanying application failure. Existing joblib files must be loaded in a compatible trusted training environment or retrained; these tested package versions do not prove compatibility with missing historical artifacts.

## Reproduce on Windows CMD

Install dependencies as described in the root README. From the backend directory:

```cmd
cd /d F:\Projects\AcousticSpace\backend
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\verify_release.py --audio PATH_TO_REAL_TEST_CLIP.wav
```

The second command checks original installed models/reports and executes predictions on the supplied clip. It is an inference smoke test, not a labeled accuracy experiment. Install trusted original artifacts under `backend/models/` first, or use the operator overrides documented in the [API contract](API_CONTRACT.md).

Browser suite (frontend directory; Python must resolve to the ML environment):

```cmd
cd /d F:\Projects\AcousticSpace\frontend
npm ci
npx playwright install chromium
set ACOUSTICSPACE_TEST_PYTHON=F:\Projects\AcousticSpace\backend\.venv\Scripts\python.exe
npm run test:e2e
```

Stop other development servers on ports 8000 and 5173 first. Tests start dedicated servers using ignored `.e2e` synthetic artifacts, never the user's models. Optional `ACOUSTICSPACE_CHROMIUM` accepts an absolute path to an already-installed Chromium executable.

## Limits of this verification

Local browser launch was blocked by the workspace OS; the separate browser could not reach its loopback server, and Docker is not installed locally. The successful browser and Docker results above came from GitHub Actions, not local execution. Chromium automation does not certify every browser, device, accessibility requirement or recording format variant. There is no measured concurrency/latency SLA, public production deployment or independent research evaluation.

See the [complete requirement-by-requirement limitations report](WORKFLOW_AUDIT.md) for AST/CNN training, speech-to-RIR inference, breathing/alignment and segment localization gaps. The implemented classifiers are **speech-acoustic logistic regressions**; the isolated-RIR analyzer is a separate measurement tool. No RIR-based deepfake-detection accuracy is claimed.
