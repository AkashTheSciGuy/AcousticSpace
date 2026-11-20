# Verification evidence — 20 November 2026

Base reviewed: `e352f104fac33a5531193c7908fe8fa520f29bc5` (main). PR #27 was already merged when this audit began; no second merge was attempted. Its exact-head GitHub Actions run 44781557342 concluded success. Its diff addressed corrupt deserialization, stale UI errors, ignored model artifacts, CI and truthful setup guidance.

Initial clean install: backend **104 passed**; frontend build passed; lint reported two React effect warnings. After the audit fixes, backend **112 passed** before the additional all-endpoint integration test; frontend build and lint passed without source warnings. Final checks will be recorded after the CI run.

Runtime used: Python 3.12, Node 24.22.3, FastAPI 0.165.0, librosa 1.0.1, NumPy 2.5.5, scikit-learn 1.9.1, joblib 1.6.0. Pytest emits one upstream Starlette/httpx deprecation warning; no application failure accompanies it.

## What model verification means here

The original two trained models, dataset, manifest and saved comparison/evaluation JSON are deliberately absent from Git and were **not supplied** in this environment. A default `scripts/verify_release.py` run exits 2 and reports `incomplete` for both model loads and reports. The historical 70.63%/68.75% test accuracies were **not independently reproduced**.

Backend and browser integration tests generate tiny trusted synthetic logistic-regression artifacts. They test actual loading, ordered three/six-feature inference, API scores and UI behavior. Their predictions and fixture reports are not research results and never replace the original trained models.

## Reproduce on Windows CMD

With the backend environment active and dependencies installed:

```cmd
cd /d F:\Projects\AcousticSpace\backend
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\verify_release.py --audio PATH_TO_REAL_TEST_CLIP.wav
```

The second command checks original installed models/reports and executes predictions on the supplied clip. It is an inference smoke test, not a labeled accuracy experiment.

Browser suite (frontend directory; Python executable must resolve to the ML environment):

```cmd
npm ci
npx playwright install chromium
set ACOUSTICSPACE_TEST_PYTHON=F:\Projects\AcousticSpace\backend\.venv\Scripts\python.exe
npm run test:e2e
```

Stop other development servers on ports 8000 and 5173 first. Tests start dedicated servers using ignored `.e2e` synthetic artifacts, never the user's models. An optional `ACOUSTICSPACE_CHROMIUM` absolute executable path supports an already-installed Chromium.

## Environment limits

This workspace cannot launch system Chromium (OS socket operation denied), the browser download host returned 403, and the separate browser could not reach the workspace's loopback server. Docker is not installed here. Browser and image smoke tests are therefore included in GitHub Actions; no local pass is claimed for either. Actual outcomes will be recorded below after CI completes.
