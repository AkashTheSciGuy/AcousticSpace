# AcousticSpace Day 3 — optional six-feature candidate

> Historical user-recorded experiment notes. Original data, model artifacts and report JSON were unavailable for the 20 November audit; these results have not been independently reproduced. See [current verification evidence](../docs/VERIFICATION.md).
The original three-feature baseline remains the default for `/api/model/status` and `/api/model/predict`.
Day 3 adds an **opt-in**, separately trained six-feature speech-acoustic candidate. Neither model is an RIR estimator or a forensic tool.

## Train on the same grouped dataset

From Windows CMD in `F:\Projects\AcousticSpace\backend`:

```cmd
.venv\Scripts\python.exe scripts\train_acoustic_candidate.py "F:\Datasets\ASVspoof2019\LA" "F:\Datasets\ASVspoof2019\manifest.csv" --output models\acoustic_candidate.joblib
```

This verifies the manifest, extracts all six features, and saves the trusted local model to `backend/models/acoustic_candidate.joblib`. Existing model files are **not overwritten**.

The prior `models/acoustic_comparison.json` contains evaluation of the baseline versus candidate on the same 400/140/160 grouped samples. It is not itself a trained inference model. The candidate is **not automatically selected based on the test results**.

## Run the API

```cmd
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

- `GET /api/model/status` — default baseline
- `GET /api/model/status?variant=acoustic` — candidate readiness
- `POST /api/model/predict` — default baseline inference
- `POST /api/model/predict?variant=acoustic` — candidate inference, same WAV/MP3 multipart `file` field

The frontend selector offers both variants, but only allows prediction when the selected model is available. The candidate can also be loaded from a separately controlled local path via `ACOUSTICSPACE_ACOUSTIC_MODEL`.

**Research limitations:** Six features are derived from speech texture and energy variation, *not* measured room impulse responses. ASVspoof 2019 LA training and development splits do not establish field generalization. The official evaluation set was not used. The candidate achieved 76.67% fake recall but only 68.75% overall accuracy on the 160-recording exploratory test split, based on the user's local comparison. No deployment selection is justified by this single experiment.
