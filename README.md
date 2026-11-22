# AcousticSpace

**Local audio-forensics research prototype.** AcousticSpace inspects speech recordings, compares two experimental deepfake classifiers, and analyzes *separately supplied* room impulse response (RIR) WAV recordings.

> The current classifiers are **not RIR-based detectors**. They use either three standard signal features or six exploratory speech-acoustic descriptors with logistic regression. Predictions and class scores are not forensic proof.

## Implemented features

- React + TypeScript + Vite dashboard and WaveSurfer waveform player.
- WAV/MP3 file upload validation (up to 20 MiB) and audio inspection (up to 60 seconds).
- RMS, zero-crossing rate, spectral centroid, spectral flatness, energy variation and spectral flux.
- Isolated RIR analysis: T20-derived RT60, clarity C50 and approximate direct/reverberant energy ratio.
- Trained local baseline and opt-in six-feature experimental candidate, both served through FastAPI.
- Validation/test metric reports, confusion matrices and the feature comparison dashboard.
- Exploratory overlapping-window score timeline, session-only metadata history, and log-mel spectrogram/cadence observations (not validated spoof localization or breath detection).
- Optional Hugging Face AST checkpoint training/inference interfaces; a local one-epoch head-only experiment was reported but achieved weak validation performance. Model weights are not included in GitHub.
- Docker Compose local demo with Nginx and FastAPI, plus GitHub Actions CI.

## Start locally (Windows CMD)

**Backend**, in one terminal:

```cmd
cd /d F:\Projects\AcousticSpace\backend
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -r requirements-ml.txt
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open API docs at http://localhost:8000/docs .

**Frontend**, in a separate terminal:

```cmd
cd /d F:\Projects\AcousticSpace\frontend
npm ci
npm run build
npm run dev
```

Open http://localhost:5173 . Vite proxies `/api` to the backend on port 8000.

## Final one-command demo verification (Windows)

From the repository root, run `verify_local_demo.cmd` to execute backend tests, in-process API smoke tests, a frontend production build, and lint. For a demonstration that **requires** trained baseline and candidate classifiers, run `verify_local_demo.cmd --require-models`; this fails if local models or the comparison report are missing. The smoke tests do not measure forensic or model accuracy.

See [Final handoff](docs/FINAL_HANDOFF.md) for the demo sequence, local acceptance criteria, recorded model limitations, and what to submit.

## Local model training and verification

See [backend/DATASET_TRAINING.md](backend/DATASET_TRAINING.md), [backend/DAY3_CANDIDATE.md](backend/DAY3_CANDIDATE.md) and [backend/DAY20_DELIVERY.md](backend/DAY20_DELIVERY.md).

Run from `backend` after training the models:

```cmd
.venv\Scripts\python.exe scripts\verify_release.py
```

The `backend/models/` directory contains generated artifacts and reports. **Do not commit datasets, manifest files or trained `.joblib` models.** Only load trusted, self-generated joblib models.

## Measured-RIR follow-up (Phase 2C)

A reproducible experiment combining ARU anechoic speech with measured RVB2014 room responses is documented in [Phase 2C](docs/RIR_PHASE2C_REPRODUCIBILITY.md). Despite **93.75% validation accuracy**, its 16-sample held-out test was **50% accurate (ROC-AUC 0.5625)**. That is a negative generalization result, *not* evidence of a deployable RIR-based deepfake detector. The reference-aided classifier needs dry signals and isolated wet stems, which ordinary uploads do not provide.

Preparation scripts, archive audit, and train/validation-only ablation diagnostics are now version-controlled under `backend/scripts/`. Do not tune on this previously viewed test split.

## Experimental evaluation

The recorded local ASVspoof 2019 LA subset contained **400 train / 140 validation / 160 test recordings**.

| Test-set measure | Three-feature baseline | Six-feature candidate |
| --- | ---: | ---: |
| Accuracy | 70.63% | 68.75% |
| Fake-class recall | 63.33% | 76.67% |
| Fake-class F1 | 61.79% | 64.79% |

This was a within-dataset exploratory comparison. The official evaluation partition was not used, and generalization to unknown recordings has not been verified. The candidate detects more spoofed samples in this test but also increases false positives.

## Containerized local demo

See [docs/DOCKER.md](docs/DOCKER.md). The demo uses Docker Compose with a read-only local model mount and Nginx proxy. Images do not bundle datasets or weights.

## What remains unverified or unfinished

The original PDF calls for a **fine-tuned** AST model, physics-based RIR extraction from speech, true syllable/breath alignment, validated suspicious-segment localization, robust independent accuracy and deployment readiness. A one-epoch, **head-only** AST training run was reported locally (47.14% validation accuracy on 140 clips), but end-to-end encoder fine-tuning and independent accuracy have not been established. The spectrogram, timing peaks and window scores are observational/experimental rather than validated physical or forensic inference. The RIR analyzer remains separate and requires a measured or simulated impulse response. Public production hardening and external-dataset evaluation remain research/engineering work. Windows in-process timing and concurrency measurements were recorded, but networked production performance is not established.

For actual annotated breath/syllable and tamper-window evaluation, see [Independent annotation validation](docs/ANNOTATED_VALIDATION.md). No independently verified breath/tamper recordings have been provided; the evaluation tooling does not fabricate labels.

See [Project 1 requirements checklist](docs/PROJECT1_REQUIREMENTS.md) and [AST training instructions](backend/AST_TRAINING.md) for exact coverage.

See [Final local demo handoff](docs/FINAL_HANDOFF.md) for a one-command Windows verification checklist and honest project acceptance scope. See [backend/DAY20_DELIVERY.md](backend/DAY20_DELIVERY.md) for the earlier research-prototype milestone. GitHub Actions runs backend tests and frontend checks on new pull requests.
