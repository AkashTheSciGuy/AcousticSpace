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

## Start locally (Windows CMD)

**Backend**, in one terminal:

\`\`\`cmd
cd /d F:\Projects\AcousticSpace\backend
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -r requirements-ml.txt
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
\`\`\`

Open API docs at http://localhost:8000/docs .

**Frontend**, in a separate terminal:

\`\`\`cmd
cd /d F:\Projects\AcousticSpace\frontend
npm ci
npm run build
npm run dev
\`\`\`

Open http://localhost:5173 . Vite proxies \`/api\` to the backend on port 8000.

## Local model training and verification

See [backend/DATASET_TRAINING.md](backend/DATASET_TRAINING.md), [backend/DAY3_CANDIDATE.md](backend/DAY3_CANDIDATE.md) and [backend/DAY20_DELIVERY.md](backend/DAY20_DELIVERY.md).

Run from \`backend\` after training the models:

\`\`\`cmd
.venv\Scripts\python.exe scripts\verify_release.py
\`\`\`

The \`backend/models/\` directory contains generated artifacts and reports. **Do not commit datasets, manifest files or trained \`.joblib\` models.** Only load trusted, self-generated joblib models.

## Experimental evaluation

The recorded local ASVspoof 2019 LA subset contained **400 train / 140 validation / 160 test recordings**.

| Test-set measure | Three-feature baseline | Six-feature candidate |
| --- | ---: | ---: |
| Accuracy | 70.63% | 68.75% |
| Fake-class recall | 63.33% | 76.67% |
| Fake-class F1 | 61.79% | 64.79% |

This was a within-dataset exploratory comparison. The official evaluation partition was not used, and generalization to unknown recordings has not been verified. The candidate detects more spoofed samples in this test but also increases false positives.

## What is not implemented

The original project workflow proposed an AST/CNN model, PyTorch fine-tuning, breathing/cadence analysis, suspicious segment localization, analysis history, Docker and CI/CD deployment. **Those should not be presented as completed features.** This repository currently has an experimental speech-based classifier and a **separate** RIR analyzer; it does not reconstruct RIR from a speech clip.

See [backend/DAY20_DELIVERY.md](backend/DAY20_DELIVERY.md) for the final local demo checklist. GitHub Actions runs backend tests and frontend checks on new pull requests.
