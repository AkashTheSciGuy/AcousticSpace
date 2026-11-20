# AcousticSpace Frontend

The AcousticSpace frontend is a React, TypeScript, and Vite interface for examining audio recordings and displaying experimental audio-analysis results.

## Features

- Select WAV or MP3 recordings, inspect their waveform, and control playback.
- Submit recordings to the FastAPI backend for audio inspection.
- View room impulse response (RIR) analysis separately from ordinary recording inspection.
- Check whether the experimental baseline classifier is installed.
- Display validation and test metrics from the locally trained baseline model, including accuracy, fake-class precision, recall, F1, and labeled confusion matrices.
- Upload a supported recording for a baseline prediction when a locally trained model is available.

The classifier is a research baseline using RMS amplitude, zero-crossing rate, and spectral centroid. It does **not** perform RIR-based deepfake detection, and its class scores should not be treated as calibrated probabilities or forensic evidence.

## Run locally (Windows CMD)

Start the FastAPI backend in a separate terminal before opening the frontend:

```cmd
cd /d F:\Projects\AcousticSpace\backend
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Then run the frontend:

```cmd
cd /d F:\Projects\AcousticSpace\frontend
npm.cmd ci
npm.cmd run dev
```

Open `http://localhost:5173`.

To verify the frontend production build:

```cmd
npm.cmd run build
npm.cmd run lint
```

## Baseline integration checkpoint — November 15, 2026

The locally trained baseline was confirmed available through `GET /api/model/status` (HTTP 200), and the frontend's baseline prediction flow was manually verified with a recording. The local ASVspoof 2019 LA experiment used 400 training, 140 validation, and 160 test recordings. Its validation accuracy was 75.71%, and test accuracy was 70.63%. These are **dataset-specific experimental measurements**, not general real-world detection accuracy. Neither the dataset nor the trained model is shipped in this repository.

For dataset preparation and training, see `backend/DATASET_TRAINING.md` in the repository root.

## Current audit additions

Both the three-feature baseline and six-feature speech-acoustic candidate are supported. Successful predictions include artifact identity and server processing time, with the latest 20 results kept only in page memory. See [workflow audit](../docs/WORKFLOW_AUDIT.md), [API contract](../docs/API_CONTRACT.md), [architecture](../docs/ARCHITECTURE.md), and [verification evidence/browser test instructions](../docs/VERIFICATION.md). The November 15 checkpoint above is historical and was not independently reproduced by the current audit.
