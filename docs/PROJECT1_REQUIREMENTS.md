# AcousticSpace — Project 1 PDF acceptance checklist

Basis: user-provided `PROJECT(1).pdf`, Project 1 on pages 1–2. This document distinguishes implemented code, successful CI, local verification and experimental claims. It does not substitute for an independent research review.

| Original requirement | Implementation / evidence | Status |
| --- | --- | --- |
| FastAPI gateway and audio upload | `backend/app/main.py`, `audio_validation.py`, `audio_inspection.py` | Implemented; backend pytest in CI |
| React dashboard and waveform player | `frontend/src/components/AudioUpload.tsx`, `AudioPreview.tsx` | Implemented; Vite build/lint in CI |
| Librosa processing and log-mel spectrogram | `preprocessing.py`, `features.py`, `speech_diagnostics.py` | Implemented signal analysis, not source/RIR isolation |
| Extract environmental reverb or RIR *from speech* | Isolated RIR WAV analyzer `rir_analysis.py`; ordinary speech features remain confounded | **Not achieved** |
| Baseline CNN/Transformer | Three-/six-feature logistic regression baselines; optional AST training workflow | Baselines evaluated; no trained CNN/AST |
| Mid-review voice/background separation | Not demonstrated; mixed mono speech does not identify a unique blind room response | **Not achieved** |
| Fine-tune Hugging Face AST | `scripts/train_ast.py`, optional dependencies, `/api/ast/status` and `/api/ast/predict` | Script/API implemented; **no model training / evaluation verified** |
| Check breathing cadence versus syllables | Energy peaks and noisy quiet intervals `speech_diagnostics.py` | Exploration only; **not validated breath/syllable alignment** |
| Suspicious segments in results UI | `segment_analysis.py`, `SegmentReview.tsx`, overlapping classifier windows | Functional exploratory scores; **not validated tamper localization** |
| Prediction display | `BaselineClassifier.tsx` + `AcousticComparison.tsx` | Implemented; values are uncalibrated classifier scores |
| Dashboard state/history | `analysisHistory.ts`, `HistoryPanel.tsx` | Session-only metadata; no backend audit log |
| Docker containerization | `backend/Dockerfile`, `frontend/Dockerfile`, `compose.yaml` | Built successfully in CI; live local Docker demo not yet verified |
| CI/CD | `.github/workflows/ci.yml` | Backend pytest + frontend build/lint + Compose build |
| Low-latency API | Upload bounds and batched window scores; no comprehensive latency benchmark | **Not benchmarked** |
| Final physics-based deepfake detection | Standalone RIR metrics + speech classification are separate systems | **Not verified** |

## Experimental baseline results (user-supplied local run)

Dataset: ASVspoof 2019 LA subset; 400 train, 140 validation, 160 test. The original evaluation partition was not used.

| Test metric | 3-feature | 6-feature |
| --- | ---: | ---: |
| Accuracy | 70.63% | 68.75% |
| Fake recall | 63.33% | 76.67% |
| Fake F1 | 61.79% | 64.79% |

These are exploratory dataset-specific results, not independent forensic benchmark performance. Do not use scores as calibrated probabilities.

## Repeatable verification

Backend from `backend/`:

```cmd
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\verify_release.py
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Frontend from `frontend/`:

```cmd
npm.cmd ci
npm.cmd run build
npm.cmd run lint
npm.cmd run dev
```

Optional AST status: `GET /api/ast/status`; training instructions: `backend/AST_TRAINING.md`.

Docker demo: see `docs/DOCKER.md`. CI tests source and builds images but does not prove public deployment or hardware inference.

## Scientific blocker for original goal

A single observed mono speech waveform is a convolution of voice, room, device response and noise. Without extra physical constraints (paired source/environment recordings, multichannel signals, known excitation, synthetic ground truth, or validated blind deconvolution) the original isolated room impulse response generally cannot be uniquely identified. The existing ASVspoof real/spoof labels alone do not provide known room impulse-response targets.

**Consequently, this work is a research prototype, not the specified validated physics-based forensic detector.** A credible final research claim needs controlled RIR/mismatch datasets, separate evaluation partitions, independently checked breath/phonetic annotations and measured latency.
