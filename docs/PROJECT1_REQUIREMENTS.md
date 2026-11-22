# AcousticSpace — Project 1 PDF acceptance checklist

Basis: user-provided `PROJECT(1).pdf`, Project 1 on pages 1–2. This document distinguishes implemented code, successful CI, local verification and experimental claims. It does not substitute for an independent research review.

| Original requirement | Implementation / evidence | Status |
| --- | --- | --- |
| FastAPI gateway and audio upload | `backend/app/main.py`, `audio_validation.py`, `audio_inspection.py` | Implemented; backend pytest in CI |
| React dashboard and waveform player | `frontend/src/components/AudioUpload.tsx`, `AudioPreview.tsx` | Implemented; Vite build/lint in CI |
| Librosa processing and log-mel spectrogram | `preprocessing.py`, `features.py`, `speech_diagnostics.py` | Implemented signal analysis, not source/RIR isolation |
| Extract environmental reverb or RIR *from speech* | Isolated RIR WAV analyzer `rir_analysis.py`; ordinary speech features remain confounded | **Not achieved** |
| Baseline CNN/Transformer | Three-/six-feature logistic regression baselines; locally reported one-epoch AST head-only experiment | Baselines evaluated; head-only AST result is not a validated CNN/fine-tuned Transformer |
| Mid-review voice/background separation | Not demonstrated; mixed mono speech does not identify a unique blind room response | **Not achieved** |
| Fine-tune Hugging Face AST | `scripts/train_ast.py`, optional dependencies, `/api/ast/status` and `/api/ast/predict`; local last-one-layer partial-encoder fine-tune completed on RTX 3050 (3 epochs) | **91.43% accuracy, 87.25% balanced accuracy on 140 validation files** (user-reported); independent new test and full-encoder fine-tune **not verified** |
| Check breathing cadence versus syllables | Exploratory energy peaks/noisy quiet intervals; `scripts/evaluate_annotations.py` measures their agreement with separately supplied event labels | Instrumented; **independent breath/syllable validation still missing** |
| Suspicious segments in results UI | `segment_analysis.py`, `SegmentReview.tsx`, overlapping classifier windows; optional annotated time-IoU evaluation | Functional exploratory scores; **independent tamper localization validation still missing** |
| Prediction display | `BaselineClassifier.tsx` + `AcousticComparison.tsx` | Implemented; values are uncalibrated classifier scores |
| Dashboard state/history | `analysisHistory.ts`, `HistoryPanel.tsx` | Session-only metadata; no backend audit log |
| Docker containerization | `backend/Dockerfile`, `frontend/Dockerfile`, `compose.yaml` | Built successfully in CI; live local Docker demo not yet verified |
| CI/CD | `.github/workflows/ci.yml` | Backend pytest + frontend build/lint + Compose build |
| Low-latency API | User-reported 20-sample Windows in-process route timing; repeatable sequential and concurrent ASGI benchmark scripts | **Measured locally for generated one-second inputs; network/production SLO not verified** |
| Final physics-based deepfake detection | Standalone RIR metrics + speech classification are separate systems | **Not verified** |

See [Independent annotation evaluation](ANNOTATED_VALIDATION.md) for a fail-closed local breath/syllable and tamper-window scoring workflow using real independently annotated WAV/MP3 files; no ground-truth recording labels have been supplied or fabricated.

## Research dataset provenance safeguards

The new `scripts/validate_rir_provenance.py` checks a **manually curated** physical-room and recording-session manifest for split leakage, duplicate WAV hashes, microphone/source-position diversity and format integrity. See [RIR provenance validation](RIR_PROVENANCE_VALIDATION.md) and the header-only CSV template.

This implements a structural guard **only**. The current 226-file candidate archive has **not** passed independent human physical-room verification, and no new held-out accuracy or real deepfake results are available.

For repeatable local processing-time instrumentation, see [API latency benchmark](LATENCY_BENCHMARK.md). Actual target-machine measurements are still required.

A later RTX 3050 partial-AST experiment updated the last encoder layer, achieved 91.43% validation accuracy (87.25% balanced accuracy) and retained its checkpoint as `backend/models/ast_partial_top1_v1`. A separate read-only evaluation successfully reloaded that checkpoint, reproduced its validation confusion matrix and reported ROC-AUC **0.9215** (500-sample stratified bootstrap 95% interval **0.8539–0.9728**). This is a **within-dataset validation result**, not an independent generalization claim. See `backend/AST_EVALUATION.md` for the recorded outcome and [external AST evaluation](AST_EXTERNAL_VALIDATION.md) for the frozen-checkpoint independent-corpus workflow.

## Experimental baseline results (user-supplied local run)

Dataset: ASVspoof 2019 LA subset; 400 train, 140 validation, 160 test. The original evaluation partition was not used.

| Test metric | 3-feature | 6-feature |
| --- | ---: | ---: |
| Accuracy | 70.63% | 68.75% |
| Fake recall | 63.33% | 76.67% |
| Fake F1 | 61.79% | 64.79% |

These are exploratory dataset-specific results, not independent forensic benchmark performance. Do not use scores as calibrated probabilities.

## Repeatable verification

Use `verify_local_demo.cmd` from the repository root for backend tests, in-process HTTP smoke checks, frontend build and lint; add `--require-models` to require trained baseline/candidate models and a comparison report. Read `docs/FINAL_HANDOFF.md` for precise pass criteria.

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
