# AcousticSpace — Final demo and submission handoff

**Delivery:** Internship Project 1 — local audio-forensics research prototype  
**Date:** 22 November 2026

The application is an educational research prototype. It does **not** establish forensic-grade deepfake detection, blind room response recovery from arbitrary speech, authentic breathing/syllable alignment, or validated tamper localization.

## Delivered software

| Capability | State | Boundary |
| --- | --- | --- |
| React upload and waveform | Implemented | WAV/MP3 up to 20 MiB |
| FastAPI validation/inspection | Implemented | Inspects up to 60 s; no authenticity verdict |
| Log-mel spectrogram/cadence | Implemented | Energy peaks are not syllable/breath labels |
| Isolated room response metrics | Implemented | Requires a **separate RIR WAV** (RT60, C50 and approximate DRR) |
| 3-feature baseline and 6-feature candidate | Implemented | Needs locally trained models; neither is RIR-based |
| Overlapping-window scores | Implemented | Not verified tamper localization |
| AST trainer and optional inference | Implemented workflow | Initial local head-only experiment gave poor validation accuracy |
| Controlled RIR consistency study | Reproducible scripts | Requires isolated wet stems and dry references |
| Docker Compose and CI | Implemented | Local demonstration; not production deployment |
| History | Implemented | Browser session only, not a persistent database |

## One-command Windows verification

After pulling the final merged main, run in **Windows CMD**:

~~~cmd
cd /d F:\Projects\AcousticSpace
verify_local_demo.cmd
~~~

This runs backend pytest, in-process FastAPI HTTP acceptance checks using generated *non-speech* WAV fixtures, the frontend production build, and frontend lint. It never overwrites datasets, trains models, or starts network listeners.

**If the demo will show trained model predictions**, use strict mode instead:

~~~cmd
verify_local_demo.cmd --require-models
~~~

Strict mode fails unless the trusted locally trained baseline, six-feature candidate and comparison report are all present and usable. Core mode passes even without optional model artifacts; it does not claim otherwise.

To record the model-backed smoke result under a unique filename:

~~~cmd
cd /d F:\Projects\AcousticSpace\backend
.venv\Scripts\python.exe scripts\final_acceptance.py --require-models --output models\final_acceptance_2026-11-22.json
~~~

The report writer refuses to overwrite an existing file. Optional AST model readiness (potentially slow model loading) can be checked separately:

~~~cmd
.venv\Scripts\python.exe scripts\final_acceptance.py --check-ast
~~~

No smoke test measures classifier accuracy or demonstrates forensic validity.

## Running the demonstration

Open **two separate Windows CMD windows**.

Backend terminal:

~~~cmd
cd /d F:\Projects\AcousticSpace\backend
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
~~~

Frontend terminal:

~~~cmd
cd /d F:\Projects\AcousticSpace\frontend
npm.cmd run dev
~~~

Open the dashboard at http://localhost:5173 and API docs at http://localhost:8000/docs. The frontend routes API requests to the FastAPI backend.

Demo sequence:

1. Choose an actual WAV/MP3 recording, play the waveform, click **Inspect recording**, show numerical signal features.
2. Click **Inspect spectrogram and cadence**, explain that detected energy peaks and quiet intervals are **observations**, not annotated syllables or breathing.
3. Upload the separately existing local **backend/synthetic_rir_demo.wav** to the **RIR analysis** panel, show estimated reverberation/clarity; say explicitly that the RIR was supplied independently and is synthetic, not extracted from the uploaded speech.
4. Refresh baseline status and run an experimental classifier only if its trusted local artifact is loaded. Scores are uncalibrated, not evidence of authenticity.
5. Optionally demonstrate six-feature candidate, overlapping window scores and AST status. A flagged window is not a verified tamper location; an unavailable AST must be reported unavailable.
6. Show session history; do not represent it as permanent storage.

For Docker use docs/DOCKER.md. The container CI demo does **not** verify user's GPU or bundle model weights.

## Actual reported experimental evidence

These are previous user-reported local evaluations, not new independently reproduced results.

| Experiment | Result | Meaning |
| --- | --- | --- |
| ASVspoof 2019 LA subset, 3-feature model | **70.63% test accuracy** | Exploratory, within-dataset baseline, 160 test clips; official evaluation partition not used |
| Same subset, 6-feature model | **68.75% test accuracy**, 76.67% fake recall | More fake recall but more false positives |
| Local head-only AST v1 (1 epoch) | **47.14% validation accuracy**, 66% real-class false-positive rate on 140 clips | Neither reliable AST accuracy nor full encoder fine-tuning established |
| Local partial-encoder AST top-one-layer (3 epochs) | **91.43% validation accuracy, 87.25% balanced accuracy**, 3/100 real false positives and 9/40 fake misses on 140 clips | User-reported validation; checkpoint saved locally, but independent external spoof accuracy and blind RIR performance not established |
| Reference-aided measured-RIR consistency experiment | **93.75% validation accuracy** and **50% held-out test accuracy**, 16 per split | Weak generalization; labels identify synthetic room assignment, **not genuine/fake audio** |

The original RIR test split was examined; do not tune models, features or thresholds against it. A larger independently sourced and pre-registered experiment would be needed for new accuracy claims. Physical room provenance and microphone acquisition protocols still require confirmation.

The local RIR archive contained 226 decodable PCM-16, 16-kHz candidate files. A quality audit identified 169 selected-channel recordings without heuristic flags. WAV validity and absence of signal-quality flags do not demonstrate independent physical rooms.

## Scope/acceptance decision

The **software engineering prototype** is demonstrable for the internship once the verification command passes and the dashboard/API operate as documented. Acceptance against the full original internship specification still requires real scientific evaluation evidence and mentor review. If classifier-backed predictions are part of the live demonstration, strict verification must pass on the actual machine.

The newer AST experiment successfully updated part of the encoder on the RTX 3050; it is **not full-encoder fine-tuning**. See `backend/AST_EVALUATION.md` for detailed training outcomes and the read-only validation command. The previously seen test split should not be reused for model selection.

The **original full research goal** remains scientifically unproven: it would require validated separation and independent acoustic-response estimation from a mixed speech recording, fully evaluated transformer training, independently labeled breathing/phonetics, real tamper-localization ground truth, and a genuinely new held-out room dataset.

A single mono speech mixture does not uniquely reveal dry voice, noise, and their separate transfer responses without additional physical constraints. No small code patch can create those missing measurements.

See docs/PROJECT1_REQUIREMENTS.md, docs/ANNOTATED_VALIDATION.md, docs/RIR_PHASE2C_REPRODUCIBILITY.md, backend/AST_EVALUATION.md and GitHub issue #41 for evidence and remaining research requirements.

## Submission checklist

Submit source, tests, README, this handoff and relevant research reports. Save local CI/acceptance output and genuine demo screenshots for the presentation. Do **not** commit downloaded audio archives, private recordings, local models, backend/models, backend/datasets, virtual environments or node_modules. Never represent generated fixture audio as evidence of real-world deepfake accuracy.

**Current internship delivery status:** a reproducible, operational, transparently limited audio-forensics research prototype. The specification's blind-RIR deepfake detection and independent breath/tamper/AST evaluations remain open; do not call the full research objective completed without evidence and mentor acceptance.
