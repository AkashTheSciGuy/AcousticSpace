# AcousticSpace — baseline training workflow

This is an **experimental audio-feature classifier**, not an RIR-based or AST-based deepfake detector. The dataset and independent evaluation are not supplied by this repository.

## 1. Prepare permitted, labeled audio recordings

Place recordings under a local dataset directory (do not commit audio or speaker metadata to GitHub without authorization):

```text
F:\Datasets\AcousticSpace\
  real\speaker_a\recording1.wav
  fake\speaker_a\recording2.wav
  real\speaker_b\recording3.wav
  fake\speaker_b\recording4.wav
  ...
```

Use WAV or MP3 files. A useful dataset needs **multiple independent speakers or sources**, not several near-identical clips from one speaker. Confirm the provenance and licenses of the recordings. A clip named `fake` must actually be AI-generated or otherwise correctly labeled.

## 2. Map all audio files to speaker/source groups

Create a CSV *outside the Git repository*, e.g. `F:\Datasets\AcousticSpace\groups.csv`:

```csv
path,group
real/speaker_a/recording1.wav,speaker_a
fake/speaker_a/recording2.wav,speaker_a
real/speaker_b/recording3.wav,speaker_b
fake/speaker_b/recording4.wav,speaker_b
```

Include **every** unique WAV/MP3 file in `real` and `fake`. Assign the **same group** to every recording from the same person/source across both real and fake labels. Do not use filenames, labels, or random IDs as a substitute for real provenance. At least three independent groups containing each class are required; more are strongly recommended.

## 3. Prepare group-independent splits (Windows CMD)

```cmd
cd /d F:\Projects\AcousticSpace\backend
.venv\Scripts\python.exe scripts\prepare_dataset.py F:\Datasets\AcousticSpace --groups F:\Datasets\AcousticSpace\groups.csv --output F:\Datasets\AcousticSpace\manifest.csv
```

The manifest includes a `group` column. Group identifiers are kept out of cross-split overlap. This addresses exact-match and declared-group overlap, but **cannot guarantee** independent sources when the supplied group IDs are inaccurate or when data contains near-duplicates.

Without `--groups`, the script retains the old clip-level splitting logic; do not report that as speaker-independent evaluation.

## 4. Train only after the dataset is ready

```cmd
cd /d F:\Projects\AcousticSpace\backend
.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
if not exist models mkdir models
.venv\Scripts\python.exe scripts\train_baseline.py F:\Datasets\AcousticSpace F:\Datasets\AcousticSpace\manifest.csv --output models\baseline_model.joblib
```

The trainer saves `backend/models/baseline_model.joblib` and `backend/models/baseline_model.json`. Do **not** run against an existing output path unless you intentionally move or archive the old results first; the script deliberately refuses overwriting.

Train/validation/test results are only **experiment results**. Review label quality, class balance, speaker/source independence, compression/device confounders, and errors before citing accuracy. Model files use `joblib` and must only be loaded from a trusted local training run.

## 5. Verify dashboard readiness

Start FastAPI and Vite as usual. Open `http://localhost:8000/api/model/status` or refresh the Baseline Model status in the dashboard. The model should remain unavailable until a compatible local artifact exists.
