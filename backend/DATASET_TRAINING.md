# AcousticSpace — ASVspoof 2019 LA baseline training

This workflow trains the **experimental three-feature logistic-regression baseline** (RMS, zero-crossing rate, spectral centroid). It is **not** the proposed RIR/AST deepfake detector.

## 1. Obtain the original dataset

Download **ASVspoof 2019 Logical Access (LA)** from the official Edinburgh DataShare record:

https://datashare.ed.ac.uk/items/31074a11-b6f6-4e92-a4ad-07093f8c0c45

The `LA.zip` archive is approximately 7.12 GB. Review the upstream license before use. Extract it on a drive with sufficient free space (for example `F:\Datasets\ASVspoof2019\LA`). Do not commit recordings or model artifacts to GitHub.

The extracted `LA` directory should include:

```text
LA/
  ASVspoof2019_LA_train/flac/
  ASVspoof2019_LA_dev/flac/
  ASVspoof2019_LA_cm_protocols/
    ASVspoof2019.LA.cm.train.trn.txt
    ASVspoof2019.LA.cm.dev.trl.txt
```

The importer also accepts `ASVspoof2019_LA_protocols/` as a protocol-directory name.

## 2. Import a bounded, speaker-disjoint dataset manifest

Run from **Windows CMD**:

```cmd
cd /d F:\Projects\AcousticSpace\backend
.venv\Scripts\python.exe scripts\import_asvspoof2019.py "F:\Datasets\ASVspoof2019\LA" --output "F:\Datasets\ASVspoof2019\manifest.csv" --max-per-speaker-class 10
```

The importer reads official protocol labels and speaker IDs, samples at most 10 recordings of each class per speaker, hashes the selected FLAC recordings, and generates the same CSV schema used by `train_baseline.py`. Official **training** recordings remain in `train`. Official **development** speakers are assigned entirely to `val` or `test`; no speaker crosses splits. The official **evaluation** partition is left untouched.

The importer never edits the audio or overwrites a manifest. For quicker local debugging, reduce the sampling limit to `3`. For more extensive research, increase it.

## 3. Train the experimental baseline

```cmd
cd /d F:\Projects\AcousticSpace\backend
.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
if not exist models mkdir models
.venv\Scripts\python.exe scripts\train_baseline.py "F:\Datasets\ASVspoof2019\LA" "F:\Datasets\ASVspoof2019\manifest.csv" --output models\baseline_model.joblib
```

The script writes `models\baseline_model.joblib` and adjacent `models\baseline_model.json` metrics. The API already looks for this model path by default. Only load a **model artifact you created yourself**; joblib/pickle deserialization can execute code.

## 4. Verify and view results

Start FastAPI:

```cmd
cd /d F:\Projects\AcousticSpace\backend
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open `http://localhost:8000/api/model/status`. If the model trained successfully, `available` should be `true`. In the frontend, click **Refresh status**, then run an experimental prediction on a WAV/MP3 recording. The dashboard should show validation/test metrics from the local report.

**Research limitations:** This is a quick, small audio-feature baseline, **not an RIR-based classifier**. Development-data reuse can bias later tuning; reserve the official evaluation partition for a separate final evaluation. Scores are neither calibrated confidence nor forensic proof. Never claim benchmark accuracy without reporting the actual independent evaluation protocol and dataset.
