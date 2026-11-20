# Experimental Baseline — Initial ASVspoof 2019 LA Run

> Historical user-recorded experiment notes. Original data, model artifacts and report JSON were unavailable for the 20 November audit; these results have not been independently reproduced. See [current verification evidence](../docs/VERIFICATION.md).
**Date:** November 15, 2026  
**Project:** AcousticSpace  
**Status:** Local baseline trained and verified through FastAPI and the frontend

## Experiment setup

- **Dataset:** ASVspoof 2019 Logical Access (LA), downloaded from the [official Edinburgh DataShare record](https://datashare.ed.ac.uk/items/31074a11-b6f6-4e92-a4ad-07093f8c0c45).
- **Manifest generation:** `scripts/import_asvspoof2019.py`, with `--max-per-speaker-class 10` and default seed.
- **Recordings:** 700 selected FLAC files across 40 declared speakers.
- **Splits:** 400 train (20 speaker groups), 140 validation (10 groups), 160 test (10 groups).
- **Class counts:** train 200 real/200 fake; validation 100 real/40 fake; test 100 real/60 fake.
- **Validation:** `scripts/check_training_data.py` and `scripts/dataset_readiness.py` both passed with no readiness warnings. These checks validate manifest identity, counts, and declared group separation, not real-world robustness.
- **Model:** Scikit-learn logistic regression with standard scaling, class balancing, and three features: RMS amplitude, zero-crossing rate, and spectral centroid.
- **Model artifacts:** Saved locally as `backend/models/baseline_model.joblib` with adjacent `baseline_model.json`. These local artifacts are not committed.

The official ASVspoof **evaluation** partition was not used. The test split here was formed from a subset of official development speakers; these numbers must not be described as official ASVspoof benchmark results.

## Measured results

| Metric | Validation (n=140) | Test (n=160) |
|---|---:|---:|
| Accuracy | 75.71% | 70.63% |
| Fake-class precision | 56.00% | 60.32% |
| Fake-class recall | 70.00% | 63.33% |
| Fake-class F1 | 62.22% | 61.79% |

### Confusion matrices

Rows represent actual classes (real, fake); columns represent predicted classes (real, fake).

**Validation:**

| Actual / Predicted | Real | Fake |
|---|---:|---:|
| Real | 78 | 22 |
| Fake | 12 | 28 |

**Test:**

| Actual / Predicted | Real | Fake |
|---|---:|---:|
| Real | 75 | 25 |
| Fake | 22 | 38 |

On the test split, the model missed 22 of 60 spoofed recordings and incorrectly flagged 25 of 100 genuine recordings.

## Integration verification

- FastAPI started successfully with Uvicorn.
- `GET /api/model/status` returned HTTP 200 with `available: true` and the recorded validation/test metrics.
- The frontend displayed the trained baseline and successfully returned a prediction for a manually uploaded recording.

## Limitations and next steps

This is a **three-feature audio baseline**, **not** the proposed RIR-based or Audio Spectrogram Transformer detector. Its scores are not calibrated probabilities, and the results do not establish performance on unseen real-world recording conditions or new synthesis methods. Future work should include independent evaluation, richer acoustic/reverberation features, and stronger models. Avoid using this model for consequential authenticity decisions.
