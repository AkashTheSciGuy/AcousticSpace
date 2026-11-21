# Phase 2B — Reference-aided RIR consistency prototype

## Goal and limits

The original project proposes detecting when **voice and environment acoustics disagree**. We now have a controlled experiment that produces mixtures with labels:
- **consistent**: speech and ambient noise passed through two different impulse responses of the same declared room
- **mismatched**: the very same sources have speech in one room and ambient noise in a different room

Phase 2B adds a deliberately limited **reference-aided** consistency feature extractor, using four oracle recordings saved by the Phase 2A generator:

1. Known dry speech
2. Separately recorded wet speech stem
3. Known dry noise/ambience source
4. Separately recorded wet ambience stem

The system estimates transfer responses with regularized frequency-domain deconvolution and compares estimated decay, C50, approximate DRR, late energy and broadband spectral-shape descriptors. **Neither ground-truth RIR WAV, room ID nor filename enters the feature vector.** Room IDs are used only for label/leakage validation.

**Critical:** These references and isolated stems are not available when a user uploads an arbitrary WAV/MP3. This is not blind RIR extraction, direct deepfake detection, an automatically deployable prediction endpoint, or a calibrated forensic assessment.

## Windows CMD: run the first smoke experiment

If you have already created `datasets/rir_lab_v1`, leave it untouched.

```cmd
cd /d F:\Projects\AcousticSpace
git switch main
git pull --ff-only origin main
cd backend

.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\train_rir_detector.py datasets\rir_lab_v1 --output models\rir_consistency_toy_v1.joblib
```

This saves a trained **experimental logistic-regression consistency classifier** and metrics JSON beside it. It only fits on train and reports validation; it does **not** decode test WAV files by default. It does verify the full manifest split structure and mixture hashes. `backend/models/` is gitignored.

The smoke inputs have **12 generated mixtures**: four each in train, validation and test, with two of each class per split. This is nowhere near adequate for a meaningful supervised performance estimate. Any reported scores are just plumbing checks, and should not be described as deepfake accuracy.

For another run, choose a **new output filename**: commands will not overwrite trained model artifacts. Existing AST v1/v2, the baseline real/fake classifiers, and the dashboard remain unchanged.

## Next: collect scientifically useful paired data

- Real dry speech and environmental sources, with documented recording provenance
- Two or more measured room impulse responses *per room* (different source/microphone positions)
- Multiple distinct rooms assigned exclusively to train, validation or test, with independent speakers/source channels
- Matched and deliberately mismatched pairs at controlled SNRs and background types
- Negative controls that match reverberation time across *different* rooms, to prevent a trivial RT60-only shortcut
- Independent sessions, microphones and recording chains to assess real generalization

A classifier on isolated oracle stems is only an intermediate check. Recovering room characteristics from a mixed mono clip without dry references requires independently validated source-separation and blind-deconvolution assumptions. It must be assessed on genuine held-out recordings and never described as proven from these generated examples.

See also [RIR_EXPERIMENT.md](RIR_EXPERIMENT.md) and [AST_EVALUATION.md](AST_EVALUATION.md).
