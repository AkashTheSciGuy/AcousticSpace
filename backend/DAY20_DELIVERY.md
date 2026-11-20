# AcousticSpace — Day 20 local research prototype delivery

> Historical user-recorded experiment notes. Original data, model artifacts and report JSON were unavailable for the 20 November audit; these results have not been independently reproduced. See [current verification evidence](../docs/VERIFICATION.md).
## Verified experiment (ASVspoof 2019 LA)

The selected dataset manifest contains 700 recordings from 40 declared speaker groups:
400 training, 140 validation, 160 test. The original official evaluation partition
was **not** used. The importer verifies file identity and declared speaker-group
separation; it does not establish real-world source independence.

| Experimental metric on 160-recording test split | Three-feature baseline | Six-feature candidate |
| --- | ---: | ---: |
| Accuracy | 70.63% | 68.75% |
| Fake-class precision | 60.32% | 56.10% |
| Fake-class recall | 63.33% | 76.67% |
| Fake-class F1 | 61.79% | 64.79% |

These are user-recorded local experiment results, **not independently reproduced benchmark
scores**. No claim about out-of-dataset or forensic accuracy is warranted. The six-feature
candidate catches more fake recordings within the experiment but produces more false alarms.

## Local demo checklist (Windows CMD)

Run from a checkout containing both merged feature branches. Do not commit
`backend/models/`, any FLAC files or the dataset manifest.

```cmd
cd /d F:\Projects\AcousticSpace\backend
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\verify_release.py
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

In a second CMD window:

```cmd
cd /d F:\Projects\AcousticSpace\frontend
npm.cmd run build
npm.cmd run dev
```

Visit `http://localhost:5173`. Confirm audio upload and waveform, speech-acoustic
inspection, isolated RIR WAV analysis, both model status variants, baseline and
candidate predictions, and the local comparison panel.

## Scope limits

- The existing RIR analyzer accepts a **separate isolated impulse-response WAV**.
- Speech-acoustic descriptors are not a recovered room impulse response, RT60 or C50.
- The baseline and candidate are three/six-feature logistic-regression prototypes,
  not AST or independently validated RIR-based deepfake detectors.
- Probabilities are uncalibrated model scores, not proof of authenticity.
- Production deployment, independent cross-dataset evaluation and RIR inference
  from speech remain future research and engineering work.
