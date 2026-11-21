# Phase 2C — controlled experiment with measured room responses

**Result: the pipeline ran, but the RIR classifier did not generalize on its held-out test set.**

This document preserves the user's local experimental observations (November 21, 2026) and the exact methodological limitations. Source audio, trained models, and dataset manifests are intentionally not committed to Git.

## Experiment design

- **Speech:** 24 anechoic ARU speech clips from 12 speakers, originally 65,536 Hz mono and resampled to 16 kHz. Two 2-second clips per speaker.
- **RIRs:** 24 measured RVB2014 type-1 candidates from six named rooms. The original WAVs have 8 channels, 16 kHz sample rate, and 1 second duration. Channel 1 was selected individually, **not mixed across microphones**.
- **Conditions:** Controlled synthesis convolving genuine dry speech and seeded *artificial* noise with room responses. Same-room versus different-room *assignment*, not genuine versus fake speech.
- **References:** The classifier uses known dry source audio and isolated wet speech/noise stems. These are **oracle controls** unavailable for typical user-uploaded audio.
- **Splitting:** Speaker and named-room IDs do not overlap across train/validation/test. This is a manifest-level check; independent physical-room provenance needs source documentation.
- **Classifier:** Seven RIR-consistency descriptors, StandardScaler + balanced logistic regression, fixed 0.5 threshold. No probability calibration.

| Split | Speakers | Assigned room labels | Mixtures |
| --- | --- | --- | ---: |
| Train | ID01, ID04, ID07, ID10 | mediumroom1, smallroom1 | 16 |
| Validation | ID02, ID05, ID08, ID11 | largeroom1, smallroom2 | 16 |
| Test | ID03, ID06, ID09, ID12 | largeroom2, mediumroom2 | 16 |

Samples are paired and correlated: two label outcomes share dry speech/noise and wet speech, and recordings reuse speakers/rooms. Treating all 48 as independent is incorrect.

## Observed held-out results

| Metric | Validation (n=16) | Test (n=16) |
| --- | ---: | ---: |
| Accuracy | 93.75% | **50.00%** |
| Mismatched-class F1 | 0.9412 | 0.5556 |
| ROC-AUC | 1.0000 | 0.5625 |
| False-mismatch rate | 12.50% | 62.50% |
| Confusion matrix (consistent, mismatch rows) | [[7,1],[0,8]] | [[3,5],[3,5]] |

The test result is **chance-level accuracy** for the balanced set. Do not describe it as successful deepfake detection.

Train/validation-only ablations were also run:

- All seven descriptors: 93.75% validation accuracy.
- RT60 difference + C50 difference + late-energy difference: 93.75%.
- RT60 difference + C50 difference: 87.5%.

Ablation may guide follow-up research but **does not repair** the known test failure. The old test set is now exposed and must not be used to choose features or thresholds.

## Reproducible Windows CMD workflow

From \`F:\Projects\AcousticSpace\backend\`, after downloading the publicly available source datasets into the local folders described below, use **new output names** (the scripts refuse to overwrite):

\`\`\`cmd
.venv\Scripts\python.exe -m pytest -q

.venv\Scripts\python.exe scripts\audit_rir_archive.py datasets\rir_sources\rirs_noises.zip --output datasets\rir_sources\rir_archive_audit.json

.venv\Scripts\python.exe scripts\prepare_measured_rirs.py datasets\rir_sources\measured_rirs --output datasets\rir_sources\prepared_rirs_v2 --channel 1

.venv\Scripts\python.exe scripts\prepare_aru_rir_inputs.py datasets\speech_sources\ARU_Speech_Corpus_v1_0.zip datasets\rir_sources\prepared_rirs_v2 --output datasets\rir_sources\aru_measured_inputs_v2 --seconds 2

.venv\Scripts\python.exe scripts\build_rir_dataset.py datasets\rir_sources\aru_measured_inputs_v2 datasets\rir_sources\aru_measured_inputs_v2\speech.csv datasets\rir_sources\aru_measured_inputs_v2\responses.csv --output datasets\rir_lab_measured_v2 --seconds 2 --pairs-per-speech 1 --snr-db 10

.venv\Scripts\python.exe scripts\train_rir_detector.py datasets\rir_lab_measured_v2 --output models\rir_consistency_measured_v2.joblib

.venv\Scripts\python.exe scripts\diagnose_rir_ablation.py datasets\rir_lab_measured_v2
\`\`\`

The \`v2\` example reproduces the **same six-room experiment**, not an independent new benchmark. Do not repeatedly evaluate the old test split to tune candidates. The original report is in the user's local \`models\rir_consistency_measured_v1_final.json\`.

Preparation records source SHA-256s, original ARU archive member names, sample rate, microphone channel, and selected start frames in \`preparation_metadata.json\`. Preserve the raw archives and source dataset citation/licensing metadata.

## More measured RIR candidates

The local SLR28 archive listing contained **24 RVB2014, 118 RWCP, and 84 AIR candidate RIR files** (226 files total). These do *not* represent 226 independent physical rooms. The archive audit groups names by *recording series only*; it never creates model splits from guessed room identities. Independent room and recording-chain provenance must be established before augmenting the benchmark.

## Research milestones that remain open

1. Obtain documented independent room/session/microphone provenance and create larger group-disjoint train and validation partitions.
2. Design an entirely **new independent holdout**; include difficult matched-RT60 room negatives, microphone changes, independent noise sources, and real room recordings.
3. Evaluate a frozen protocol once on that new set. Preserve uncertainty estimates, failures and confusion matrices.
4. Investigate source separation/blind response estimation on arbitrary speech uploads; the required oracle stems and dry sources are absent, so no physics-based RIR deepfake inference endpoint is justified yet.
5. Validate a separately trained AST spoof detector, breathing/phoneme alignment, suspicious segment localization, latency and deployment claims with independent evidence.

**Conclusion:** Phase 2C is a reproducible research experiment, not a completed or independently validated physics-based deepfake detector.

## Signal-quality gate before selecting additional RIRs

The Phase 2C follow-up adds a **read-only single-microphone-channel signal audit**, because a valid 16-kHz PCM WAV header is not the same as an acoustically usable impulse response.

From `backend` in Windows CMD:

```cmd
.venv\Scripts\python.exe scripts\audit_rir_signal_quality.py datasets\rir_sources\rirs_noises.zip --output datasets\rir_sources\rir_signal_quality_v1.json --channel 1
```

The audit computes each selected microphone channel's peak, RMS, peak position, energy in the final 10% of the recording, and a hash. It flags silence, near-full-scale runs, short recordings, high tail energy, late peaks, non-16-kHz input and recordings outside the existing dataset generator's 0.016–2-second RIR frame bounds. No audio is extracted or transformed, and no model/training split is changed.

**Caution:** These flags are heuristics, not scientifically validated accept/reject labels. Responses whose tails remain energetic need inspection; longer recordings cannot simply be truncated without measuring the acoustic consequences. AIR binaural channels, RWCP array channels and RVB2014 microphone channels are not interchangeable or independent rooms. Preserve a consistent channel-selection and physical-room provenance protocol.

Source documentation: [OpenSLR SLR28](https://www.openslr.org/28), [Aachen AIR documentation](https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/), and [OpenSLR RWCP Sound Scene Database](https://www.openslr.org/13). The three source collections have different acquisition protocols; mixing their recording-series IDs without source metadata can create invalid labels or shortcut signals. AIR's `stairway` recordings and RWCP's `circle_ane` recordings should not be interpreted as separate physical rooms based on their counts.

**Next gate:** review the audit, independently verify the real room identities, choose multiple response measurements from each confirmed room, and allocate room-disjoint splits *before* building any expanded training benchmark. No old held-out test results should guide feature selection.
