# Curated RIR provenance gate — research dataset integrity

The available 226 AIR/RVB2014/RWCP candidate WAV recordings have passed header and signal-sanity inspections. **They do not yet constitute an independently verified, split-safe room dataset.** Names such as \`smallroom1\`, \`stairway\` and \`cirline_e2a\` do not themselves prove physical room independence.

The new validator is deliberately strict and **requires a human-curated metadata manifest**. It never derives or fabricates physical room names from audio filenames. Do not claim new ML accuracy until room identities have been checked against original measurement documentation.

## 1. Prepare a curated manifest

Copy the header-only template at \`docs/rir_provenance_template.csv\` to a **new, untracked local CSV** in your prepared RIR root. Fill one row per mono RIR recording from the original dataset's official measurement metadata.

The required columns are:

| Column | Meaning |
| --- | --- |
| \`path\` | Relative POSIX WAV path below the dataset root |
| \`rir_id\` | Stable unique measurement identifier |
| \`site_id\` | Building/site identifier |
| \`physical_room_id\` | Physically verified room identity **within that site** |
| \`session_id\` | Original measurement session identifier |
| \`mic_device\` | Measurement microphone/device model or ID |
| \`mic_position\` | Microphone pose/position marker from source metadata |
| \`source_position\` | Sound-source pose/position from source metadata |
| \`provenance_reference\` | Official dataset record or documentation supporting the assignments |
| \`split\` | \`train\`, \`val\` or \`test\` |
| \`sha256\` | Full lowercase or uppercase SHA-256 of the **prepared mono WAV bytes** |

Do not use the basename, recording series or arbitrary dummy room IDs in place of physical provenance. Position identifiers must distinguish at least two microphone/source configurations per room. If source metadata does not contain enough information, the scientific gate is **not ready** and must remain open.

## 2. Validate the source-manifest structure

From Windows CMD, using a **new report filename**:

~~~cmd
cd /d F:\Projects\AcousticSpace\backend
.venv\Scripts\python.exe scripts\validate_rir_provenance.py datasets\prepared_rirs datasets\prepared_rirs\curated_rooms.csv --output datasets\prepared_rirs\provenance_audit_v1.json
~~~

The script reads, but never modifies, source WAVs; it refuses to overwrite an audit report.

Checks include:

- 16-kHz mono WAV, 256–32,000 sample frames and 4 MiB file bound.
- Every WAV SHA-256 matches the manually curated CSV.
- No duplicate input paths, RIR IDs, or identical WAV bytes.
- No self-declared physical room or recording session appears in two data splits.
- At least two distinct declared physical rooms in each train/validation/test split.
- At least two different microphone/source positions per room.
- No blank or obviously placeholder source documentation fields.

## 3. Limits of this validation

**\`metadata_checks_passed\` does not mean physical provenance has been independently verified.** The tool cannot determine whether the supplied room assignments, source identities, sites, sessions and measurement records are true. It reports \`physical_provenance_independently_verified: false\` in every successful output.

A reviewer must read the original source documentation, confirm recording setups and separation, verify compatible data licenses, and sign off the actual rooms before training. Passing this gate also cannot prove microphone independence, source independence, independent deepfake labels, or the accuracy of a blind-RIR detector.

Previously exposed test mixtures must not be used to adjust models or tune thresholds. A **new independent test protocol** is still required, including cases with similar RT60 and different actual source/room conditions.

## 4. Related tools

- \`scripts/inspect_rir_audio.py\` — original ZIP header inventory.
- \`scripts/audit_rir_signal_quality.py\` — bounded sample-level heuristics, not exclusion labels.
- \`scripts/prepare_measured_rirs.py\` — prepared candidate signals, filename-derived groups that **require review**.
- \`scripts/validate_rir_provenance.py\` — manually curated recording provenance structural gate (this document).
- \`scripts/build_rir_dataset.py\`, \`scripts/train_rir_detector.py\` — controlled experiments; not proof of deepfake detection.

Document the source record, acoustic environment, model selection protocol and independently held-out evaluation results before describing AcousticSpace as a validated research detector.
