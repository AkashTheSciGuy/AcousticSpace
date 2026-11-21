# Controlled RIR consistency experiment — Phase 2A

**Status:** Dataset generator and integrity checks are implemented. This does **not** recover RIR from ordinary speech, demonstrate neural deepfake detection, or validate an audio-forensics model.

## Scientific question

Can a system detect **acoustic-environment consistency** when the impulse responses of the speech and background source are controlled and the room labels are known?

Each **paired** control sample uses identical dry speech and an identically seeded dry noise source:

- "consistent": speech convolved with RIR A; ambience convolved with a **different RIR measured/simulated in the same room A**.
- "mismatched": the same speech convolved with RIR A; the same dry noise convolved with RIR B from a **different room**.

Within each pair, the level of wet speech and wet ambience is normalized to the same target SNR, and both outcomes share a clipping-prevention gain. Distinct same-room responses avoid a trivial "RIR files are identical" shortcut. This only tests **assigned-room consistency under synthesis assumptions**, not authenticity of speech or deepfake tampering.

**Limitations:** Different sources/positions inside the same room have differing RIRs. Different rooms can have similar reverberation. A mono mixture cannot uniquely reveal room responses without additional source/reference assumptions. Our isolated stems, dry references, and original RIRs are oracle controls, not available to a real forensic classifier. Do not train a classifier on source RIR filenames/room IDs, or claim generalization from the toy inputs.

## Run a deterministic smoke test on Windows CMD

From `F:\Projects\AcousticSpace\backend`:

```cmd
.venv\Scripts\python.exe scripts\generate_rir_demo_inputs.py --output datasets\rir_demo_inputs

.venv\Scripts\python.exe scripts\build_rir_dataset.py datasets\rir_demo_inputs datasets\rir_demo_inputs\speech.csv datasets\rir_demo_inputs\responses.csv --output datasets\rir_lab_v1 --seconds 1 --pairs-per-speech 1 --snr-db 10
```

The first script writes **synthetic speech-like harmonics** and artificial RIRs: six "speakers" and twelve responses across room/speaker-disjoint train, val, and test splits. They are NOT recorded speech or measured rooms.

The second script writes `datasets/rir_lab_v1/manifest.csv`, `metadata.json`, an `rirs/` folder, and WAV files under `mixtures/`, `stems/`, `references/`. Labels are `consistent` and `mismatched` (NOT "real" and "fake"). Every mixture includes its SHA-256 and reference-RIR hashes in the manifest; inputs are bounded and explicit.

Both scripts **refuse to overwrite existing output directories**. Choose new names for repeated runs; do not delete original assets.

The generated lab has 12 mixtures (4 per split). It is only a functional proof that paired synthesis, label bookkeeping, and isolation constraints work.

## Real controlled recordings

Prepare `speech.csv` with columns:

```csv
path,speaker_id,split
speech/sp1.wav,sp1,train
speech/sp2.wav,sp2,val
speech/sp3.wav,sp3,test
```

Prepare `responses.csv` with columns:

```csv
path,rir_id,room_id,split
rirs/room_a_pos1.wav,a1,room_a,train
rirs/room_a_pos2.wav,a2,room_a,train
rirs/room_b_pos1.wav,b1,room_b,train
rirs/room_b_pos2.wav,b2,room_b,train
```

The excerpts above illustrate the schema only; **each split needs at least one speech source and two rooms with two distinct RIR WAV recordings each**. Speaker and room IDs must never cross splits. Inputs must be genuine isolated dry-source recordings and measured or purposefully simulated RIRs, not ordinary ASVspoof clips relabeled as dry speech. WAV/FLAC paths are relative to the chosen input root. Group assignments must describe actual source provenance; filename checks alone cannot prove it.

Build your own version:

```cmd
.venv\Scripts\python.exe scripts\build_rir_dataset.py "F:\MyRirExperiment" "F:\MyRirExperiment\speech.csv" "F:\MyRirExperiment\responses.csv" --output datasets\rir_lab_measured_v1 --seconds 3 --pairs-per-speech 1 --snr-db 10
```

## Next research milestone

Implement and evaluate **signal-derived** acoustic consistency features from controlled isolated stems, with room/speaker disjoint groups and non-degenerate negative controls. The source RIRs should be ground truth for evaluation, **not direct classifier inputs**. Keep the AST v2 checkpoint and evaluation reports unchanged until there is independent evidence that adding an acoustic-consistency signal is beneficial.
