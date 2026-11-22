# Reference-aided RIR comparison — Jerusha backend integration

This module exposes the existing research feature extractor through an optional, bounded FastAPI endpoint and offline CLI. It does **not** estimate independent dry sources from a single mixed recording. It does **not** perform trained real/fake classification. The previously reported room-disjoint RIR experiment had 50% held-out test accuracy, so no forensic mismatch verdict is returned.

## Inputs

The four **separately supplied** recordings must be mono, 16-kHz PCM-compatible WAV files:

| Form field | Meaning |
| --- | --- |
| \`dry_speech\` | Original dry speech before reverberation |
| \`wet_speech\` | Isolated reverberant speech stem (not the total mix) |
| \`dry_ambience\` | Original dry environmental sound |
| \`wet_ambience\` | Isolated reverberant ambient stem (not the total mix) |

Each file is capped at 4 MiB, at most 12 seconds, with at least 256 samples. No file is stored by the API.

A valid controlled experiment must pair the dry sources and wet stems properly; supplying mismatched or unknown references produces uninterpretable distances.

## HTTP API

The API is registered at:

\`POST /api/rir/reference-consistency\`

In API docs at http://localhost:8000/docs, expand the **reference-aided-rir** section. Attach all four WAVs in the multipart form. The response provides seven scalar comparisons: RT60 difference, RT60 availability flag, C50 and DRR differences, late-energy fraction difference, decay-profile distance, and spectral-shape distance. These are descriptive quantities, **not** calibrated probabilities of mismatch, and \`prediction\` is always \`null\`.

## Offline CLI

Run from \`backend\` in Windows CMD with your actual experiment files:

~~~cmd
.venv\Scripts\python.exe scripts\compare_reference_rir.py --dry-speech datasets\your_dry_speech.wav --wet-speech datasets\your_wet_speech_stem.wav --dry-ambience datasets\your_dry_ambience.wav --wet-ambience datasets\your_wet_ambience_stem.wav --output models\reference_comparison_new.json
~~~

This command refuses to overwrite an existing report. The output JSON contains audio hashes for traceability, feature differences, and a conspicuous limitation statement.

Do not feed individual raw mixed recordings into all four inputs, and do not create labels based on file names.

## Verification

~~~cmd
cd /d F:\Projects\AcousticSpace\backend
.venv\Scripts\python.exe -m pytest -q tests\test_reference_rir_api.py tests\test_compare_reference_rir.py
.venv\Scripts\python.exe scripts\final_acceptance.py --require-models
~~~

The acceptance smoke test exercises this endpoint using generated controlled WAV signals. It does not establish real-world accuracy.

The original design's **blind RIR recovery from ordinary mono speech**, **breath vs syllable alignment**, **source separation** and **independent model validation** remain separate, unachieved research requirements. No endpoint in this PR should be represented as solving those requirements.
