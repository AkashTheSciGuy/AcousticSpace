# Independent breath, syllable and tamper annotation evaluation

The internship specification requires validation of breathing/syllable timing and suspicious-segment localization. Existing dashboard outputs are exploratory observations; they have not been demonstrated as accurate breathing events, phonetic alignment, or forensic tamper locations.

The read-only offline evaluator in backend/scripts/evaluate_annotations.py measures those proxies on **independently annotated local recordings**. It does not invent labels, recover blind room impulse responses, or train a model.

## Required annotation manifest

Copy docs/annotated_events_template.json to a new local JSON under your ignored dataset directory. Its recordings list is intentionally empty.

Each recording object requires:

- recording_id: unique recording identifier
- path: safe relative WAV or MP3 path under the dataset root
- sha256: actual SHA-256 of the original audio bytes
- speaker_id, session_id: source-provided identifiers
- annotation_source: real provenance for the human-made timestamp labels
- annotations.breaths: pairs of start/end times in seconds
- annotations.syllables: pairs of start/end times
- annotations.tamper: pairs of genuinely manipulated start/end times, or an empty list only when independently confirmed no manipulation exists

The root JSON has schema_version equal to 1, partition equal to val or test, and a recordings list containing 1–200 entries. All required task annotations must be present, even when empty; other annotation categories can be omitted for a task.

Each interval is a two-number [start_seconds, end_seconds] pair, strictly positive in length, sorted and nonoverlapping, bounded to the recording's actual duration (up to 60 s).

Find actual WAV checksums in Windows PowerShell using:

~~~powershell
(Get-FileHash "F:\Datasets\YourCorpus\clips\your_audio.wav" -Algorithm SHA256).Hash.ToLowerInvariant()
~~~

## Breathing and syllable timing proxies

From backend in Windows CMD, after creating a real independent annotated corpus:

~~~cmd
cd /d F:\Projects\AcousticSpace\backend
.venv\Scripts\python.exe scripts\evaluate_annotations.py F:\Datasets\YourCorpus F:\Datasets\YourCorpus\events.json --task cadence --tolerance-seconds 0.2 --output models\cadence_annotated_v1.json
~~~

The evaluator compares **candidate quiet/noisy interval midpoints** with manually annotated breathing intervals and **energy-peak times** with manually annotated syllable intervals. It uses fixed 0.2 s one-to-one matching and records matched, missed and false-positive events plus precision, recall and F1.

Matches only measure heuristic timing overlap; they are not proof of physiological inhalation or actual phonetic syllable detection. The tolerance and recording selection must be fixed before evaluating an unexamined test partition.

## Suspicious-segment localization proxies

With independently labeled real manipulation boundaries and a trusted locally trained baseline checkpoint:

~~~cmd
.venv\Scripts\python.exe scripts\evaluate_annotations.py F:\Datasets\YourCorpus F:\Datasets\YourCorpus\events.json --task tamper --model baseline --output models\tamper_annotated_v1.json
~~~

This uses the existing overlapping 3-second window classifier at the **fixed deployed threshold 0.5**, combines overlapping flagged windows, and measures temporal precision, recall, intersection-over-union and false-positive/missed seconds. False positives in no-tamper files are counted, not ignored.

Use --model acoustic with a different output name to measure the six-feature candidate separately. Do not choose the "best" model after repeated inspection of an exposed test set.

## Interpretation, safety and integrity

- These metrics require external ground-truth event labels. An empty, mislabeled or unreviewed corpus is not scientific evidence. No annotation identities are fabricated.
- The tool checks path safety, bounded file sizes, SHA-256 matches, recording IDs, metadata and timestamp validity but cannot certify that manual annotations or physical-room identities are truthful.
- Null precision/recall values mean the denominator was zero; they **do not** mean 100% accuracy.
- Generated audio and synthetic labels used by software unit tests are not independent test data and must not appear in the internship's accuracy claims.
- A "test" partition named by the user is not proven untouched or source/speaker independent.
- Do not tune these algorithms against already examined test recordings.
- This evaluator does not prove blind RIR extraction, a reliable deepfake verdict or forensic detection.

A scientifically complete submission still needs independently verified AST and room-response results plus annotated breath/tamper recordings; code alone cannot establish those empirical findings.
