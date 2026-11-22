# External AST generalization check — separate frozen-checkpoint protocol

The user's selected AST checkpoint achieved **91.43% accuracy** and **87.25% balanced accuracy** on 140 within-dataset ASVspoof 2019 LA validation clips, with ROC-AUC 0.9215, fake recall 77.5% and real false-positive rate 3.0%. Those numbers are *not* out-of-dataset results. This procedure measures a frozen checkpoint on independently obtained, labeled evaluation recordings, without retraining or choosing thresholds after seeing answers.

## 1. Obtain genuinely different labeled recordings

A documented candidate is the **official ASVspoof 2021 Deepfake (DF) evaluation corpus**, with official evaluation keys and metadata:

- Official challenge dataset and keys: https://www.asvspoof.org/index2021.html
- Original DF audio archives: https://zenodo.org/records/4835108
- Official evaluation-package documentation: https://github.com/asvspoof-challenge/2021/tree/main/eval-package

Check the data license and disk capacity before downloading. The original DF collection is large and is delivered in multiple parts. An independently preselected, provenance-auditable balanced subset can be used for a bounded test, but it is **not equivalent to the official complete challenge evaluation**.

**Important:** Physical independence is not guaranteed merely by using a dataset with a different year. Original speech source/speakers, transcoding and generator/attack types may overlap across corpora. The challenge's own file-mapping and metadata releases should be reviewed. If source identities cannot be established, report the remaining uncertainty.

Do not cherry-pick easy recordings, tune the model using DF keys, or repeatedly evaluate many checkpoints against these labels. Freeze the already selected checkpoint and **0.5 threshold** before running the external test.

## 2. Construct a small, auditable evaluation manifest

Copy docs/ast_external_manifest_template.csv into an untracked local directory. Add one row for every selected audio recording, with the following columns:

| CSV column | Required value |
| --- | --- |
| path | Relative file path under a distinct external data root; .wav, .mp3 or .flac |
| label | Exactly real or fake, sourced from official trial keys |
| sha256 | Hex SHA-256 of the exact local audio file's bytes |
| group | Nonempty source group from official speaker/source metadata; no fabricated IDs |
| source_reference | An actual document/metadata reference supporting the label and group |

The utility accepts at most **1,000 files**, each no larger than **20 MiB**, and requires both real and fake labels and at least two distinct declared source groups. This is a deliberately bounded experiment, not official full-corpus scoring. Generate SHA-256 values with PowerShell:

~~~powershell
(Get-FileHash "F:\Datasets\IndependentSpoofCorpus\path\sample.flac" -Algorithm SHA256).Hash.ToLowerInvariant()
~~~

Never invent class labels or speaker/source groups. If the metadata lacks a defensible group identity, do not claim speaker independence; obtain authoritative metadata first.

## 3. Evaluate the saved AST once

After the real external files and manifest are present, from backend in Windows CMD, substitute the actual external paths below:

~~~cmd
cd /d F:\Projects\AcousticSpace\backend
.venv\Scripts\python.exe scripts\evaluate_ast_external.py F:\Datasets\IndependentSpoofCorpus F:\Datasets\IndependentSpoofCorpus\eval_manifest.csv --training-manifest F:\Datasets\ASVspoof2019\manifest.csv --model-dir models\ast_partial_top1_v1 --bootstrap-samples 500 --seed 42 --output models\ast_external_v1.json
~~~

The utility checks external audio hashes, duplicate local files, safe paths, source metadata and *exact byte hash overlap* against the original ASVspoof 2019 manifest. Exact overlap causes a hard failure. It loads the **saved** AST checkpoint and scores only the first 10 seconds per recording, matching training and demo inference.

It returns fixed-threshold classification metrics, real false-positive rate, balanced accuracy, ROC-AUC, fake precision/recall/F1, confusion matrix and stratified bootstrap uncertainty. It records both manifest hashes and explicitly warns that exact-audio hashes do not catch re-encoded utterances or shared speakers.

The script **never modifies the checkpoint**, never selects a threshold, never writes individual raw audio/identity to Git, and refuses to overwrite an existing report. The report itself is saved locally under backend/models.

## 4. Interpret results without overstating scientific validity

A decline from 91.43% validation accuracy would be meaningful evidence of dataset shift, not a software failure. Do not revise the model or choose a new checkpoint after inspecting the same external test labels and continue calling it independent.

Even credible external AST results establish only **speech spoofing classification**. They do not prove RIR extraction from a mono mixture, room/acoustic mismatch detection, or independently annotated breathing/tamper location accuracy. Those are separate internship requirements.

The separate experimental RIR held-out room study remains at the reported 50% accuracy, and the annotation evaluator in docs/ANNOTATED_VALIDATION.md still needs real source-labeled recordings.

No external dataset or metric is included in GitHub. The operator must provide lawful access to genuine audio files and official labels before this experiment can produce a score.
