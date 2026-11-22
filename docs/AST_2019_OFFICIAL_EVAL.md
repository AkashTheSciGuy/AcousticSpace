# AST — Official 2019 evaluation-partition experiment (fixed subset)

Operator confirmed that both ASVspoof 2019 LA official evaluation FLAC audio and its protocol with labels are installed under F:\Datasets\ASVspoof2019\LA.

**This is a new partition within the same ASVspoof 2019 dataset, NOT a different-dataset test.** The earlier 160-file development-derived test partition has already been examined; do not reuse that subset for model selection. Neither partition establishes physics-based RIR detection.

Lock choices before scoring:
- Already trained and selected checkpoint: models\ast_partial_top1_v1
- Decision threshold: **0.5**
- Fixed evaluation subset: **100 genuine and 100 spoof**, predetermined seed **42**
- Selection spreads actual protocol speakers and spoof attacks by round-robin without observing predictions
- No training, threshold tuning, new model comparison, or changing sample composition after observing results

## 1. Build the manifest from the existing official protocol

Windows CMD, from backend:

~~~cmd
cd /d F:\Projects\AcousticSpace\backend
.venv\Scripts\python.exe scripts\prepare_asvspoof2019_eval.py "F:\Datasets\ASVspoof2019\LA" --per-class 100 --seed 42 --output "F:\Datasets\ASVspoof2019\eval_fixed_100_each_seed42.csv"
~~~

The importer parses the official five-field protocol and refuses to invent labels. It checks label/attack consistency, selected audio existence, SHA-256 fingerprints, duplicate audio content and basic speaker diversity, then produces:

path,label,sha256,group,source_reference

Source group derives from the documented speaker field; the source reference includes official protocol line and attack marker. The printed report includes protocol and manifest hashes, label counts, attack distribution, and group count.

**Preserve the generated CSV and first-run logs.** The importer intentionally refuses to overwrite an existing manifest. A different seed or sample count chosen after examining results would invalidate the fixed-sample evaluation protocol.

This is a 200-clip balanced and attack-stratified subset. It is **not** official full-corpus evaluation and does not preserve natural class prevalence. Report it as a fixed balanced subsample, not ASVspoof 2019 full benchmark accuracy.

## 2. Score the frozen saved checkpoint exactly once

~~~cmd
.venv\Scripts\python.exe scripts\evaluate_ast_external.py "F:\Datasets\ASVspoof2019\LA" "F:\Datasets\ASVspoof2019\eval_fixed_100_each_seed42.csv" --training-manifest "F:\Datasets\ASVspoof2019\manifest.csv" --model-dir models\ast_partial_top1_v1 --bootstrap-samples 500 --seed 42 --output models\ast_2019_official_eval_fixed200_v1.json
~~~

The evaluator refuses exact audio SHA-256 overlap with **any** recording in the original train/development-derived manifest, and rejects declared source/speaker-group overlap. These checks are necessary but do not independently establish source-signal, microphone, or physical-room independence. The AST checkpoint is read-only, no thresholds are selected, and only the first 10 seconds of each source clip are analyzed.

The output includes full confusion matrix, accuracy and balanced accuracy, genuine false-positive rate, spoof recall/F1, ROC-AUC, average precision, and class-stratified bootstrap intervals.

Do **not** choose a different sample, threshold or checkpoint after observing this output and still claim that the result is untouched. Record a weak/negative result as evidence of generalization limits.

## 3. Interpretation

Prior within-dataset validation: **91.43%** accuracy, **87.25%** balanced accuracy, **ROC-AUC 0.9215**, real false-positive rate **3%**, and fake recall **77.5%** on 140 development-derived validation clips.

This experiment tests a different, previously unused official evaluation partition (provided no one has tuned against it). It does **not** establish cross-dataset generalization; follow docs/AST_EXTERNAL_VALIDATION.md for an external corpus such as ASVspoof 2021 DF.

No genuine result can be produced until the commands run on the operator's local FLAC files and saved AST checkpoint. Synthetic CI files test the **code**, not deepfake detection science. RIR independence, breath annotations and tamper-localization evidence remain open as described in docs/PROJECT1_REQUIREMENTS.md.

Original audio and model weights remain local, no dataset or checkpoint is overwritten, and nothing from the private local corpus is committed to GitHub.
