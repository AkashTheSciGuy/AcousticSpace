# AST evaluation and experiment 2

The first locally trained Audio Spectrogram Transformer (`models/ast_model`) produced:
- Validation: 140 recordings = 100 real / 40 fake
- Threshold 0.5: 47.14% accuracy; 80% fake recall; 32.65% fake precision
- Confusion matrix (actual real/fake rows): [[34, 66], [8, 32]]
- Real false-positive rate: 66%, and balanced accuracy: 57.0%

This first run is a **head-only classifier trained for one epoch**, NOT a fine-tuned AST encoder or a validated RIR mismatch detector. The uncalibrated softmax score is not forensic confidence.

## A. Analyze the EXISTING checkpoint first

From Windows CMD, in backend, with a working CUDA-enabled PyTorch and Transformers installation:

```cmd
cd /d F:\Projects\AcousticSpace\backend
.venv\Scripts\python.exe scripts\evaluate_ast.py "F:\Datasets\ASVspoof2019\LA" "F:\Datasets\ASVspoof2019\manifest.csv" --model-dir models\ast_model --split val --select-threshold --output models\ast_v1_validation.json
```

The tool verifies manifest hashes and the speaker/source-disjoint grouping. It scores the *existing* weights, with no training. It reports:
- Accuracy, balanced accuracy, fake precision/recall/F1 and actual×predicted confusion matrix at threshold 0.5.
- Real specificity, real false-positive rate, ROC-AUC and average precision.
- An **exploratory validation-only** threshold that maximizes balanced accuracy, ties favoring fewer real false positives.

**Do not choose a threshold on the test split.** Because the same validation data selects the threshold, its selected score is optimistic. The live API's 0.5 threshold is deliberately unchanged.

## B. Controlled second training run

**Do not overwrite** the already working `models/ast_model` directory. Start a distinct checkpoint:

```cmd
.venv\Scripts\python.exe scripts\train_ast.py "F:\Datasets\ASVspoof2019\LA" "F:\Datasets\ASVspoof2019\manifest.csv" --epochs 5 --batch-size 1 --accumulate 4 --lr 0.0003 --selection-metric balanced_accuracy --patience 2 --output models\ast_model_v2
```

The new options let us compare multiple epochs, print training loss, keep the epoch with the highest *validation* balanced accuracy, and stop after two epochs without improvement. A higher learning rate is appropriate to **experiment with** for the newly initialized classifier head; it does not guarantee better accuracy. The frozen AST encoder uses eval mode during head training. On a 4 GB GPU, use batch size 1; CUDA out-of-memory may still require an alternative environment.

The second run still trains **only the classification head**. Full transformer fine-tuning is not demonstrated on this hardware. The training script leaves official evaluation recordings untouched.

## C. Evaluate the second checkpoint on validation

```cmd
.venv\Scripts\python.exe scripts\evaluate_ast.py "F:\Datasets\ASVspoof2019\LA" "F:\Datasets\ASVspoof2019\manifest.csv" --model-dir models\ast_model_v2 --split val --select-threshold --output models\ast_v2_validation.json
```

Compare v1 versus v2 ROC-AUC and average precision (ranking quality), balanced accuracy at 0.5 (default decisions), and the false-positive rate (real recordings mistaken as fake). Threshold selection can alter the precision/recall trade-off without improving underlying ranking.

## D. Only AFTER the final checkpoint/threshold choice

Reserve the local `test` split for one final check with a **preselected numerical threshold**; for example, if the chosen threshold was 0.65:

```cmd
.venv\Scripts\python.exe scripts\evaluate_ast.py "F:\Datasets\ASVspoof2019\LA" "F:\Datasets\ASVspoof2019\manifest.csv" --model-dir models\ast_model_v2 --split test --threshold 0.65 --output models\ast_v2_test.json
```

0.65 is only an example, **not a recommended calibrated threshold**. If the same test split has already informed previous model/hyperparameter choices, it is no longer independent. Reserve a fresh dataset for a credible final result. Do not iterate to improve test metrics.

To try v2 in the existing dashboard *after verification*, start the backend with a local model override in its CMD terminal (no file moves):

```cmd
set "ACOUSTICSPACE_AST_MODEL=F:\Projects\AcousticSpace\backend\models\ast_model_v2"
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Stop that server and clear/change the environment variable to return to v1. The dashboard still uses a fixed 0.5 decision threshold; alternative thresholds are **offline analysis only**.

## E. Optional uncertainty estimates for fixed-threshold evaluation

The offline evaluator now supports **seeded, class-stratified 95% nonparametric bootstrap intervals**. It resamples real and fake examples separately at a **fixed** threshold, then reports intervals for balanced accuracy, real false-positive rate, fake recall/F1 and ROC-AUC. For example, using your already trained v1 checkpoint and a **new report filename**:

~~~cmd
.venv\Scripts\python.exe scripts\evaluate_ast.py "F:\Datasets\ASVspoof2019\LA" "F:\Datasets\ASVspoof2019\manifest.csv" --model-dir models\ast_model --split val --threshold 0.5 --bootstrap-samples 500 --bootstrap-seed 42 --output models\ast_v1_uncertainty.json
~~~

These uncertainty intervals do **not** turn 47.14% accuracy into a reliable model, and they do **not** correct speaker/source/room correlations, dataset shift or threshold-selection bias. If the same validation set chooses a threshold, intervals on those data are optimistic. The evaluated checkpoint's `training_meta.json` is read only to **report** head-only/full-encoder status, never to infer unrecorded training.

The output also contains the source manifest's SHA-256 checksum so later reports can identify an unchanged evaluation manifest. Model files are never overwritten.

## Important research limits

Neither v1 nor v2 extracts a true room impulse response from speech; ASVspoof real/fake labels are not ground-truth room response measurements. These are experimental audio classifiers. A small ASVspoof subset does not support forensic deployment claims. The scoring code reports metrics transparently but does not guarantee good generalization.
