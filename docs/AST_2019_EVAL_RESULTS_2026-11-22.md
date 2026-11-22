# AST — Fixed official ASVspoof 2019 LA evaluation subset results

**Run date:** 22 November 2026  
**Evidence:** User-reported execution on Windows with a local RTX 3050 Laptop GPU. The GitHub repository contains neither the selected evaluation FLACs nor the trained AST checkpoint.  
**Scope:** Previously unused official ASVspoof 2019 LA evaluation-partition SUBSET, not full benchmark or cross-dataset evaluation.

## Frozen experimental conditions

- Official evaluation protocol SHA-256: 84fce576efff092269a0a6ad247d2e9f2f7b4d4e25a4545efb2518da92708036
- Fixed manifest SHA-256: 7b0c6e3b12c170b8b59ffba95644f382023d8a82142a4e4cfb37e83e41dac0d2
- Original training/dev manifest SHA-256: 1754879b8550e6c6c403d70d13698b77f9f2d952ed691699a38768c85a3a55a7
- Sample selected BEFORE prediction: seed 42, 100 genuine + 100 spoof, genuine spread across speakers and spoof spread across official attack IDs A07–A19
- Distinct official speaker groups in subset: 67
- Exact file-hash overlap with original train/validation/development test manifest: 0
- Declared speaker-group overlap with original manifest: 0
- Checkpoint fixed: backend/models/ast_partial_top1_v1 — one AST encoder layer + classifier fine-tuned for three epochs
- Threshold fixed: 0.50; first 10 seconds of each audio recording analyzed
- Bootstrap: 500 class-stratified iterations, seed 42
- Locally retained result: backend/models/ast_2019_official_eval_fixed200_v1.json (not committed)

## Measured classification results

| Measure | Value |
| --- | ---: |
| Samples | 200 (100 real, 100 fake) |
| Accuracy | **87.00%** |
| Balanced accuracy | **87.00%** |
| Genuine specificity | **99.00%** |
| Genuine false-positive rate | **1.00%** |
| Fake recall | **75.00%** |
| Fake precision | **98.6842%** |
| Fake F1 | **85.2273%** |
| ROC-AUC | **0.9500** |
| Average precision for fake class | **0.962926** |
| Samples predicted fake | 76 |

Confusion matrix (rows: actual, columns: predicted):

| Actual | Predicted genuine | Predicted fake |
| --- | ---: | ---: |
| Genuine (100) | **99** | 1 |
| Fake (100) | **25** | **75** |

**Critical limitation:** 25% of the spoof recordings were missed. Fake precision is measured at an artificially balanced 50:50 prevalence and should not be treated as real-world positive predictive value.

## Reported bootstrap 95% intervals

These intervals resample individual files within classes, assuming independence. They do not adjust for correlated source speakers/attack conditions, evaluation partition selection, or domain shift.

| Metric | Point estimate | Bootstrap 95% interval |
| --- | ---: | ---: |
| Balanced accuracy | 0.8700 | **0.8250–0.9100** |
| Genuine false-positive rate | 0.0100 | **0.0000–0.0300** |
| Fake recall | 0.7500 | **0.6600–0.8300** |
| Fake F1 | 0.852273 | **0.791743–0.901099** |
| ROC-AUC | 0.9500 | **0.921948–0.977205** |

## Comparison with development validation

On the earlier 140-item development validation partition, the same saved AST checkpoint achieved 91.43% accuracy, 87.25% balanced accuracy, ROC-AUC 0.9215, genuine false-positive rate 3%, and fake recall 77.5%.

On this fixed official evaluation-partition subset, balanced accuracy was 87.00%, ROC-AUC 0.9500, genuine false-positive rate 1%, and fake recall 75%. The model was frozen. Overall accuracy and precision are not directly comparable across the two differently class-balanced populations. An apparent change in ROC-AUC is a difference in sample composition, not a model improvement.

## Research integrity and unfinished objectives

The result is a meaningful **separate within-ASVspoof-2019 evaluation-partition test**. It is NOT the official full-corpus benchmark, NOT cross-dataset generalization, and NOT evidence of physical room-response inference.

Still unverified: source-specific blind RIR extraction from ordinary mixed mono speech, room-consistency accuracy on independently proven physical rooms, independent annotated breathing/syllable and manipulation localization accuracy, and performance on a genuinely different spoof corpus.

The operator must retain the original protocol, SHA-256-pinned subset manifest, and first report. Never change the subset/seed/threshold after inspecting this evaluation result while claiming it remains untouched. Locally saved recordings and model weights are intentionally not committed to GitHub.
