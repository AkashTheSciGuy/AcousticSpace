# Optional PyTorch + Hugging Face AST experiment

The official Project 1 PDF specifies an **Audio Spectrogram Transformer (AST)** fine-tuned for audio spoof detection. The preexisting logistic-regression models do not satisfy this part of the research objective.

This optional script supplies a training workflow; **no AST weights are included in GitHub**. The user's first local one-epoch head-only AST run is documented in `AST_EVALUATION.md` (47.14% validation accuracy); this is not full fine-tuning or independent evidence of deepfake accuracy. The AST endpoint reports unavailable when a compatible local checkpoint is not present.

## Important model and hardware requirements

AST input uses the Hugging Face feature extractor with 16-kHz audio (up to 10 s per clip). A pre-trained AST classification head is replaced for real/fake classes. The default script **freezes the transformer encoder and trains its classification head** to reduce memory usage; this is *not full transformer fine-tuning*. Supply --full-finetune to update the encoder as well, provided GPU VRAM is sufficient. A 4 GB GPU may not fit full AST fine-tuning: handle CUDA out-of-memory by using the head-only default or a larger device. No benchmark accuracy can be claimed before this training and independent evaluation.

If torch is already installed with working CUDA, check the local torch/torchaudio compatibility before installing new packages. Avoid blindly replacing a working CUDA configuration.

## Windows CMD

From backend:

```cmd
cd /d F:\Projects\AcousticSpace\backend
.venv\Scripts\python.exe -m pip install -r requirements-ast.txt
.venv\Scripts\python.exe scripts\train_ast.py "F:\Datasets\ASVspoof2019\LA" "F:\Datasets\ASVspoof2019\manifest.csv" --epochs 1 --batch-size 1 --accumulate 4 --output models\ast_model
```

The script verifies the grouped manifest, computes the AST log-mel features, trains only on train, chooses the best checkpoint by validation fake-class F1, and saves **safetensors** and a JSON report locally under `backend/models/ast_model`. Default uses head-only transfer learning, not full fine-tuning. The official evaluation split is not used.

Test metrics are omitted unless `--evaluate-test` is explicitly passed after model/hyperparameter selection. Repeated test usage can bias model decisions.

Before attempting full encoder fine-tuning, verify correct torch, torchaudio and transformers installations and available GPU memory. The recorded local run trained the head only; full fine-tuning has not been demonstrated on the target 4 GB GPU.

## Practical next experiment: partially fine-tune AST on the 4 GB GPU

The trainer now has an intermediate mode that updates the classification head, the **last N AST transformer encoder blocks**, and the final encoder normalization. Unlike a head-only run this is actual *partial encoder fine-tuning*; it is still **not** full-encoder fine-tuning.

Both Transformers 4.x and 5.x encoder layouts are explicitly supported. The script fails if it cannot identify the layers, instead of silently substituting a head-only training run.

From Windows CMD, verify that the source files exist before launching:

~~~cmd
cd /d F:\Projects\AcousticSpace\backend
dir F:\Datasets\ASVspoof2019\LA
dir F:\Datasets\ASVspoof2019\manifest.csv
.venv\Scripts\python.exe -c "import torch,transformers; print('torch:',torch.__version__,'transformers:',transformers.__version__,'CUDA:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
~~~

Then start with **only the last one encoder layer** and use a *new output folder*:

~~~cmd
.venv\Scripts\python.exe scripts\train_ast.py "F:\Datasets\ASVspoof2019\LA" "F:\Datasets\ASVspoof2019\manifest.csv" --epochs 3 --batch-size 1 --accumulate 4 --lr 0.00003 --selection-metric balanced_accuracy --patience 2 --train-last-layers 1 --gradient-checkpointing --output models\ast_partial_top1_v1
~~~

This command is an experiment, **not a guaranteed way to increase accuracy**. On 4 GB VRAM, even one trainable encoder block and gradient checkpointing may fail with CUDA out-of-memory. Do not reinstall or replace the existing working CUDA environment just to force this mode. If training fails, preserve the full traceback; reduce the model load or use a GPU with more memory. A head-only run with multiple epochs, validation-only early stopping and a unique output directory remains the lower-memory fallback.

Do **not** pass --full-finetune together with --train-last-layers. Setting --train-last-layers 0 preserves the old head-only behavior. Full fine-tuning updates all model weights and remains an optional experiment requiring suitable GPU memory. Training metadata records which mode actually ran; model evaluation reads the mode rather than calling partial fine-tuning "head-only."

Evaluate the new checkpoint on **validation only**, without overwriting an existing report:

~~~cmd
.venv\Scripts\python.exe scripts\evaluate_ast.py "F:\Datasets\ASVspoof2019\LA" "F:\Datasets\ASVspoof2019\manifest.csv" --model-dir models\ast_partial_top1_v1 --split val --threshold 0.5 --bootstrap-samples 500 --bootstrap-seed 42 --output models\ast_partial_top1_val_v1.json
~~~

Compare its *balanced accuracy*, **false-positive rate**, ROC-AUC and fake recall with the original checkpoint. If no improvement is demonstrated, report that negative result. **Do not tune against the previously examined test split.** The ASVspoof subset is small and source selection can limit generalizability. For a defensible final claim, confirm the selected model against a truly independent external corpus with documented recording and spoof provenance.


## Follow-up evaluation and second experiment

After the first checkpoint is saved, use [AST_EVALUATION.md](AST_EVALUATION.md) for **read-only validation diagnostics** and a separate head-only v2 experiment. Do not overwrite your working checkpoint.

## Limitations

- AST classifies speech recordings; it does not isolate RIR from unknown speech.
- Scores are uncalibrated softmax probabilities, not proof of tampering.
- The ASVspoof subset alone does not prove generalization or external accuracy.
- Fine-tuning without a controlled source/environment dataset does not train a physics-based voice/background mismatch detector.
