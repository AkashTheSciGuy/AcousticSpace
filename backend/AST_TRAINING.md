# Optional PyTorch + Hugging Face AST experiment

The official Project 1 PDF specifies an **Audio Spectrogram Transformer (AST)** fine-tuned for audio spoof detection. The preexisting logistic-regression models do not satisfy this part of the research objective.

This optional script supplies a training workflow; **no AST weights are included or trained in GitHub**. The new AST endpoint must report unavailable until a locally trained model exists.

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

Before running full fine-tuning, verify correct torch, torchaudio and transformers installations on the actual machine; the optional dependencies and training loop have not been exercised on the user's local GPU.

## Limitations

- AST classifies speech recordings; it does not isolate RIR from unknown speech.
- Scores are uncalibrated softmax probabilities, not proof of tampering.
- The ASVspoof subset alone does not prove generalization or external accuracy.
- Fine-tuning without a controlled source/environment dataset does not train a physics-based voice/background mismatch detector.
