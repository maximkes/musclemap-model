# Task Prompts — One per Cursor Agent session
# Always run /run-tests before starting a new task.

---
## Task 1 — src/dataset.py

Implement MuscleActivationDataset and build_dataloaders.

Spec (.cursor/rules/005-dataset.mdc):
- Scan dataset_root for valid sequence dirs (activations.npy + smplx_322.npy + semantic_label.txt)
- Load and validate muscle_names.json (assert len == n_muscles)
- Filter sequences with T < config.data.min_T
- Group by canonical name; shuffle groups seed=42; split 90/5/5
- Window T > max_T with stride = max_T // 2
- Return dict: {text, motion[T,322], acts[T,80], mask[T], true_T}
- Label cleaning: prefer semantic_label.txt if not numeric; else clean_label(sample_id)
- Version selector: load from dataset_root/dataset_version/ if set

Write tests/test_dataset.py using tmp_path fixture (no real dataset needed):
- 10 synthetic sequences, 3 canonical groups, T in range 10-300
- Test: split has no canonical group overlap between train/val/test
- Test: windowing produces correct count of windows for long sequence
- Test: mask is True for real frames, False for padding
- Test: min_T filter removes sequences with T=10
- Test: numeric blob label falls back to clean_label
- Test: dataset_version subdir is used when set

Run: pytest tests/test_dataset.py -v

---
## Task 2 — src/head.py: LengthPredictor

Implement LengthPredictor (see .cursor/rules/003-model.mdc):
- GlobalAvgPool -> Linear(768->128) -> GELU -> Dropout(0.1) -> Linear(128->1)
- Output: log(T_frame)
- predict_T method with clamp
- Xavier uniform init

Write tests (CPU only, batch=2, T_enc=10):
- Output shape [B, 1]
- predict_T output within [min_T, max_T]
- Gradients flow through module

Run: pytest tests/test_head.py::TestLengthPredictor -v

---
## Task 3 — src/head.py: ActivationHead

Implement ActivationHead (cross-attention upsampler, see .cursor/rules/003-model.mdc):
- input_proj: Linear(768->256)
- Cross-attention: learned pos queries + nn.MultiheadAttention(256, 4, batch_first=True)
- 3x nn.TransformerEncoderLayer(256, 4, batch_first=True, dropout=0.1)
- output_proj: Linear(256->80, bias=False)
- Returns raw logits — NO sigmoid in forward

Write tests (CPU, batch=2, T_tok=8):
- forward(hidden, T_frame=16) -> [2, 16, 80]
- forward(hidden, T_frame=32) -> [2, 32, 80]
- No sigmoid: output can be negative
- Gradients flow

Run: pytest tests/test_head.py -v

---
## Task 4 — src/model.py: MuscleMAPModel

Implement load_motiongpt and MuscleMAPModel (see .cursor/rules/003-model.mdc):

load_motiongpt(config):
- Load from vendor/MotionGPT/ using their utilities
- Freeze all parameters
- Register forward hooks on T5 decoder (last layer) and T5 encoder
- Return backbone

MuscleMAPModel:
- Holds backbone + LengthPredictor + ActivationHead
- On first forward: run SVD warm-start on ActivationHead.input_proj from backbone.lm_head.weight
- forward(text_tokens, motion_tokens=None):
  - training (motion_tokens given): teacher forcing via VQ-VAE encoder
  - inference (no motion_tokens): backbone.generate()
  - Return (logits [B,T_frame,80], pred_log_T [B,1], motion_output)
- parameters_to_train(): yields head + predictor params only
- apply_lora(config): peft LoRA on T5 decoder

Write tests using mocked backbone (no MotionGPT download needed):
- SVD warm-start on random [512,768] weight matrix -> head.input_proj.weight has correct shape
- Only head/predictor params are trainable before apply_lora
- After apply_lora, LoRA params also trainable
- Teacher forcing path: output shapes correct

Run: pytest tests/test_model.py -v

---
## Task 5 — src/losses.py and src/metrics.py

Implement activation_loss:
- BCEWithLogitsLoss (apply mask: zero out padded frames)
- Smoothness: Sigmoid(logits)[:,1:] - Sigmoid(logits)[:,:-1]).abs().mean()
- Length: SmoothL1Loss on log predictions
- Return (total, {bce, smooth, length})

Implement compute_metrics:
- MPJAE, per-muscle R2 (sort + report top/bottom 10), mean bias, Pearson r, smoothness, length_mae
- Return complete dict

Write tests:
- Loss is near zero when sigmoid(logits) perfectly matches targets
- Mask correctly zeroes padded frames in loss
- Metrics return correct shapes on synthetic data (batch=4, T=16, N=80)
- R2 is 1.0 when pred == true

Run: pytest tests/test_losses.py tests/test_metrics.py -v

---
## Task 6 — src/trainer.py and scripts/train.py

Implement Trainer:
- DDP on activation_head only (not backbone)
- bf16 autocast, clip_grad_norm, zero_grad(set_to_none=True)
- AdamW, linear warmup + cosine decay
- W&B logging (rank 0 only): scalars, metrics, histograms, sample table
- Atomic checkpoint save/load; auto-resume
- Stage 2 transition: save checkpoint, apply_lora, rebuild optimizer

Implement scripts/train.py:
- Parse --config
- torchrun-compatible (LOCAL_RANK env var)
- W&B init on rank 0 only
- Call Trainer.fit()

Write tests:
- Trainer.train_epoch on CPU with mocked backbone + tiny dataset (batch=1, T=16, N=80)
- Checkpoint save -> load -> loss value preserved
- Stage 2 transition adds LoRA params to optimizer

Run: pytest tests/test_trainer.py -v

---
## Task 7 — scripts/evaluate.py

Implement evaluation script:
- Load model from checkpoint (specified via --ckpt or latest in checkpoint_dir)
- Run inference on val or test split
- compute_metrics on all sequences
- Save to results/{split}_metrics.json
- Print sorted per-muscle R2 table
- W&B log if active

Test manually: python scripts/evaluate.py --config config/train.yaml --split val

---
## Task 8 — app/app.py and app/demo.py

Implement app/app.py (Gradio, see .cursor/rules/006-app.mdc):
1. cp vendor/MotionGPT/app.py app/app.py
2. Add: import matplotlib; matplotlib.use('Agg')  # FIRST
3. Load MuscleMAPModel at startup
4. Add gr.File output for activations
5. Add activation heatmap gr.Image (top-20 muscles, viridis, headless PNG)
6. api_name="predict" on inference function
7. P1 visualization via sys.path (graceful fallback if not found)

Implement app/demo.py:
1. cp vendor/MotionGPT/demo.py app/demo.py
2. Add --output-npy and --show-activations flags

Verify: python app/app.py (starts without error)

---
## Task 9 — Integration tests and README

Write tests/test_integration.py:
- End-to-end: synthetic text -> mocked model -> activations [1, T, 80], dtype float32
- After Sigmoid: all values in [0, 1]
- app inference function writes .npy with correct shape + dtype
- demo.py --output-npy writes correct file

Update README.md with full setup, training, evaluation, demo instructions.

Final gate:
  pytest -q              # ALL tests pass
  ruff check src/ scripts/ app/ tests/  # zero errors
