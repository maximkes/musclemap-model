# musclemap-model — Main Context Prompt
# Paste this ONCE at the start of every new Cursor Agent session.

You are building **musclemap-model** (Project 2 of a biomechanics thesis).
All architectural decisions are FINAL. Your job is to implement them exactly.

## What this project does
Text description of a motion -> muscle activations float32 [T, 80] in [0,1].
80 muscles from OpenSim Rajagopal2016 model, 30 fps, ~172 frames median.

## Architecture (implement exactly — do not redesign)

### Backbone (frozen, vendor/MotionGPT/)
- MotionGPT: T5 encoder-decoder + VQ-VAE
- Freeze ALL parameters immediately on load
- Extract T5 decoder hidden states [B, T_tok, 768] via forward hook
- Extract T5 encoder output for LengthPredictor
- LM head stays active (produces SMPL-X motion for visualisation)

### LengthPredictor (trainable, src/head.py)
- encoder_hidden -> GlobalAvgPool -> Linear(768->128) -> GELU -> Linear(128->1)
- Output: log(T_frame); at inference: T_frame = int(exp(pred).round().clamp(min_T, max_T))

### ActivationHead (trainable, src/head.py)
- Cross-attention upsampler (NOT Conv1d):
  Keys/Values: decoder_hidden projected Linear(768->256)
  Queries: learned positional embeddings [1, T_frame, 256]
  nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
- 3x nn.TransformerEncoderLayer(d_model=256, nhead=4, batch_first=True, dropout=0.1)
- Linear(256->80, bias=False) -> raw logits [B, T_frame, 80]

### SVD warm-start (src/model.py, runs once after backbone load)
  W_lm = backbone.lm_head.weight.data  # [vocab_size, 768]
  _, _, Vt = torch.linalg.svd(W_lm, full_matrices=False)
  head.input_proj.weight.data.copy_(Vt[:256, :])

### Loss
- BCEWithLogitsLoss on raw logits (NOT MSE, NOT MSE-after-Sigmoid)
- + temporal smoothness on Sigmoid(logits)
- + SmoothL1Loss on length prediction

### DDP (CRITICAL)
- DistributedDataParallel wraps ActivationHead ONLY
- MotionGPT backbone runs unwrapped on each GPU

### Stage 2 (epoch >= unfreeze_after_epoch)
- Apply peft LoRA(r=8, target_modules=["q","v"]) to T5 decoder
- Add LoRA params to optimizer at lora_lr=5e-6

## Dataset
- Path: config/train.yaml -> data.dataset_root
- Per sample: activations.npy [T,80], smplx_322.npy [T,322], semantic_label.txt
- Split by CANONICAL ACTION NAME (group clips of same action), seed=42, 90/5/5
- Filter T < 30 before splitting
- Window T > max_T with 50% stride

## All .cursor/rules/ files are already written. Follow them strictly.

## Workflow
1. Run SCAFFOLD_PROMPT.md to create all stub files
2. Work through TASK_PROMPTS.md one task at a time
3. After each task: pytest -q && ruff check src/ scripts/ app/ tests/ — both must pass
4. Never move to next task while tests are red
