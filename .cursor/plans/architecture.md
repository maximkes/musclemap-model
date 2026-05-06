# Architecture Plan — APPROVED

## Training data flow
```
text str
  -> T5 tokenizer -> input_ids [B, L]
  -> T5 Encoder -> encoder_hidden [B, L, 768]
  -> LengthPredictor -> log_T_pred [B, 1]
  -> T5 Decoder (teacher-forced via VQ-VAE(smplx_322)) -> decoder_hidden [B, T_tok, 768]
  -> ActivationHead.input_proj Linear(768->256) -> [B, T_tok, 256]
  -> CrossAttention (queries: learned pos_emb[T_frame,256]) -> [B, T_frame, 256]
  -> 3x TransformerEncoderLayer -> [B, T_frame, 256]
  -> Linear(256->80) -> logits [B, T_frame, 80]
  -> BCEWithLogitsLoss(logits, acts_gt) + smoothness(Sigmoid(logits)) + SmoothL1(log_T_pred)
```

## Inference data flow
```
text str
  -> T5 Encoder -> LengthPredictor -> T_frame (int)
  -> MotionGPT.generate() -> decoder hidden states [B, T_tok, 768]
  -> ActivationHead -> logits -> Sigmoid -> activations [T_frame, 80]
  -> (also) VQ-VAE Decoder -> SMPL-X pose for visualisation
```

## DDP topology
- Both GPUs: full frozen MotionGPT backbone (not synced — frozen)
- Both GPUs: activation_head wrapped in DistributedDataParallel (synced)
- Gradients flow only through activation_head (+ LoRA in Stage 2)

## Stage 2 transition (at epoch = unfreeze_after_epoch)
1. Save checkpoint
2. peft.get_peft_model(backbone, LoraConfig(...)) applied to T5 decoder
3. Add LoRA params as new optimizer param group at lora_lr
4. Continue training

## File ownership
- vendor/MotionGPT/: READ ONLY
- src/head.py: ActivationHead, LengthPredictor
- src/model.py: MuscleMAPModel, load_motiongpt, SVD init
- src/dataset.py: MuscleActivationDataset, build_dataloaders
- src/trainer.py: Trainer (DDP, W&B, checkpointing)
- src/losses.py: activation_loss
- src/metrics.py: compute_metrics
- scripts/train.py: torchrun entry point
- scripts/evaluate.py: test-time evaluation
- app/app.py: Gradio (adapted from vendor/MotionGPT/app.py)
- app/demo.py: CLI (adapted from vendor/MotionGPT/demo.py)
