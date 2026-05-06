# musclemap-model — Text to Muscle Activations

Takes a natural-language motion description and outputs muscle activation patterns:
`float32 [T, 80]`, values in `[0, 1]`, 80 muscles (Rajagopal2016) at 30 fps.

Built on frozen [MotionGPT](https://github.com/OpenMotionLab/MotionGPT) with a
trainable parallel activation head + LoRA fine-tuning (Stage 2).

## Prerequisites
- Linux / macOS ARM / Windows WSL2
- CUDA 12.x
- Conda or Micromamba

## Setup
```bash
# Create the conda env (first time only)
#
# Linux + NVIDIA GPU:
conda env create -f environment.yml
#
# macOS (osx-arm64): pytorch-cuda is not available, so install PyTorch without it:
# conda create -n musclemap-model python=3.10 -y
# conda activate musclemap-model
# conda install -c pytorch -c conda-forge pytorch numpy scipy scikit-learn matplotlib tqdm pyyaml -y
# pip install -r <(python - <<'PY'
# print("\n".join([
#   "transformers>=4.38,<5.0",
#   "peft>=0.10",
#   "wandb>=0.16",
#   "gradio>=4.20,<5.0",
#   "einops>=0.7",
#   "accelerate>=0.27",
#   "ruff>=0.4",
#   "pytest>=7.4",
#   "pytest-cov>=4.1",
# ]))
# PY
# )
#
# If the env already exists:
# conda env update -n musclemap-model -f environment.yml --prune

bash scripts/setup.sh
conda activate musclemap-model
# Populate vendor/MotionGPT + download MotionGPT checkpoint (see MotionGPT docs)
# Edit config/train.yaml -> data.dataset_root
```

## Train
```bash
torchrun --nproc_per_node=2 scripts/train.py --config config/train.yaml
```

## Evaluate
```bash
python scripts/evaluate.py --config config/train.yaml --split test
python scripts/evaluate.py --config config/train.yaml --split val --ckpt checkpoints/epoch_0009.pt
```

## Demo
```bash
python app/app.py              # Gradio at http://localhost:7860
python app/demo.py --text "a person walks forward" --output-npy out.npy
python app/demo.py --text "a person walks forward" --output-npy out.npy --show-activations
```

## Output format
`activations.npy`: float32 numpy array `[T, 80]`, values in `[0, 1]`.
Muscle order matches `{dataset_root}/muscle_names.json`.
Identical format to musclemap-data (Project 1) output.

## Dataset dependency
Point `config/train.yaml -> data.dataset_root` at the musclemap-data output directory.
Note: Project 1 activations were generated with RRA disabled (IK -> Static Optimization
directly) as Motion-X++ lacks ground reaction forces for full RRA.
