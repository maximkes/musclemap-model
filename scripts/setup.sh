#!/usr/bin/env bash
# Run once after cloning: bash scripts/setup.sh
set -euo pipefail

# Pinned MotionGPT commit — update only after full test-suite validation
MOTIONGPT_COMMIT="MotionGPT-V1.0"
MOTIONGPT_REPO="https://github.com/OpenMotionLab/MotionGPT.git"

CONDA_PYTHON="/opt/anaconda3/envs/musclemap-model/bin/python"
CONDA_PIP="/opt/anaconda3/envs/musclemap-model/bin/pip"

echo "==> Creating conda env musclemap-model"
if conda env list | grep -q "musclemap-model"; then
  echo "    Already exists — skipping"
else
  if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "    Detected macOS: creating env without pytorch-cuda"
    conda create -n musclemap-model python=3.10 -y
    conda install -n musclemap-model -c pytorch -c conda-forge \
      pytorch numpy scipy scikit-learn matplotlib tqdm pyyaml -y
    conda run -n musclemap-model pip install \
      "transformers>=4.38,<5.0" \
      "peft>=0.10" \
      "wandb>=0.16" \
      "gradio>=4.20,<5.0" \
      "einops>=0.7" \
      "accelerate>=0.27" \
      "ruff>=0.4" \
      "pytest>=7.4" \
      "pytest-cov>=4.1" \
      "pyyaml>=6.0"
  else
    conda env create -f environment.yml
  fi
fi

echo "==> Cloning MotionGPT at ${MOTIONGPT_COMMIT}"
mkdir -p vendor
if [ ! -d "vendor/MotionGPT/.git" ]; then
  git clone "${MOTIONGPT_REPO}" vendor/MotionGPT
fi
cd vendor/MotionGPT && git fetch --tags --quiet && git checkout "${MOTIONGPT_COMMIT}" && cd ../..
MOTIONGPT_RESOLVED_COMMIT="$(git -C vendor/MotionGPT rev-parse HEAD)"

echo "==> Installing MotionGPT requirements"
# No --quiet: surface any install failures immediately
"${CONDA_PIP}" install -r vendor/MotionGPT/requirements.txt

echo "==> Ensuring spaCy is installed (may be absent from MotionGPT requirements)"
"${CONDA_PIP}" install spacy

echo "==> Installing spaCy English model"
# This download hits GitHub (compatibility.json) and can time out on restricted networks.
# Make it best-effort so setup doesn't fail.
if [[ "${SKIP_SPACY_MODEL_DOWNLOAD:-0}" == "1" ]]; then
  echo "    SKIP_SPACY_MODEL_DOWNLOAD=1 set — skipping"
else
  "${CONDA_PYTHON}" - <<'PY'
import importlib.util
spec = importlib.util.find_spec("en_core_web_sm")
print("    en_core_web_sm already installed — skipping" if spec is not None else "    en_core_web_sm missing — attempting download")
PY
  if ! "${CONDA_PYTHON}" -m spacy download en_core_web_sm --quiet; then
    echo "    spaCy model download failed (network/timeout). Continuing."
    echo "    You can retry later with:"
    echo "      ${CONDA_PYTHON} -m spacy download en_core_web_sm"
  fi
fi

echo "==> (Optional) If you see protobuf errors, pin protobuf<5"
echo "    ${CONDA_PIP} install 'protobuf<5'"

echo "==> Writing commit hash to config"
"${CONDA_PYTHON}" - <<PY
import yaml, pathlib
p = pathlib.Path("config/train.yaml")
cfg = yaml.safe_load(p.read_text())
cfg["model"]["motiongpt_commit"] = "${MOTIONGPT_RESOLVED_COMMIT}"
p.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
PY

echo ""
echo "Next steps:"
echo "  1. conda activate musclemap-model"
echo "  2. Download MotionGPT dependencies + pretrained models"
echo "     cd vendor/MotionGPT"
echo "     bash prepare/download_smpl_model.sh"
echo "     bash prepare/prepare_t5.sh"
echo "     bash prepare/download_t2m_evaluators.sh"
echo "     bash prepare/download_pretrained_models.sh"
echo "     cd ../../"
echo "     - Sanity check (should exist):"
echo "         ls vendor/MotionGPT/checkpoints/MotionGPT-base/"
echo "     - If you store checkpoints elsewhere, update config/train.yaml -> model.motiongpt_ckpt accordingly"
echo "  3. Edit config/train.yaml -> data.dataset_root"
echo "  4. IMPORTANT: run training from the musclemap-model repo root (NOT inside vendor/MotionGPT)"
echo "     Quick check (should end with /musclemap-model):"
echo "       pwd"
echo "     Quick check (should point to conda env python 3.10):"
echo "       python -c \"import sys; print(sys.executable, sys.version)\""
echo "  5. Train:"
echo "     - macOS (CPU/MPS):"
echo "         python scripts/train.py --config config/train.yaml"
echo "     - Linux + NVIDIA (2 GPUs):"
echo "         torchrun --nproc_per_node=2 scripts/train.py --config config/train.yaml"