# Scaffold Prompt — Run Second (after CURSOR_MAIN_PROMPT.md)
# Creates all empty stub files. No implementation — stubs only.

Create these files as importable stubs:

src/__init__.py — empty

src/dataset.py
  class MuscleActivationDataset(torch.utils.data.Dataset)
  def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]

src/head.py
  class LengthPredictor(nn.Module)
    forward(encoder_hidden: Tensor) -> Tensor  # [B, 1]
    predict_T(encoder_hidden: Tensor, min_T: int, max_T: int) -> Tensor  # [B]
  class ActivationHead(nn.Module)
    forward(decoder_hidden: Tensor, T_frame: int) -> Tensor  # [B, T_frame, 80]

src/model.py
  def load_motiongpt(config: dict) -> nn.Module
  class MuscleMAPModel(nn.Module)
    forward(text_tokens, motion_tokens=None) -> tuple[Tensor, Tensor, Any]

src/losses.py
  def activation_loss(logits, targets, mask, config) -> tuple[Tensor, dict]

src/metrics.py
  def compute_metrics(pred: np.ndarray, true: np.ndarray, muscle_names: list[str]) -> dict

src/trainer.py
  class Trainer
    def train_epoch(self) -> dict
    def val_epoch(self) -> dict
    def save_checkpoint(self, epoch: int, val_loss: float) -> None
    def load_checkpoint(self) -> int  # returns start epoch

scripts/train.py — argparse --config; print stub message

scripts/evaluate.py — argparse --config --split; print stub message

app/app.py — import gradio as gr; print stub message (with matplotlib.use('Agg') first)

app/demo.py — argparse --text --output-npy --show-activations; print stub message

tests/conftest.py — shared fixtures: tiny_config, tiny_dataset_dir, synthetic_batch
tests/test_dataset.py — one test: from src.dataset import MuscleActivationDataset; assert True
tests/test_head.py — one test: from src.head import ActivationHead, LengthPredictor; assert True
tests/test_model.py — one test: from src.model import MuscleMAPModel; assert True
tests/test_metrics.py — one test: from src.metrics import compute_metrics; assert True

After creating all stubs:
  pytest -q           # must pass (all stubs importable)
  ruff check src/ scripts/ app/ tests/  # must pass

Do not implement anything. Only create the stubs.
