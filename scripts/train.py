from __future__ import annotations

import argparse
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _ensure_bert_score_stub() -> None:
    if "bert_score" in sys.modules:
        return
    mod = types.ModuleType("bert_score")

    def score(*args, **kwargs):  # noqa: ANN001
        import torch
        return torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0])

    mod.score = score
    sys.modules["bert_score"] = mod


_ensure_bert_score_stub()

from src.dataset import build_dataloaders  # noqa: E402
from src.model import MuscleMAPModel, load_motiongpt  # noqa: E402
from src.trainer import Trainer  # noqa: E402


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("PyYAML is required to load config files") from e
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping")
    return cfg


def _maybe_init_distributed() -> None:
    if torch.distributed.is_available() and ("RANK" in os.environ) and (not torch.distributed.is_initialized()):
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)


def _configure_logging() -> None:
    """Enable INFO logs for training; honor LOGLEVEL (default INFO)."""

    level_name = os.environ.get("LOGLEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            stream=sys.stderr,
        )
    else:
        root.setLevel(level)
    # Reduce Transformers chatter; embedding resize notes stay at INFO on their logger.
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.WARNING)


def main() -> None:
    _configure_logging()
    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = _load_yaml(Path(args.config))
    _maybe_init_distributed()

    train_dl, val_dl, _test_dl = build_dataloaders(config)
    log.info(
        "Dataloaders: train_samples=%d train_batches=%d val_samples=%d val_batches=%d",
        len(train_dl.dataset),
        len(train_dl),
        len(val_dl.dataset),
        len(val_dl),
    )
    if len(train_dl) == 0:
        raise RuntimeError(
            "Train DataLoader has zero batches. Increase data under data.dataset_root, "
            "lower training.batch_size, or fix splits so the train split is non-empty."
        )

    backbone = load_motiongpt(config)
    model = MuscleMAPModel(backbone=backbone, config=config)

    trainer = Trainer(config=config, model=model, train_loader=train_dl, val_loader=val_dl)
    trainer.fit()


if __name__ == "__main__":
    main()
