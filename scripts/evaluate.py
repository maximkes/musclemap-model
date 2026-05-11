from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _ensure_bert_score_stub() -> None:
    """MotionGPT metrics import bert_score; we only need backbone weights for eval."""

    if "bert_score" in sys.modules:
        return
    mod = types.ModuleType("bert_score")

    def score(*args, **kwargs):  # noqa: ANN001
        import torch as _torch

        return _torch.tensor([0.0]), _torch.tensor([0.0]), _torch.tensor([0.0])

    mod.score = score
    sys.modules["bert_score"] = mod


_ensure_bert_score_stub()

from src.dataset import MuscleActivationDataset  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.model import MuscleMAPModel, load_motiongpt  # noqa: E402


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


def _resolve_ckpt(config: dict[str, Any], ckpt_arg: str | None) -> Path:
    if ckpt_arg:
        return Path(ckpt_arg)
    ckpt_dir = Path(str(config["logging"]["checkpoint_dir"]))
    ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return ckpts[-1]


def _json_safe(obj: Any) -> Any:
    """Convert nested metrics (numpy arrays, scalars) for json.dumps."""

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _r2_table(r2: np.ndarray, muscle_names: list[str], k: int = 10) -> str:
    order = np.argsort(r2)
    bottom = order[:k]
    top = order[::-1][:k]
    lines = ["Top R2:"]
    for i in top:
        lines.append(f"{muscle_names[int(i)]:>24s}  {float(r2[int(i)]): .4f}")
    lines.append("")
    lines.append("Bottom R2:")
    for i in bottom:
        lines.append(f"{muscle_names[int(i)]:>24s}  {float(r2[int(i)]): .4f}")
    return "\n".join(lines)


def main() -> None:
    """Entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()

    config = _load_yaml(Path(args.config))
    split = str(args.split).lower()
    if split not in {"val", "test"}:
        raise ValueError("--split must be val or test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root = Path(str(config["data"]["dataset_root"]))
    ds = MuscleActivationDataset(dataset_root, config=config, split=split)

    backbone = load_motiongpt(config)
    model = MuscleMAPModel(backbone=backbone, config=config).to(device)
    model.eval()

    ckpt_path = _resolve_ckpt(config, args.ckpt)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)

    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    with torch.no_grad():
        for sample in ds:
            # Dataset returns padded max_T; use mask to keep real frames only.
            acts = sample["acts"]  # [T,80]
            mask = sample["mask"]  # [T]
            true = acts[mask].cpu().numpy().astype(np.float32, copy=False)  # [T_true,80]

            logits, _pred_log_T, _motion = model(text_tokens=[sample["text"]], motion_tokens=None)
            probs = torch.sigmoid(logits)[0].cpu().numpy().astype(np.float32, copy=False)  # [T_pred,80]

            # Align lengths by truncation to shortest.
            L = min(true.shape[0], probs.shape[0])
            all_true.append(true[:L])
            all_pred.append(probs[:L])

    pred_arr = np.stack(all_pred, axis=0)
    true_arr = np.stack(all_true, axis=0)

    metrics = compute_metrics(pred_arr, true_arr, ds.muscle_names)

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}_metrics.json"
    out_path.write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")

    print(f"Loaded ckpt: {ckpt_path}")
    print(f"Wrote: {out_path}")
    print(_r2_table(metrics["r2_per_muscle"], ds.muscle_names, k=10))


if __name__ == "__main__":
    main()

