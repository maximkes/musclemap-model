from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.model import MuscleMAPModel, load_motiongpt

logger = logging.getLogger(__name__)


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


def _maybe_enable_p1_visualization(config: dict[str, Any]) -> None:
    p1_src_path = str(config.get("app", {}).get("p1_src_path", ""))
    if not p1_src_path:
        return
    sys.path.insert(0, p1_src_path)
    try:
        __import__("musclemap_data")  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        logger.warning("P1 visualization not available; continuing without it.")


def _load_muscle_names(config: dict[str, Any]) -> list[str]:
    dataset_root = Path(str(config["data"]["dataset_root"]))
    dataset_version = str(config["data"].get("dataset_version", ""))
    root = dataset_root / dataset_version if dataset_version else dataset_root
    names_path = root / str(config["data"].get("muscle_names_json", "muscle_names.json"))
    names = json.loads(names_path.read_text(encoding="utf-8"))
    if not isinstance(names, list):
        raise ValueError("muscle_names.json must be a list")
    return [str(x) for x in names]


def _plot_top_activations_png(acts: np.ndarray, muscle_names: list[str], top_k: int = 10) -> bytes:
    acts = np.asarray(acts, dtype=np.float32)
    mean_act = acts.mean(axis=0)
    idx = np.argsort(mean_act)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(acts[:, idx])
    ax.set_title("Top activations (sigmoid)")
    ax.set_xlabel("Frame")
    ax.legend([muscle_names[int(i)] for i in idx], fontsize=6, ncol=2, loc="upper right")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def predict_and_save(
    *,
    text: str,
    output_npy: Path,
    config: dict[str, Any],
    model: Any | None = None,
) -> np.ndarray:
    """Run inference and write activations to disk."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        backbone = load_motiongpt(config)
        model = MuscleMAPModel(backbone=backbone, config=config).to(device)
        model.eval()

    with torch.no_grad():
        logits, _pred_log_T, _motion = model(text_tokens=[text], motion_tokens=None)
        acts = torch.sigmoid(logits)[0].float().cpu().numpy().astype(np.float32, copy=False)

    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, acts)
    return acts


def main() -> None:
    """Entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-npy", required=True)
    parser.add_argument("--show-activations", action="store_true")
    parser.add_argument("--config", default="config/train.yaml")
    args = parser.parse_args()

    config = _load_yaml(Path(args.config))
    _maybe_enable_p1_visualization(config)

    out_path = Path(args.output_npy)
    acts = predict_and_save(text=args.text, output_npy=out_path, config=config, model=None)
    print(f"Wrote {out_path}")

    if args.show_activations:
        names = _load_muscle_names(config)
        png = _plot_top_activations_png(acts, names, top_k=10)
        png_path = out_path.with_suffix(".png")
        png_path.write_bytes(png)
        print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()

