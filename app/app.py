from __future__ import annotations

import hashlib
import io
import logging
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
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


def _hash_name(prompt: str) -> str:
    h = hashlib.md5(f"{prompt}{time.time()}".encode("utf-8")).hexdigest()  # noqa: S324
    return h[:8]


def _heatmap_png(acts: np.ndarray, muscle_names: list[str], top_k: int = 20) -> bytes:
    """Render a top-k muscle heatmap to PNG bytes."""

    acts = np.asarray(acts, dtype=np.float32)
    if acts.ndim != 2:
        raise ValueError(f"acts must be [T,N], got {acts.shape}")
    T, N = acts.shape
    if N != len(muscle_names):
        raise ValueError("muscle_names length mismatch")

    mean_act = acts.mean(axis=0)
    idx = np.argsort(mean_act)[::-1][:top_k]
    data = acts[:, idx].T  # [K, T]
    labels = [muscle_names[int(i)] for i in idx]

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Frame")
    ax.set_title("Top muscle activations (sigmoid)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def _load_model(config: dict[str, Any]) -> tuple[MuscleMAPModel, list[str], torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_motiongpt(config)
    model = MuscleMAPModel(backbone=backbone, config=config).to(device)
    model.eval()

    # Muscle names are loaded by dataset normally; here we read from JSON path.
    dataset_root = Path(str(config["data"]["dataset_root"]))
    dataset_version = str(config["data"].get("dataset_version", ""))
    root = dataset_root / dataset_version if dataset_version else dataset_root
    names_path = root / str(config["data"].get("muscle_names_json", "muscle_names.json"))
    muscle_names = _load_yaml(names_path) if names_path.suffix in {".yml", ".yaml"} else None
    if muscle_names is None:
        import json

        muscle_names = json.loads(names_path.read_text(encoding="utf-8"))
    if not isinstance(muscle_names, list):
        raise ValueError("muscle_names.json must be a list")
    return model, [str(x) for x in muscle_names], device


def predict_and_save(
    *,
    prompt: str,
    model: Any,
    muscle_names: list[str],
    out_dir: Path,
) -> tuple[Path, bytes, np.ndarray]:
    """Run inference and save activations .npy (float32 [T,80] after sigmoid)."""

    if not prompt.strip():
        raise ValueError("Prompt is empty")

    with torch.no_grad():
        logits, _pred_log_T, _motion = model(text_tokens=[prompt], motion_tokens=None)
        acts = torch.sigmoid(logits)[0].float().cpu().numpy().astype(np.float32, copy=False)  # [T, 80]

    name = _hash_name(prompt)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"activations_{name}.npy"
    np.save(out_path, acts)

    png = _heatmap_png(acts, muscle_names=muscle_names, top_k=20)
    return out_path, png, acts


def build_app(config_path: Path) -> gr.Blocks:
    """Build the Gradio UI."""

    config = _load_yaml(config_path)
    _maybe_enable_p1_visualization(config)

    model, muscle_names, device = _load_model(config)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    def predict(prompt: str) -> tuple[str, bytes]:
        """Gradio prediction function."""
        out_path, png, _acts = predict_and_save(prompt=prompt, model=model, muscle_names=muscle_names, out_dir=out_dir)
        return str(out_path), png

    with gr.Blocks() as demo:
        gr.Markdown("# MuscleMAP — Text to Muscle Activations")
        prompt = gr.Textbox(label="Motion description", placeholder="e.g., a person walks forward", lines=2)
        btn = gr.Button("Predict")
        out_file = gr.File(label="Download Activations (.npy)")
        out_img = gr.Image(label="Activation heatmap (top-20 muscles)")

        btn.click(predict, inputs=[prompt], outputs=[out_file, out_img], api_name="predict")

    _ = (device,)  # keep reference for debugging if needed
    return demo


def main() -> None:
    """Entry point."""

    config_path = Path("config/train.yaml")
    app = build_app(config_path)
    cfg = _load_yaml(config_path)
    port = int(cfg.get("app", {}).get("gradio_port", 7860))
    share = bool(cfg.get("app", {}).get("gradio_share", False))
    app.launch(server_port=port, share=share)


if __name__ == "__main__":
    main()

