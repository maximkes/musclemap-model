from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from src.head import ActivationHead, LengthPredictor

from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval)


logger = logging.getLogger(__name__)


def _freeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False

def _try_register_hidden_hooks(backbone: nn.Module, on_encoder: Any, on_decoder: Any) -> None:
    """Hook T5 encoder/decoder hidden states. Path: backbone.lm.language_model.{encoder,decoder}"""
    registered_encoder = False
    registered_decoder = False

    # Primary: known path from backbone tree inspection
    lm = getattr(backbone, "lm", None)
    if lm is not None:
        language_model = getattr(lm, "language_model", None)
        if language_model is not None:
            enc = getattr(language_model, "encoder", None)
            dec = getattr(language_model, "decoder", None)
            if enc is not None and isinstance(enc, nn.Module):
                enc.register_forward_hook(lambda _m, _inp, out: on_encoder(out))
                registered_encoder = True
            if dec is not None and isinstance(dec, nn.Module):
                dec.register_forward_hook(lambda _m, _inp, out: on_decoder(out))
                registered_decoder = True

    # Fallback: walk all T5Stack modules
    if not registered_encoder or not registered_decoder:
        for name, mod in backbone.named_modules():
            if type(mod).__name__ != "T5Stack":
                continue
            is_decoder = getattr(getattr(mod, "config", None), "is_decoder", None)
            if is_decoder is False and not registered_encoder:
                mod.register_forward_hook(lambda _m, _inp, out: on_encoder(out))
                registered_encoder = True
            elif is_decoder is True and not registered_decoder:
                mod.register_forward_hook(lambda _m, _inp, out: on_decoder(out))
                registered_decoder = True
            if registered_encoder and registered_decoder:
                break

    if not registered_encoder or not registered_decoder:
        logger.warning("Hidden state hooks NOT registered (enc=%s, dec=%s)", registered_encoder, registered_decoder)

def load_motiongpt(config: dict[str, Any]) -> nn.Module:
    """Load the frozen MotionGPT backbone from vendor/MotionGPT/.

    Note: vendor files are not shipped in unit tests; this is a runtime path.
    """

    motiongpt_dir = str(config["model"]["motiongpt_dir"])
    if not motiongpt_dir:
        raise ValueError("config.model.motiongpt_dir is required")

    # Add vendor path for imports.
    if motiongpt_dir not in sys.path:
        sys.path.insert(0, motiongpt_dir)

    # MotionGPT is vendored as a repo with top-level module `mGPT/`, not a `MotionGPT` python package.
    try:
        from omegaconf import OmegaConf  # type: ignore[import-not-found]
        from mGPT.data.build_data import build_data  # type: ignore[import-not-found]
        from mGPT.models.build_model import build_model  # type: ignore[import-not-found]
    except Exception as e:  # noqa: BLE001
        raise FileNotFoundError(
            "MotionGPT vendor dependencies not importable. "
            "Ensure `vendor/MotionGPT` exists and its requirements were installed."
        ) from e

    vendor_root = Path(motiongpt_dir)
    vendor_cfg_assets = vendor_root / "configs" / "assets.yaml"
    vendor_cfg_exp = vendor_root / "configs" / "config_h3d_stage3.yaml"
    if not vendor_cfg_assets.exists():
        raise FileNotFoundError(f"Missing MotionGPT assets config at {vendor_cfg_assets}")
    if not vendor_cfg_exp.exists():
        raise FileNotFoundError(f"Missing MotionGPT experiment config at {vendor_cfg_exp}")

    # Replicate MotionGPT's `parse_args()` merge logic so interpolations like
    # `${evaluator.tm2t}` resolve correctly.
    cfg_assets = OmegaConf.load(str(vendor_cfg_assets))
    config_folder = (vendor_root / str(cfg_assets.CONFIG_FOLDER)).resolve()
    cfg_base = OmegaConf.load(str(config_folder / "default.yaml"))
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(str(vendor_cfg_exp)))
    if not bool(cfg_exp.get("FULL_CONFIG", False)):
        import glob

        for yaml_path in glob.glob(str(config_folder / "*" / "*.yaml")):
            rel = Path(yaml_path).relative_to(config_folder)
            nodes = str(rel).replace(".yaml", "").replace("/", ".")
            OmegaConf.update(cfg_exp, nodes, OmegaConf.load(yaml_path))
    cfg = OmegaConf.merge(cfg_exp, cfg_assets)

    # Some MotionGPT configs (or local edits) may inline/hardcode lm/motion_vae blocks
    # and break interpolations that expect `vq.default` / `lm.default`.
    # Ensure these exist and `codebook_size` can be resolved.
    try:
        if "vq" in cfg and "default" in cfg.vq and "motion_vae" in cfg.model.params:
            mv = cfg.model.params.motion_vae
            if not (isinstance(mv, dict) or hasattr(mv, "get")):
                pass
            # If missing the expected `params.code_num`, restore from vq.default.
            if not (hasattr(mv, "params") and hasattr(mv.params, "code_num")):
                cfg.model.params.motion_vae = cfg.vq.default
        if "lm" in cfg and "default" in cfg.lm and "lm" in cfg.model.params:
            lm = cfg.model.params.lm
            if not (hasattr(lm, "params") and (hasattr(lm.params, "model_path") or hasattr(lm.params, "model_type"))):
                cfg.model.params.lm = cfg.lm.default
        # Final fallback for codebook_size.
        if not hasattr(cfg.model.params, "codebook_size"):
            cfg.model.params.codebook_size = 512

        # MotionGPT's lm.default sometimes points to a local path (e.g. ../memData/deps/flan-t5-base).
        # If that path doesn't exist, replace with a valid HF model id so Transformers can download it.
        hf_flan = "google/flan-t5-base"
        if "lm" in cfg and "default" in cfg.lm and hasattr(cfg.lm.default, "params") and hasattr(cfg.lm.default.params, "model_path"):
            model_path = str(cfg.lm.default.params.model_path)
            if model_path and (model_path.startswith("..") or model_path.startswith("/") or model_path.startswith("~")):
                if not Path(model_path).expanduser().exists():
                    cfg.lm.default.params.model_path = hf_flan

        if hasattr(cfg, "model") and hasattr(cfg.model, "params") and hasattr(cfg.model.params, "lm"):
            lm_cfg = cfg.model.params.lm
            if hasattr(lm_cfg, "params") and hasattr(lm_cfg.params, "model_path"):
                model_path = str(lm_cfg.params.model_path)
                if model_path and (model_path.startswith("..") or model_path.startswith("/") or model_path.startswith("~")):
                    if not Path(model_path).expanduser().exists():
                        lm_cfg.params.model_path = hf_flan
    except Exception:
        # Best-effort; MotionGPT config structure is not stable across versions.
        pass


    # Point MotionGPT at the checkpoint tar.
    ckpt_setting = str(config["model"].get("motiongpt_ckpt", ""))
    if not ckpt_setting:
        raise ValueError("config.model.motiongpt_ckpt is required")
    ckpt_path = (vendor_root / ckpt_setting).resolve()
    if ckpt_path.is_dir():
        tars = sorted(ckpt_path.glob("*.tar"))
        if not tars:
            raise FileNotFoundError(f"No .tar checkpoints found in {ckpt_path}")
        ckpt_path = tars[0]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"MotionGPT checkpoint not found at {ckpt_path}")

    cfg.TEST.CHECKPOINTS = str(ckpt_path)
    cfg.TRAIN.PRETRAINED_VAE = str(ckpt_path)

    # Inject DATASET.HUMANML3D sub-config that HumanML3DDataModule.__init__ requires.
    # We only use MotionGPT as a backbone (no actual data loading), but the datamodule
    # still expects the *normalization stats* under deps/t2m/.../meta/{mean,std}.npy.
    _deps_root = vendor_root / "deps"
    # After `prepare/download_t2m_evaluators.sh`, stats live under:
    #   deps/t2m/t2m/t2m/<checkpoint>/meta/{mean,std}.npy
    # MotionGPT constructs:
    #   MEAN_STD_PATH / t2m / <checkpoint> / meta / mean.npy
    # so MEAN_STD_PATH must be deps/t2m/t2m.
    _mean_std_root = _deps_root / "t2m" / "t2m"
    _dummy_data_root = str(ckpt_path.parent)
    humanml3d_patch = OmegaConf.create({
        "DEBUG": False,
        "DATASET": {
            "HUMANML3D": {
                "ROOT": _dummy_data_root,
                "MEAN_STD_PATH": str(_mean_std_root),
                "MAX_MOTION_LEN": 196,
                "MIN_MOTION_LEN": 40,
                "MAX_TEXT_LEN": 20,
                "UNIT_LEN": 4,
                "STD_TEXT": False,
            },
            "WORD_VERTILIZER_PATH": str(_deps_root / "glove"),
            "TASK_PATH": str(vendor_root / "prepare" / "instructions"),
            "CODE_PATH": "TOKENS",
        }
    })
    cfg = OmegaConf.merge(cfg, humanml3d_patch)

    # MotionGPT metrics expect TM2T checkpoints under:
    #   cfg.METRIC.TM2T.t2m_path / "t2m" / "text_mot_match/model/finest.tar"
    # With `prepare/download_t2m_evaluators.sh`, that resolves correctly when
    # t2m_path is set to deps/t2m/t2m.
    metric_patch = OmegaConf.create({
        "METRIC": {"TM2T": {"t2m_path": str(_mean_std_root)}},
    })
    cfg = OmegaConf.merge(cfg, metric_patch)

    # MotionGPT imports spaCy unconditionally in some dataset modules. We do not use
    # those datasets, so provide a lightweight stub if spaCy is not installed.
    try:
        import spacy as _spacy  # noqa: F401
    except ModuleNotFoundError:
        import types

        _spacy_stub = types.ModuleType("spacy")

        def _spacy_load(_name: str):  # noqa: ANN001
            return object()

        _spacy_stub.load = _spacy_load  # type: ignore[attr-defined]
        sys.modules["spacy"] = _spacy_stub

    # Monkey-patch get_sample_set so HumanML3DDataModule.__init__ never
    # tries to read actual motion/text files from disk. We only need the
    # datamodule object to pass metadata (nfeats) to build_model.
    try:
        from mGPT.data import BASEDataModule as _BASE  # type: ignore[import-not-found]
        import types

        def _stub_get_sample_set(self, overrides=None):  # noqa: ANN001
            class _FakeSampleSet:
                nfeats = 263  # HumanML3D joint feature dimensionality
            return _FakeSampleSet()

        _BASE.get_sample_set = types.MethodType(_stub_get_sample_set, _BASE) if False else _stub_get_sample_set  # noqa: SIM210
        # Patch at class level so all instances use it.
        _BASE.get_sample_set = _stub_get_sample_set
    except Exception:  # noqa: BLE001
        pass

    # Monkey-patch WordVectorizer to avoid requiring GloVe assets in deps/glove/.
    # We don't need word vectors since we never use the datamodule for training/eval.
    try:
        import mGPT.data.HumanML3D as _H3D  # type: ignore[import-not-found]

        class _StubWordVectorizer:  # noqa: D401
            """Minimal stub for MotionGPT datamodule init."""

            def __init__(self, meta_root: str, prefix: str):  # noqa: ANN001
                self.word2vec = {"unk": 0.0}

            def __len__(self) -> int:
                return 1

        _H3D.WordVectorizer = _StubWordVectorizer  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    # Device selection (MotionGPT supports cpu/gpu/mps).
    if torch.cuda.is_available():
        cfg.ACCELERATOR = "gpu"
        cfg.DEVICE = [0]
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        cfg.ACCELERATOR = "mps"
        cfg.DEVICE = [0]
    else:
        cfg.ACCELERATOR = "cpu"
        cfg.DEVICE = [0]

    datamodule = build_data(cfg, phase="test")
    backbone = build_model(cfg, datamodule)

    def _format_module_tree(module: nn.Module, prefix: str = "", max_depth: int = 4, depth: int = 0) -> list[str]:
        if depth > max_depth:
            return []
        lines: list[str] = []
        for name, child in module.named_children():
            lines.append(f"{prefix}{name}: {type(child).__name__}")
            lines.extend(_format_module_tree(child, prefix + "  ", max_depth, depth + 1))
        return lines

    logger.info("=== BACKBONE TREE ===\n%s\n=== END BACKBONE TREE ===", "\n".join(_format_module_tree(backbone)))

    # Load weights (state_dict stored under "state_dict" in their checkpoints).
    try:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TimeoutError as e:
        raise RuntimeError(
            "Timed out while reading the MotionGPT checkpoint file. "
            f"The file at {ckpt_path} exists but could not be read reliably. "
            "This commonly happens when the checkpoint lives on a cloud-synced folder "
            "(OneDrive/iCloud/Dropbox) and is still 'online-only' or otherwise stalled. "
            "Fix: ensure the checkpoint is fully downloaded ('Always Keep on This Device') "
            "or copy it to a local (non-synced) path and point `config.model.motiongpt_ckpt` "
            "to that local directory."
        ) from e
    except (EOFError, RuntimeError, OSError) as e:
        raise RuntimeError(
            "Failed to load MotionGPT checkpoint. "
            f"The file at {ckpt_path} appears corrupted or truncated. "
            "Re-download/re-copy the MotionGPT checkpoint tar into "
            "`vendor/MotionGPT/checkpoints/MotionGPT-base/` (or update "
            "`config.model.motiongpt_ckpt` to point at a valid checkpoint)."
        ) from e
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    # MotionGPT overrides `load_state_dict` and may expect evaluator/metrics to exist.
    # We only need backbone weights, so bypass custom logic.
    nn.Module.load_state_dict(backbone, state_dict, strict=False)

    _freeze_module(backbone)
    return backbone


class MuscleMAPModel(nn.Module):
    """Top-level model: text -> (logits, log_T, motion_output)."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        activation_head: ActivationHead | None = None,
        length_predictor: LengthPredictor | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.backbone = backbone
        _freeze_module(self.backbone)

        self.length_predictor = length_predictor if length_predictor is not None else LengthPredictor()
        self.activation_head = activation_head if activation_head is not None else ActivationHead()

        self._svd_done = False
        self._cached_encoder_hidden: Tensor | None = None
        self._cached_decoder_hidden: Tensor | None = None

        _try_register_hidden_hooks(
            self.backbone,
            on_encoder=self._cache_encoder_hidden,
            on_decoder=self._cache_decoder_hidden,
        )

    def _cache_encoder_hidden(self, out: Any) -> None:
        if isinstance(out, Tensor):
            self._cached_encoder_hidden = out
            return
        if isinstance(out, (tuple, list)) and out and isinstance(out[0], Tensor):
            self._cached_encoder_hidden = out[0]
            return
        if isinstance(out, dict) and isinstance(out.get("last_hidden_state", None), Tensor):
            self._cached_encoder_hidden = out["last_hidden_state"]
            return
        lhs = getattr(out, "last_hidden_state", None)
        if isinstance(lhs, Tensor):
            self._cached_encoder_hidden = lhs
            return

    def _cache_decoder_hidden(self, out: Any) -> None:
        if isinstance(out, Tensor):
            self._cached_decoder_hidden = out
            return
        if isinstance(out, (tuple, list)) and out and isinstance(out[0], Tensor):
            self._cached_decoder_hidden = out[0]
            return
        if isinstance(out, dict) and isinstance(out.get("last_hidden_state", None), Tensor):
            self._cached_decoder_hidden = out["last_hidden_state"]
            return
        lhs = getattr(out, "last_hidden_state", None)
        if isinstance(lhs, Tensor):
            self._cached_decoder_hidden = lhs
            return
    def _maybe_svd_warm_start(self) -> None:
        if self._svd_done:
            return

        def _get_lm_weight() -> Tensor | None:
            # Unwrap DDP on backbone if needed
            backbone = self.backbone.module if hasattr(self.backbone, "module") else self.backbone

            # Preferred: backbone.lm_head.weight (T5ForConditionalGeneration style)
            if hasattr(backbone, "lm_head") and hasattr(backbone.lm_head, "weight"):
                w = backbone.lm_head.weight
                if isinstance(w, Tensor):
                    return w

            # MotionGPT (vendor) nesting: backbone.lm.language_model.lm_head.weight
            lm = getattr(backbone, "lm", None)
            if lm is not None:
                language_model = getattr(lm, "language_model", None)
                if language_model is not None:
                    if hasattr(language_model, "lm_head") and hasattr(language_model.lm_head, "weight"):
                        w = language_model.lm_head.weight
                        if isinstance(w, Tensor):
                            return w
                    if hasattr(language_model, "get_output_embeddings"):
                        emb = language_model.get_output_embeddings()
                        if emb is not None and hasattr(emb, "weight") and isinstance(emb.weight, Tensor):
                            return emb.weight

            return None

        # Unwrap DDP on activation_head if needed
        act_head = self.activation_head.module if hasattr(self.activation_head, "module") else self.activation_head

        W_lm_param = _get_lm_weight()
        if W_lm_param is None:
            logger.warning("SVD warm-start skipped: could not locate backbone LM head weight.")
            self._svd_done = True
            return

        W_lm = W_lm_param.data  # [vocab_size, hidden]
        if W_lm.ndim != 2 or W_lm.shape[1] != act_head.input_proj.in_features:
            raise ValueError(f"Unexpected lm_head weight shape: {tuple(W_lm.shape)}")
        _, _, Vt = torch.linalg.svd(W_lm, full_matrices=False)
        with torch.no_grad():
            act_head.input_proj.weight.copy_(Vt[: act_head.input_proj.out_features, :])
        self._svd_done = True

    def forward(
        self,
        text_tokens: Any,
        motion_tokens: Any | None = None,
        lengths: list[int] | None = None,
        *,
        T_frame: int | None = None,
    ) -> tuple[Tensor, Tensor, Any]:

        self._maybe_svd_warm_start()
        self._cached_encoder_hidden = None
        self._cached_decoder_hidden = None

        # Determine batch size for fallback lengths
        if isinstance(text_tokens, dict):
            B = text_tokens["input_ids"].shape[0]
        elif isinstance(text_tokens, list):
            B = len(text_tokens)
        else:
            B = text_tokens.shape[0]
        if lengths is None:
            lengths = [196] * B  # HumanML3D max as safe fallback

        # Always use teacher-forcing forward() — never generate() during training.
        # generate() is autoregressive and incompatible with DDP synchronization.
        batch = {"text": text_tokens, "length": lengths}
        if motion_tokens is not None:
            batch["motion_tokens"] = motion_tokens
        motion_output = self.backbone(batch)

        encoder_hidden = self._cached_encoder_hidden
        decoder_hidden = self._cached_decoder_hidden

        # If the MotionGPT wrapper uses autoregressive generation (and thus never
        # calls T5 encoder/decoder forward), fall back to a direct teacher-forced
        # pass through the underlying HF T5 model to obtain hidden states.
        if encoder_hidden is None or decoder_hidden is None:
            lm = getattr(self.backbone, "lm", None)
            language_model = getattr(lm, "language_model", None) if lm is not None else None
            tokenizer = getattr(lm, "tokenizer", None) if lm is not None else None
            if language_model is not None and tokenizer is not None and isinstance(text_tokens, list):
                device = next(self.activation_head.parameters()).device
                enc = tokenizer(
                    text_tokens,
                    padding="max_length",
                    max_length=int(getattr(lm, "max_length", 256)),
                    truncation=True,
                    return_attention_mask=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                start_id = int(
                    getattr(getattr(language_model, "config", None), "decoder_start_token_id", None)
                    or getattr(getattr(language_model, "config", None), "pad_token_id", 0)
                )
                decoder_input_ids = torch.full(
                    (input_ids.shape[0], 1),
                    fill_value=start_id,
                    device=device,
                    dtype=input_ids.dtype,
                )
                out = language_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
                encoder_hidden = getattr(out, "encoder_last_hidden_state", None)
                decoder_hidden = getattr(out, "decoder_last_hidden_state", None)

        # Allow mocked backbones to return hidden states directly.
        if encoder_hidden is None and isinstance(motion_output, dict) and "encoder_hidden" in motion_output:
            encoder_hidden = motion_output["encoder_hidden"]
        if decoder_hidden is None and isinstance(motion_output, dict) and "decoder_hidden" in motion_output:
            decoder_hidden = motion_output["decoder_hidden"]

        if encoder_hidden is None or decoder_hidden is None:
            raise RuntimeError("Missing encoder/decoder hidden states (hooks not triggered?)")

        pred_log_T = self.length_predictor(encoder_hidden)  # [B, 1]

        # Pick one common T_frame for the whole batch.
        # During training we usually override this to match the padded target length.
        if T_frame is None:
            if self.config is not None:
                lp_cfg = self.config.get("model", {}).get("length_predictor", {})
                min_T = int(lp_cfg.get("min_T", 30))
                max_T = int(lp_cfg.get("max_T", 256))
            else:
                min_T, max_T = 30, 256
            pred_T = torch.exp(pred_log_T).round().clamp(min=float(min_T), max=float(max_T)).to(dtype=torch.int64)
            T_frame = int(pred_T.max().item())

        logits = self.activation_head(decoder_hidden, T_frame=T_frame)  # [B, T_frame, 80]
        return logits, pred_log_T, motion_output

    def parameters_to_train(self):
        """Return trainable parameters (activation head, length predictor, optional LoRA).

        Backbone weights stay frozen except LoRA adapters injected by peft.
        """
        params = list(self.activation_head.parameters())
        params += list(self.length_predictor.parameters())
        params += [p for p in self.backbone.parameters() if p.requires_grad]
        return params
    
    def apply_lora(self, config: dict[str, Any]) -> None:
        """Apply LoRA adapters to the T5 decoder (stage 2)."""

        try:
            from peft import LoraConfig, get_peft_model  # type: ignore[import-not-found]
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("peft is required for apply_lora") from e

        lora_cfg = config.get("training", {})
        r = int(lora_cfg.get("lora_r", 8))
        alpha = int(lora_cfg.get("lora_alpha", 16))
        dropout = float(lora_cfg.get("lora_dropout", 0.05))
        target_modules = list(lora_cfg.get("lora_target_modules", ["q", "v"]))

        decoder: nn.Module | None = None
        if hasattr(self.backbone, "t5") and hasattr(self.backbone.t5, "decoder"):
            decoder = self.backbone.t5.decoder
        elif hasattr(self.backbone, "decoder"):
            decoder = getattr(self.backbone, "decoder")

        if decoder is None or not isinstance(decoder, nn.Module):
            raise AttributeError("Could not locate T5 decoder module on backbone")

        peft_config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_modules)
        wrapped = get_peft_model(decoder, peft_config)

        # Ensure LoRA params are trainable.
        for p in wrapped.parameters():
            if p.requires_grad:
                continue
        if hasattr(self.backbone, "t5") and hasattr(self.backbone.t5, "decoder"):
            self.backbone.t5.decoder = wrapped
        else:
            setattr(self.backbone, "decoder", wrapped)

