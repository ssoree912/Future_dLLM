"""One place that knows how each diffusion-LM family differs.

The teacher extractor and the student trainer are identical for LLaDA and Dream
apart from a handful of things: which class loads the weights, what the mask
token is, what the layer count and hidden width are called on the config, and
how Dream's autoregressive logit shift is absorbed. Keeping that here means the
two scripts stay single-path, so a change to the label definition or the loss
cannot land on one backend and miss the other.

Backends are detected from the checkpoint's ``model_type`` rather than a flag,
so passing ``--model model/Dream-v0-Instruct-7B`` is all it takes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


@dataclass(frozen=True)
class Backend:
    """What the family-agnostic scripts need to know about one model family."""

    name: str
    mask_id: int
    n_layers: int
    hidden_dim: int
    # Total sequence length the checkpoint was trained for, for the warning in
    # the extractor. LLaDA calls it max_sequence_length, Dream reuses the Qwen2
    # max_position_embeddings.
    native_max_seq_len: int
    generate: Callable

    # Dream was adapted from an autoregressive Qwen2, so row r's logits predict
    # token r+1 and the decode has to read position r from row r-1 (Dream's own
    # generation_utils does the same one-line shift). LLaDA is natively masked
    # and predicts in place.
    logit_shift: bool = False
    # A shifted backend cannot read its block's first token from the block's own
    # rows: that token comes from the row *before* the block, which a block-only
    # forward does not have. Sparse-dLLM's answer, which this follows, is to
    # confirm that one token on the step-0 full-sequence forward, where the row
    # is available and correct. From step 2 on the shift's meaningless first row
    # then lands on an already-confirmed position and is masked out.
    #
    # The alternative -- widening the window by a preceding anchor row -- also
    # works, but it takes one token per block out of the candidate pool and so
    # scores a different candidate set than the Sparse-dLLM baseline does. The
    # window is kept identical to the baseline's instead.
    seed_block_start: bool = False


def detect_family(model_path: str | Path) -> str:
    """``llada`` or ``dream``, from the checkpoint's config.json."""
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"no config.json under {model_path}")
    model_type = str(json.loads(config_path.read_text()).get("model_type", "")).lower()
    if "dream" in model_type:
        return "dream"
    if "llada" in model_type:
        return "llada"
    raise SystemExit(
        f"unsupported model_type {model_type!r} in {config_path}; "
        "future_dllm supports LLaDA and Dream checkpoints"
    )


def load_model(model_path: str | Path, *, max_seq_len: int, block_length: int,
               keep_ratio: float = 1.0, device_map: str = "auto") -> tuple[torch.nn.Module, Backend]:
    """Load a checkpoint with the eviction cache wired in, and describe it.

    ``block_len`` and ``keep_ratio`` are injected onto the config because that is
    what the patched attention reads at selection time, in both backends.
    """
    family = detect_family(model_path)

    if family == "dream":
        from .configuration_dream import DreamConfig
        from .modeling_dream import DreamModel
        from .dream_generate import generate as dream_generate

        # The vendored config class, not AutoConfig: the checkpoint's auto_map
        # points at the Hub's own modeling code, and a remote DreamConfig would
        # not be the class our patched DreamModel expects.
        cfg = DreamConfig.from_pretrained(model_path)
        native = int(getattr(cfg, "max_position_embeddings", max_seq_len))
        # The window is the block exactly, same as Sparse-dLLM: the queried rows
        # and the columns filter_cache removes are the same set.
        cfg.block_len, cfg.keep_ratio = block_length, keep_ratio
        cfg.use_cache = False
        model = DreamModel.from_pretrained(
            model_path, config=cfg, device_map=device_map,
            torch_dtype=torch.bfloat16).eval()
        backend = Backend(
            name="dream",
            mask_id=int(cfg.mask_token_id),
            n_layers=int(cfg.num_hidden_layers),
            hidden_dim=int(cfg.hidden_size),
            native_max_seq_len=native,
            generate=dream_generate,
            logit_shift=True,
            seed_block_start=True,
        )
        return model, backend

    from transformers import AutoConfig

    from .llada_generate import generate as llada_generate
    from .modeling_llada import LLaDAModelLM

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    native = int(getattr(cfg, "max_sequence_length", max_seq_len))
    cfg.max_sequence_length = max_seq_len
    cfg.block_len, cfg.keep_ratio = block_length, keep_ratio
    model = LLaDAModelLM.from_pretrained(
        model_path, config=cfg, device_map=device_map,
        torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    backend = Backend(
        name="llada",
        mask_id=126336,
        n_layers=int(cfg.n_layers),
        hidden_dim=int(cfg.d_model),
        native_max_seq_len=native,
        generate=llada_generate,
        logit_shift=False,
        seed_block_start=False,
    )
    return model, backend
