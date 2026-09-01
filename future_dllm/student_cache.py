"""Load and run block-conditioned prompt-utility checkpoints for cache selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass(frozen=True, slots=True)
class StudentConfig:
    layer_count: int = 32
    hidden_dim: int = 4096
    proj_dim: int = 256
    mlp_dim: int = 512
    heads: tuple[str, ...] = ("score",)


def _normalize_heads(heads: tuple[str, ...] | list[str] | str) -> tuple[str, ...]:
    if isinstance(heads, str):
        normalized = (heads,)
    else:
        normalized = tuple(heads)
    if not normalized or len(set(normalized)) != len(normalized):
        raise RuntimeError(f"invalid student heads: {normalized}")
    return normalized


def _build_score_head(config: StudentConfig) -> nn.Sequential:
    # [candidate ; current-block ; candidate * current-block]
    width = config.proj_dim * 3
    return nn.Sequential(
        nn.Linear(width, config.mlp_dim),
        nn.GELU(),
        nn.Linear(config.mlp_dim, 1),
    )


class PromptUtilityStudentLayer(nn.Module):
    def __init__(self, config: StudentConfig) -> None:
        super().__init__()
        self.heads = _normalize_heads(config.heads)
        self.token_proj = nn.Linear(config.hidden_dim, config.proj_dim)
        self.block_proj = nn.Linear(config.hidden_dim, config.proj_dim)
        if self.heads == ("score",):
            self.score_head = _build_score_head(config)
        else:
            self.score_heads = nn.ModuleDict(
                {head: _build_score_head(config) for head in self.heads}
            )

    def _head(self, head: str | None) -> nn.Module:
        if hasattr(self, "score_heads"):
            selected = head or self.heads[0]
            if selected not in self.score_heads:
                raise RuntimeError(
                    f"student head {selected!r} not found in {self.heads}"
                )
            return self.score_heads[selected]
        return self.score_head

    def forward(
        self,
        hidden_states: torch.Tensor,
        candidate_indices: torch.Tensor,
        head: str | None = None,
        block_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if block_indices is None:
            raise RuntimeError("block-conditioned student needs block_indices")
        token_proj = self.token_proj(
            hidden_states.index_select(dim=1, index=candidate_indices)
        )
        block_state = hidden_states.index_select(dim=1, index=block_indices).mean(dim=1)
        block_proj = self.block_proj(block_state).unsqueeze(1).expand(
            -1, token_proj.shape[1], -1
        )
        fused = torch.cat([token_proj, block_proj, token_proj * block_proj], dim=-1)
        return self._head(head)(fused).squeeze(-1)


class PromptUtilityStudent(nn.Module):
    def __init__(self, config: StudentConfig) -> None:
        super().__init__()
        heads = _normalize_heads(config.heads)
        self.config = StudentConfig(
            layer_count=config.layer_count,
            hidden_dim=config.hidden_dim,
            proj_dim=config.proj_dim,
            mlp_dim=config.mlp_dim,
            heads=heads,
        )
        self.layer_indices = tuple(range(self.config.layer_count))
        self.layers = nn.ModuleDict(
            {
                str(layer_id): PromptUtilityStudentLayer(self.config)
                for layer_id in self.layer_indices
            }
        )

    def forward_layer(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        candidate_indices: torch.Tensor,
        head: str | None = None,
        block_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.layers[str(layer_id)](
            hidden_states, candidate_indices, head=head, block_indices=block_indices,
        )

    def block_proj_norms(self) -> list[float]:
        """Per-layer Frobenius norm of the block-condition projection."""
        return [
            float(self.layers[str(i)].block_proj.weight.norm())
            for i in self.layer_indices
        ]


def load_prompt_utility_student(
    checkpoint_dir: str | Path, device: torch.device
) -> PromptUtilityStudent:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    config_path = checkpoint_dir / "config.json"
    state_path = checkpoint_dir / "pytorch_model.bin"
    if not config_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(
            f"invalid student checkpoint directory: {checkpoint_dir}"
        )
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_config["heads"] = tuple(raw_config.get("heads", ("score",)))
    selection_step = raw_config.pop(
        "cache_selection_step", raw_config.pop("cache_delay_steps", None)
    )
    if selection_step != 0:
        raise RuntimeError(
            "checkpoint was not trained for first-step cache selection"
        )
    legacy_cond = raw_config.pop("cond", None)
    if legacy_cond not in (None, "blk"):
        raise RuntimeError(
            f"unsupported student checkpoint cond={legacy_cond!r}; "
            "this code only loads block-conditioned checkpoints"
        )
    student = PromptUtilityStudent(StudentConfig(**raw_config))
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    student.load_state_dict(state, strict=True)
    student.to(device)
    student.eval()
    return student
