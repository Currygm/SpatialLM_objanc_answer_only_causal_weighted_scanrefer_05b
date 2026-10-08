"""LoRA adapters used only by the segmentation-stage Sonata encoder."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SonataLoraConfig:
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    stages: tuple[int, ...] = (1, 2, 3, 4)
    target_modules: tuple[str, ...] = ("qkv", "proj", "fc1", "fc2")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "SonataLoraConfig":
        result = cls(
            enabled=bool(value.get("enabled", True)),
            r=int(value.get("r", 16)),
            alpha=int(value.get("alpha", 32)),
            dropout=float(value.get("dropout", 0.05)),
            stages=tuple(int(x) for x in value.get("stages", (1, 2, 3, 4))),
            target_modules=tuple(str(x) for x in value.get("target_modules", ("qkv", "proj", "fc1", "fc2"))),
        )
        if result.r < 1 or result.alpha < 1 or not 0 <= result.dropout < 1:
            raise ValueError("Invalid Sonata LoRA rank/alpha/dropout")
        if set(result.stages) - {1, 2, 3, 4}:
            raise ValueError("Sonata LoRA stages must be drawn from 1..4")
        return result


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, config: SonataLoraConfig) -> None:
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scaling = float(config.alpha) / config.r
        self.dropout = nn.Dropout(config.dropout)
        self.sonata_lora_A = nn.Parameter(torch.empty(config.r, base.in_features))
        self.sonata_lora_B = nn.Parameter(torch.zeros(base.out_features, config.r))
        nn.init.kaiming_uniform_(self.sonata_lora_A, a=math.sqrt(5))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = F.linear(self.dropout(value), self.sonata_lora_A)
        update = F.linear(update, self.sonata_lora_B)
        return self.base(value) + update * self.scaling


def apply_sonata_lora(backbone: nn.Module, config: SonataLoraConfig) -> set[int]:
    if not config.enabled:
        return set()
    ids: set[int] = set()
    stages = list(backbone.enc._modules.values())
    for stage_index in config.stages:
        stage = stages[stage_index]
        for name, module in list(stage.named_modules()):
            if isinstance(module, LoRALinear):
                ids.update(id(p) for n, p in module.named_parameters() if n.startswith("sonata_lora_"))
                continue
            if not isinstance(module, nn.Linear) or name.rsplit(".", 1)[-1] not in config.target_modules:
                continue
            parent = stage
            parts = name.split(".")
            for part in parts[:-1]:
                parent = parent._modules[part]
            wrapped = LoRALinear(module, config)
            parent._modules[parts[-1]] = wrapped
            ids.update(id(p) for n, p in wrapped.named_parameters() if n.startswith("sonata_lora_"))
    if not ids:
        raise RuntimeError("Sonata LoRA did not match any encoder linear layer")
    return ids

