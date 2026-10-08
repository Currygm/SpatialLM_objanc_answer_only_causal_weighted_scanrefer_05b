"""One shared Sonata pass for compressed Qwen tokens and fine superpoints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .backbone import CompressedVisualTokens


class ResidualSkipFusion(nn.Module):
    def __init__(self, skip_dim: int, deep_dim: int, output_dim: int) -> None:
        super().__init__()
        self.skip = nn.Sequential(nn.Linear(skip_dim, output_dim), nn.LayerNorm(output_dim))
        self.deep = nn.Sequential(nn.Linear(deep_dim, output_dim), nn.LayerNorm(output_dim))

    def forward(self, skip: torch.Tensor, deep: torch.Tensor) -> torch.Tensor:
        dtype = self.skip[0].weight.dtype
        return self.skip(skip.to(dtype)) + self.deep(deep.to(dtype))


class PointUnetFusion(nn.Module):
    version = "sonata_native_stride_unet_residual_v1"

    def __init__(self, channels: tuple[int, ...]) -> None:
        super().__init__()
        if len(channels) != 5 or int(channels[-1]) != 512:
            raise ValueError("Expected five Sonata stages ending at 512 channels")
        self.levels = nn.ModuleList(
            ResidualSkipFusion(int(channels[index]), 512, 512) for index in (3, 2, 1, 0)
        )

    def forward(self, features: list[torch.Tensor], inverses: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != 5 or len(inverses) != 4:
            raise ValueError("Point U-Net requires enc0..enc4 and four pooling inverses")
        fused = features[-1]
        for module, shallow, inverse in zip(self.levels, (3, 2, 1, 0), reversed(inverses)):
            inverse = inverse.to(fused.device, torch.long)
            if inverse.numel() != features[shallow].shape[0] or int(inverse.max()) >= fused.shape[0]:
                raise ValueError("Invalid Sonata pooling_inverse")
            fused = module(features[shallow], fused[inverse])
        return fused


def mean_pool(values: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    groups = groups.to(values.device, torch.long).reshape(-1)
    if groups.numel() != values.shape[0] or groups.numel() == 0 or int(groups.min()) != 0:
        raise ValueError("stage0_to_superpoint must align and start at zero")
    count = int(groups.max()) + 1
    if torch.unique(groups).numel() != count:
        raise ValueError("Superpoint IDs must be compact")
    output = values.new_zeros((count, values.shape[1]))
    output.index_add_(0, groups, values)
    sizes = torch.bincount(groups, minlength=count).to(values).clamp_min(1)
    return output / sizes[:, None]


@dataclass
class SharedPointFeatures:
    compressed: CompressedVisualTokens
    superpoint_hidden: torch.Tensor
    superpoint_xyz: torch.Tensor


class SharedSonataEncoder(nn.Module):
    """Keep enc4 and U-Net branches in the same autograd graph and Sonata pass."""

    def __init__(self, spatiallm: Any, point_unet: PointUnetFusion) -> None:
        super().__init__()
        object.__setattr__(self, "spatiallm", spatiallm)
        self.point_unet = point_unet

    def forward(
        self,
        point_cloud: torch.Tensor,
        stage0_to_superpoint: torch.Tensor,
        superpoint_xyz: torch.Tensor,
    ) -> SharedPointFeatures:
        from spatiallm.model.sonata_encoder import Point

        if point_cloud.ndim != 2 or point_cloud.shape[1] != 12:
            raise ValueError("point_cloud must be [N,12]")
        row = point_cloud[torch.isfinite(point_cloud).all(dim=1)]
        if row.shape[0] == 0:
            raise ValueError("Point cloud is empty")
        backbone = self.spatiallm.point_backbone
        device = next(backbone.parameters()).device
        row = row.to(device=device, dtype=torch.float32)
        point = Point({
            "coord": row[:, 3:6], "grid_coord": row[:, :3].long(), "feat": row[:, 3:].float(),
            "batch": torch.zeros(row.shape[0], device=device, dtype=torch.long),
        })
        stages = list(backbone.enc._modules.values())
        with torch.no_grad():
            point = backbone.embedding(point)
            point.serialization(order=backbone.order, shuffle_orders=backbone.shuffle_orders)
            point.sparsify()
            point = stages[0](point)
        features = [point.feat]
        inverses: list[torch.Tensor] = []
        for stage in stages[1:]:
            point = stage(point)
            features.append(point.feat)
            inverses.append(point["pooling_inverse"].long())
        enc4 = point["sparse_conv_feat"].features
        projector = self.spatiallm.point_proj
        projected = projector(enc4.to(next(projector.parameters()).dtype))
        compressed = CompressedVisualTokens(
            projected,
            point["coord"].float(),
            point["grid_coord"].float(),
        )
        stage0 = self.point_unet(features, inverses)
        pooled = mean_pool(stage0, stage0_to_superpoint)
        xyz = superpoint_xyz.to(device=device, dtype=torch.float32).reshape(-1, 3)
        if pooled.shape[0] != xyz.shape[0]:
            raise ValueError("Superpoint features and coordinates do not align")
        return SharedPointFeatures(compressed, pooled, xyz)

