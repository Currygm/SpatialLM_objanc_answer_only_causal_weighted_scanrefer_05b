"""Frozen native-stride Sonata bottleneck encoder."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch

@dataclass
class CompressedVisualTokens:
    hidden: torch.Tensor
    xyz: torch.Tensor
    grid_xyz: torch.Tensor

class FrozenCompressedSonata:
    """Run the original enc0-enc4 hierarchy and frozen base point projector."""

    def __init__(self, spatiallm_model: Any) -> None:
        self.model = spatiallm_model
        backbone_type = getattr(spatiallm_model.point_backbone_type, "value", None)
        if backbone_type != "sonata":
            raise NotImplementedError("Only the Sonata SpatialLM backbone is supported")
        strides = tuple(int(value) for value in spatiallm_model.config.point_config["stride"])
        if strides != (2, 2, 2, 2):
            raise ValueError(f"Expected native Sonata stride (2,2,2,2), got {strides}")
        if bool(spatiallm_model.config.point_config.get("enable_fourier_encode", True)):
            raise ValueError("This checkpoint route requires enable_fourier_encode=false")

    @torch.no_grad()
    def __call__(self, point_cloud: torch.Tensor) -> CompressedVisualTokens:
        from spatiallm.model.sonata_encoder import Point
        if point_cloud.ndim != 2 or point_cloud.shape[1] != 12:
            raise ValueError(f"point_cloud must be [N,12], got {tuple(point_cloud.shape)}")
        point_cloud = point_cloud[torch.isfinite(point_cloud).all(dim=1)]
        if not point_cloud.numel():
            raise ValueError("Point cloud is empty")
        backbone = self.model.point_backbone
        backbone.eval()
        self.model.point_proj.eval()
        device = next(backbone.parameters()).device
        row = point_cloud.to(device=device, dtype=torch.float32)
        grid = row[:, :3].long()
        features = row[:, 3:].float()
        point = Point({
            "coord": features[:, :3],
            "grid_coord": grid,
            "feat": features,
            "batch": torch.zeros(grid.shape[0], device=device, dtype=torch.long),
        })
        point = backbone.embedding(point)
        point.serialization(order=backbone.order, shuffle_orders=False)
        point.sparsify()
        for stage in backbone.enc._modules.values():
            point = stage(point)
        context = point["sparse_conv_feat"].features
        projector_dtype = next(self.model.point_proj.parameters()).dtype
        hidden = self.model.point_proj(context.to(dtype=projector_dtype))
        xyz = point["coord"].to(dtype=torch.float32)
        grid_xyz = point["grid_coord"].to(dtype=torch.float32)
        if hidden.shape[0] != xyz.shape[0]:
            raise RuntimeError("Bottleneck token and coordinate counts differ")
        return CompressedVisualTokens(hidden.detach(), xyz.detach(), grid_xyz.detach())
