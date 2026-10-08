"""Normal-cache contract for the official Sonata SpatialLM backbone."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


NORMAL_CACHE_VERSION = "scipy-kdtree-cuda-pca-radius0.1-maxnn30-v1"


@dataclass(frozen=True)
class NormalCacheConfig:
    enabled: bool = False
    cache_dir: str = ".cache/normals"
    require_cache: bool = True
    radius: float = 0.1
    max_nn: int = 30
    version: str = NORMAL_CACHE_VERSION

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "NormalCacheConfig":
        if value is None:
            return cls()
        valid = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: item for key, item in dict(value).items() if key in valid})

    def validate(self) -> None:
        if not self.enabled:
            raise ValueError("SpatialLMsegqwen3d_objref_fourmask_full requires normals")
        if self.radius <= 0:
            raise ValueError("normal radius must be positive")
        if self.max_nn < 3:
            raise ValueError("normal max_nn must be at least 3")
        if self.version != NORMAL_CACHE_VERSION:
            raise ValueError(f"Unsupported normal cache version: {self.version}")


def normal_cache_path(scene_path: Path, scene_root: Path, cache_dir: Path) -> Path:
    relative = scene_path.resolve(strict=True).relative_to(scene_root.resolve(strict=True))
    return (cache_dir / relative).with_suffix(".npy")


def load_scene_normals(
    scene_path: Path,
    scene_root: Path,
    config: NormalCacheConfig,
    expected_points: int,
) -> torch.Tensor:
    path = normal_cache_path(scene_path, scene_root, Path(config.cache_dir))
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing normal cache for {scene_path}: {path}. "
            "Run scripts/prepare_scanrefer_normals_v8.sh first."
        )
    normals = np.load(path, allow_pickle=False)
    if normals.shape != (expected_points, 3):
        raise ValueError(
            f"Normal cache shape mismatch for {scene_path}: "
            f"{normals.shape} != ({expected_points}, 3)"
        )
    normals = np.asarray(normals, dtype=np.float32)
    if not np.isfinite(normals).all():
        raise ValueError(f"Normal cache contains non-finite values: {path}")
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1e-6):
        raise ValueError(f"Normal cache contains zero-length normals: {path}")
    return torch.from_numpy(normals)


def validate_normal_manifest(
    config: NormalCacheConfig,
    scene_paths: Sequence[Path],
    scene_root: Path,
) -> None:
    if not config.require_cache:
        return
    manifest_path = Path(config.cache_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing normal cache manifest: {manifest_path}. "
            "Run scripts/prepare_scanrefer_normals_v8.sh first."
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected_parameters = {"radius": config.radius, "max_nn": config.max_nn}
    if (
        manifest.get("version") != config.version
        or manifest.get("parameters") != expected_parameters
    ):
        raise ValueError("Normal cache manifest parameters do not match the config")
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Invalid normal cache manifest: {manifest_path}")
    cached = {
        tuple(Path(row["scene_path"]).parts)
        for row in rows
        if isinstance(row, dict) and row.get("scene_path")
    }
    required = {
        tuple(path.resolve(strict=True).relative_to(scene_root.resolve(strict=True)).parts)
        for path in scene_paths
    }
    missing = sorted(required - cached)
    if missing:
        raise FileNotFoundError(
            f"Normal manifest is missing {len(missing)} scenes; "
            f"first missing={'/'.join(missing[0])}"
        )
