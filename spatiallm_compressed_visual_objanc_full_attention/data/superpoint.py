"""Reason3D-style superpoint segmentation for point-only PLY inputs.

Reason3D segments triangle meshes with a Felzenszwalb graph algorithm. SpatialLM
point clouds do not contain triangle faces, so this module builds a local k-NN
graph and estimates normals with PCA before applying the same component-threshold
and small-component merging rules.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SuperpointConfig:
    enabled: bool = False
    cache_dir: str = ".cache/superpoints"
    graph_k: int = 8
    normal_k: int = 20
    edge_radius_scale: float = 3.0
    k_thresh: float = 0.01
    min_vertices: int = 20
    version: str = "reason3d-knn-v1"
    require_cache: bool = False

    @classmethod
    def from_mapping(
        cls, value: Optional[Mapping[str, Any]] = None
    ) -> "SuperpointConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        valid = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: item for key, item in dict(value).items() if key in valid})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.graph_k < 1:
            raise ValueError("superpoint graph_k must be positive")
        if self.normal_k < 3:
            raise ValueError("superpoint normal_k must be at least 3")
        if self.edge_radius_scale <= 0:
            raise ValueError("superpoint edge_radius_scale must be positive")
        if self.k_thresh < 0:
            raise ValueError("superpoint k_thresh must be non-negative")
        if self.min_vertices < 1:
            raise ValueError("superpoint min_vertices must be positive")


class _UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)
        self.size = np.ones(size, dtype=np.int64)

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[item] != item:
            parent = int(self.parent[item])
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int) -> int:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return left_root

    def component_size(self, item: int) -> int:
        return int(self.size[self.find(item)])


def _as_coordinates(coordinates: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(coordinates, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError("coordinates must have shape [N, 3]")
    if not np.isfinite(result).all():
        raise ValueError("coordinates contain non-finite values")
    return result


def _estimate_normals(
    coordinates: np.ndarray, tree: cKDTree, normal_k: int
) -> np.ndarray:
    count = coordinates.shape[0]
    if count < 3:
        normals = np.zeros((count, 3), dtype=np.float32)
        if count:
            normals[:, 2] = 1.0
        return normals

    k = min(max(3, normal_k), count)
    _, neighbor_indices = tree.query(coordinates, k=k, workers=1)
    neighborhoods = coordinates[np.asarray(neighbor_indices)]
    centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered)
    covariance /= max(k - 1, 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0].astype(np.float32, copy=False)

    # PCA normals have arbitrary signs. Orient them deterministically relative to
    # the scene center, then use absolute dot products for edge similarity.
    radial = coordinates - coordinates.mean(axis=0, keepdims=True)
    orientation = np.einsum("ni,ni->n", normals, radial)
    flip = orientation < 0
    ambiguous = np.abs(orientation) <= 1e-8
    dominant_axis = np.argmax(np.abs(normals), axis=1)
    dominant_value = normals[np.arange(count), dominant_axis]
    flip |= ambiguous & (dominant_value < 0)
    normals[flip] *= -1
    return normals


def _build_edges(
    coordinates: np.ndarray,
    tree: cKDTree,
    normals: np.ndarray,
    graph_k: int,
    radius_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    count = coordinates.shape[0]
    if count <= 1:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float32)

    k = min(graph_k + 1, count)
    distances, neighbors = tree.query(coordinates, k=k, workers=1)
    distances = np.atleast_2d(distances)
    neighbors = np.atleast_2d(neighbors)
    if distances.shape[0] != count:
        distances = distances.T
        neighbors = neighbors.T

    nearest = distances[:, 1] if distances.shape[1] > 1 else distances[:, 0]
    positive_nearest = nearest[nearest > 0]
    if positive_nearest.size:
        max_edge_length = float(np.median(positive_nearest) * radius_scale)
    else:
        max_edge_length = float("inf")

    sources = np.repeat(np.arange(count, dtype=np.int64), k - 1)
    targets = neighbors[:, 1:k].reshape(-1).astype(np.int64, copy=False)
    edge_distances = distances[:, 1:k].reshape(-1)
    valid = (sources != targets) & (edge_distances <= max_edge_length)
    sources = sources[valid]
    targets = targets[valid]
    if sources.size == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float32)

    pairs = np.stack([np.minimum(sources, targets), np.maximum(sources, targets)], axis=1)
    pairs = np.unique(pairs, axis=0)
    cosine = np.abs(np.einsum("ni,ni->n", normals[pairs[:, 0]], normals[pairs[:, 1]]))
    weights = np.clip(1.0 - cosine, 0.0, 2.0).astype(np.float32)
    order = np.argsort(weights, kind="stable")
    return pairs[order], weights[order]


def segment_points(
    coordinates: np.ndarray,
    config: Optional[SuperpointConfig | Mapping[str, Any]] = None,
) -> np.ndarray:
    """Segment XYZ points into contiguous superpoint IDs."""
    config = SuperpointConfig.from_mapping(config)
    config.validate()
    coordinates = _as_coordinates(coordinates)
    count = coordinates.shape[0]
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if count == 1:
        return np.zeros(1, dtype=np.int64)

    tree = cKDTree(coordinates)
    normals = _estimate_normals(coordinates, tree, config.normal_k)
    edges, weights = _build_edges(
        coordinates, tree, normals, config.graph_k, config.edge_radius_scale
    )

    union_find = _UnionFind(count)
    thresholds = np.full(count, config.k_thresh, dtype=np.float64)
    for (left, right), weight in zip(edges, weights):
        left_root = union_find.find(int(left))
        right_root = union_find.find(int(right))
        if left_root == right_root:
            continue
        if weight <= thresholds[left_root] and weight <= thresholds[right_root]:
            root = union_find.union(left_root, right_root)
            thresholds[root] = float(weight) + config.k_thresh / union_find.component_size(root)

    for left, right in edges:
        left_root = union_find.find(int(left))
        right_root = union_find.find(int(right))
        if left_root == right_root:
            continue
        if (
            union_find.component_size(left_root) < config.min_vertices
            or union_find.component_size(right_root) < config.min_vertices
        ):
            union_find.union(left_root, right_root)

    roots = np.fromiter((union_find.find(index) for index in range(count)), dtype=np.int64)
    return np.unique(roots, return_inverse=True)[1].astype(np.int64, copy=False)


class CachedSuperpointSegmenter:
    """Generate and atomically cache deterministic raw-geometry superpoints."""

    def __init__(self, config: Optional[SuperpointConfig | Mapping[str, Any]] = None):
        self.config = SuperpointConfig.from_mapping(config)
        self.config.validate()
        self.cache_dir = Path(self.config.cache_dir).expanduser()

    def _metadata(self, coordinates: np.ndarray) -> dict[str, Any]:
        coordinate_bytes = np.ascontiguousarray(coordinates, dtype="<f4")
        digest = hashlib.blake2b(digest_size=20)
        digest.update(np.asarray(coordinate_bytes.shape, dtype="<i8").tobytes())
        digest.update(coordinate_bytes.tobytes())
        params = self.config.to_dict()
        params.pop("cache_dir", None)
        params.pop("enabled", None)
        params.pop("require_cache", None)
        digest.update(json.dumps(params, sort_keys=True).encode("utf-8"))
        return {
            "coordinate_digest": digest.hexdigest(),
            "point_count": int(coordinate_bytes.shape[0]),
            "parameters": params,
        }

    def cache_path(self, coordinates: np.ndarray) -> Path:
        metadata = self._metadata(_as_coordinates(coordinates))
        return self.cache_dir / f"{metadata['coordinate_digest']}.npz"

    def segment(self, coordinates: np.ndarray, source: Optional[str] = None) -> np.ndarray:
        coordinates = _as_coordinates(coordinates)
        metadata = self._metadata(coordinates)
        cache_path = self.cache_dir / f"{metadata['coordinate_digest']}.npz"
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                labels = cached["labels"].astype(np.int64, copy=False)
                cached_metadata = json.loads(str(cached["metadata"].item()))
            if cached_metadata == metadata and labels.shape == (coordinates.shape[0],):
                return labels
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

        if self.config.require_cache:
            raise FileNotFoundError(
                f"Missing or invalid superpoint cache for {source or 'point cloud'}: "
                f"{cache_path}. Run the Reason3D superpoint precompute script first."
            )

        labels = segment_points(coordinates, self.config)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(metadata)
        if source is not None:
            payload["source"] = os.fspath(source)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{cache_path.stem}.", suffix=".tmp", dir=self.cache_dir
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez_compressed(
                    handle,
                    labels=labels.astype(np.int32, copy=False),
                    metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                    source=np.asarray(payload.get("source", "")),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, cache_path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return labels
