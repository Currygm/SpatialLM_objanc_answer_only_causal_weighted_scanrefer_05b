"""SpatialLMSeg 数据集与点云预处理模块。

本模块把 ReasonSeg/ScanRefer 风格 JSON 标注和场景点云文件转换为训练样本，负责授权路径校验、样本 ID 生成、目标掩码构造、SpatialLM 点云预处理以及 DataLoader batch 拼接。"""

from __future__ import annotations

import ast
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Sampler

from spatiallm_compressed_visual_objanc_full_attention.config import ensure_within, ensure_within_any, remap_renamed_data_path
from spatiallm_compressed_visual_objanc_full_attention.data.normals import (
    NormalCacheConfig,
    load_scene_normals,
    validate_normal_manifest,
)
from spatiallm_compressed_visual_objanc_full_attention.data.superpoint import CachedSuperpointSegmenter, SuperpointConfig


_REQUIRED_ANNOTATION_KEYS = (
    "scene_id",
    "object_id",
    "grounding_object_ids",
    "grounding_char_spans",
    "grounding_phrases",
)
_SCENE_ALIASES = {
    "xyz": ("xyz", "coord", "coords"),
    "rgb": ("rgb", "color", "colors"),
    "superpoint_id": ("superpoint_id", "superpoint", "superpoints"),
    "instance_id": ("instance_id", "instance_label", "target_id"),
    "semantic_id": ("semantic_id", "semantic_label"),
}


def _truthy(value: Any, *, default: bool = True) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() == "true"


def _literal_list(value: Any, *, field: str) -> list[Any]:
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid list in {field}: {value!r}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a list in {field}, got {type(parsed).__name__}")
    return parsed


def _phrase_spans(text: str, phrase: str) -> list[list[int]]:
    normalized = str(phrase).replace("_", " ").strip()
    if not normalized:
        return []
    pattern = re.compile(r"(?<!\w)" + re.escape(normalized) + r"(?!\w)", re.IGNORECASE)
    return [[match.start(), match.end()] for match in pattern.finditer(text)]


def _grounding_row(
    text: str,
    phrase: str,
) -> tuple[list[list[int]], list[str]] | None:
    spans = _phrase_spans(text, phrase)
    if not spans:
        return None
    return spans, [text[start:end] for start, end in spans]


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("annotations")
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON annotation list: {path}")
    return payload


def _load_training_source(
    source: dict[str, Any],
    *,
    authorized_root: Path,
    allowed_scenes: set[str] | None,
    additional_annotation_roots: Sequence[Path] = (),
) -> list[dict[str, Any]]:
    """Convert an existing annotation source to the common grounding schema."""
    path = ensure_within_any(
        remap_renamed_data_path(Path(str(source["annotations"])).expanduser()),
        [authorized_root, *additional_annotation_roots],
        must_exist=True,
    )
    source_format = str(source.get("format", "scanents_json"))
    dataset_name = str(source["name"])
    restrict = bool(source.get("restrict_to_primary_scenes", False))

    def scene_allowed(scene_id: str) -> bool:
        return not restrict or (allowed_scenes is not None and scene_id in allowed_scenes)

    annotations: list[dict[str, Any]] = []
    if source_format == "scanents_json":
        for item in _load_json_list(path):
            row = dict(item)
            if not scene_allowed(str(row["scene_id"])):
                continue
            row.setdefault("dataset", dataset_name)
            annotations.append(row)
        return annotations

    if source_format == "referit3d_csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = csv.DictReader(handle)
            for index, row in enumerate(rows):
                scene_id = str(row["scan_id"])
                if not scene_allowed(scene_id):
                    continue
                if bool(source.get("official_filter", True)) and not (
                    _truthy(row.get("is_train"))
                    and _truthy(row.get("correct_guess"))
                    and _truthy(row.get("mentions_target_class"))
                ):
                    continue
                text = str(row["utterance"]).strip()
                target_phrase = str(row["instance_type"]).replace("_", " ")
                target = _grounding_row(text, target_phrase)
                if target is None:
                    if bool(source.get("require_exact_target_phrase", True)):
                        continue
                    raise ValueError(
                        f"Cannot locate target phrase {target_phrase!r} in {text!r}"
                    )
                object_ids = [int(row["target_id"])]
                span_rows = [target[0]]
                phrase_rows = [target[1]]
                if dataset_name == "sr3d" and bool(source.get("include_anchors", True)):
                    anchor_ids = _literal_list(row.get("anchor_ids", "[]"), field="anchor_ids")
                    anchor_types = _literal_list(
                        row.get("anchors_types", "[]"), field="anchors_types"
                    )
                    if len(anchor_ids) != len(anchor_types):
                        raise ValueError("Sr3D anchor_ids and anchors_types have unequal lengths")
                    seen = set(object_ids)
                    for anchor_id, anchor_type in zip(anchor_ids, anchor_types):
                        anchor_id = int(anchor_id)
                        if anchor_id in seen:
                            continue
                        grounding = _grounding_row(text, str(anchor_type))
                        if grounding is None:
                            continue
                        seen.add(anchor_id)
                        object_ids.append(anchor_id)
                        span_rows.append(grounding[0])
                        phrase_rows.append(grounding[1])
                annotations.append(
                    {
                        "scene_id": scene_id,
                        "object_id": int(row["target_id"]),
                        "object_name": target_phrase,
                        "ann_id": f"{dataset_name}-{index}",
                        "description": text,
                        "original_description": text,
                        "dataset": dataset_name,
                        "grounding_object_ids": object_ids,
                        "grounding_char_spans": span_rows,
                        "grounding_phrases": phrase_rows,
                        "teacher_answer": row.get("teacher_answer"),
                    }
                )
        return annotations

    if source_format == "multi3drefer_json":
        for index, item in enumerate(_load_json_list(path)):
            scene_id = str(item["scene_id"])
            if not scene_allowed(scene_id):
                continue
            object_ids = [int(value) for value in item.get("object_ids", [])]
            if bool(source.get("single_target_only", True)) and len(object_ids) != 1:
                continue
            if len(object_ids) != 1:
                raise ValueError("The current single-target model only supports one Multi3DRefer target")
            text = str(item["description"]).strip()
            object_name = str(item["object_name"]).replace("_", " ")
            grounding = _grounding_row(text, object_name)
            if grounding is None:
                if bool(source.get("require_exact_target_phrase", True)):
                    continue
                raise ValueError(f"Cannot locate Multi3DRefer target {object_name!r}")
            annotations.append(
                {
                    "scene_id": scene_id,
                    "object_id": object_ids[0],
                    "object_name": object_name,
                    "ann_id": f"{dataset_name}-{index}",
                    "description": text,
                    "original_description": text,
                    "dataset": dataset_name,
                    "grounding_object_ids": object_ids,
                    "grounding_char_spans": [grounding[0]],
                    "grounding_phrases": [grounding[1]],
                    "teacher_answer": item.get("teacher_answer"),
                }
            )
        return annotations

    raise ValueError(f"Unsupported training annotation format: {source_format}")


class GroundingMixtureDataset(ConcatDataset):
    """Concatenated grounding sources with an explicit per-epoch sampling recipe."""

    def __init__(
        self,
        datasets: Sequence[Dataset],
        *,
        source_names: Sequence[str],
        source_probabilities: Sequence[float],
        epoch_size: int,
    ) -> None:
        super().__init__(datasets)
        if len(source_names) != len(datasets) or len(source_probabilities) != len(datasets):
            raise ValueError("Mixture source metadata must match the dataset count")
        probabilities = torch.tensor(source_probabilities, dtype=torch.double)
        if probabilities.numel() == 0 or not torch.isfinite(probabilities).all():
            raise ValueError("Mixture probabilities must be finite and non-empty")
        if bool((probabilities <= 0).any()):
            raise ValueError("Every mixture source must have positive sampling probability")
        self.source_names = tuple(map(str, source_names))
        self.source_probabilities = probabilities / probabilities.sum()
        self.epoch_size = int(epoch_size)
        if self.epoch_size < 1:
            raise ValueError("data.mixture_epoch_size must be positive")


class DistributedMixtureSampler(Sampler[int]):
    """DDP-safe source-balanced sampling with replacement and epoch reseeding."""

    def __init__(
        self,
        dataset: GroundingMixtureDataset,
        *,
        num_replicas: int,
        rank: int,
        seed: int,
    ) -> None:
        if num_replicas < 1 or rank < 0 or rank >= num_replicas:
            raise ValueError("Invalid distributed sampler rank/world size")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = math.ceil(dataset.epoch_size / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        source_ids = torch.multinomial(
            self.dataset.source_probabilities,
            self.total_size,
            replacement=True,
            generator=generator,
        )
        indices = torch.empty(self.total_size, dtype=torch.long)
        offset = 0
        for source_id, source_dataset in enumerate(self.dataset.datasets):
            positions = torch.nonzero(source_ids == source_id, as_tuple=False).flatten()
            count = int(positions.numel())
            if count:
                local = torch.randint(
                    len(source_dataset), (count,), generator=generator, dtype=torch.long
                )
                indices[positions] = local + offset
            offset += len(source_dataset)
        return iter(indices[self.rank : self.total_size : self.num_replicas].tolist())

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def scene_manifest_key(path: str | Path) -> tuple[str, ...]:
    """Stable scene identity that survives moving data into Reason3D_base/data."""
    parts = Path(path).parts
    for index, part in enumerate(parts):
        if part == "scannetv2" and index + 1 < len(parts):
            return tuple(parts[index + 1 :])
    return (Path(path).name,)


def annotation_sample_id(annotation: dict[str, Any], index: int) -> str:
    """为一条标注生成稳定样本 ID。

    ScanRefer 中同一个 ann_id 可能出现在不同场景或不同 object 上，因此样本 ID 同时使用 scene_id、object_id 和 ann_id；缺少 ann_id 时用标注下标兜底，保证错误日志和训练样本都能被唯一追踪。"""
    ann_id = str(annotation.get("ann_id", index))
    return f"{annotation['scene_id']}:{annotation['object_id']}:{ann_id}"


def compact_superpoints(
    superpoint_id: torch.Tensor,
    gt_point_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compact arbitrary scene-local superpoint IDs and optionally build majority GT."""
    source = superpoint_id.to(dtype=torch.long).reshape(-1)
    if source.numel() == 0:
        raise ValueError("superpoint_id must contain at least one point")
    values, point_to_superpoint = torch.unique(source, sorted=True, return_inverse=True)
    output = {
        "point_to_superpoint": point_to_superpoint.long(),
        "superpoint_values": values.long(),
    }
    if gt_point_mask is None:
        return output
    target = gt_point_mask.float()
    single_target = target.ndim == 1
    if single_target:
        target = target[None]
    if target.ndim != 2 or target.shape[1] != source.numel():
        raise ValueError(
            "gt_point_mask must have shape [points] or [objects,points]"
        )
    count = torch.bincount(point_to_superpoint, minlength=values.numel()).to(
        dtype=target.dtype, device=target.device
    )
    positive = torch.zeros(
        (target.shape[0], values.numel()), dtype=target.dtype, device=target.device
    )
    positive.index_add_(1, point_to_superpoint.to(target.device), target)
    masks = (positive / count[None] > 0.5).float()
    output["gt_superpoint_mask"] = masks[0] if single_target else masks
    return output


def superpoint_geometry(
    coord: torch.Tensor,
    grid_coord: torch.Tensor,
    point_to_superpoint: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Pool exact raw-point geometry for a compact superpoint partition."""
    coordinates = coord.reshape(-1, 3).float()
    grid = grid_coord.reshape(-1, 3).long()
    point_to = point_to_superpoint.reshape(-1).long()
    if coordinates.shape[0] == 0 or coordinates.shape[0] != point_to.numel():
        raise ValueError(
            "coord and point_to_superpoint must be non-empty and equally sized"
        )
    if grid.shape[0] != point_to.numel():
        raise ValueError(
            "grid_coord and point_to_superpoint must be equally sized"
        )
    if int(point_to.min()) != 0:
        raise ValueError("point_to_superpoint must start at zero")
    superpoint_count = int(point_to.max()) + 1
    if torch.unique(point_to).numel() != superpoint_count:
        raise ValueError("point_to_superpoint IDs must be contiguous")
    counts = torch.bincount(point_to, minlength=superpoint_count).to(
        device=coordinates.device, dtype=coordinates.dtype
    )
    pooled_coord = coordinates.new_zeros((superpoint_count, 3))
    pooled_coord.index_add_(0, point_to.to(coordinates.device), coordinates)
    pooled_grid = coordinates.new_zeros((superpoint_count, 3))
    pooled_grid.index_add_(
        0,
        point_to.to(coordinates.device),
        grid.to(device=coordinates.device, dtype=coordinates.dtype),
    )
    denominator = counts.unsqueeze(1).clamp_min(1.0)
    return {
        "superpoint_coord": pooled_coord / denominator,
        "superpoint_grid_coord": torch.floor(pooled_grid / denominator).long(),
    }


class ReasonSegDataset(Dataset):
    """ReasonSeg/ScanRefer 数据集实现。

    初始化阶段校验标注结构和配置字段，取样阶段读取场景点云、构造目标点掩码、生成问题与模板答案，并输出 SpatialLM 适配器和分割解码器需要的张量及元信息。"""

    def __init__(
        self,
        annotation_path: str | Path,
        scene_root: str | Path,
        *,
        authorized_root: str | Path,
        split: str,
        tuple_fields: Sequence[str],
        target_label_field: str,
        annotation_text_field: str = "question",
        scene_file_suffix: str = "reason",
        dataset_name: str = "unknown",
        include_types: Sequence[str] | None = None,
        num_bins: int = 1280,
        world_extent_m: float = 32.0,
        loc_margin_m: float = 0.30,
        random_voxel_sample: bool = False,
        seed: int = 42,
        segmentation_labels: bool = True,
        superpoint_config: dict[str, Any] | None = None,
        normal_config: dict[str, Any] | None = None,
        annotation_payload: list[dict[str, Any]] | None = None,
        require_teacher_answer: bool = False,
        annotation_authorized_roots: Sequence[str | Path] | None = None,
    ) -> None:
        """初始化数据集并校验输入配置。

        Args:
            annotation_path: JSON 标注文件路径。
            scene_root: 场景 .pth 文件根目录。
            authorized_root: 允许读取的数据根目录。
            split: 当前数据划分名称。
            tuple_fields: 场景 tuple 中各字段的顺序。
            target_label_field: 用于构造目标掩码的字段，支持 semantic_id 或 instance_id。
            annotation_text_field: 标注中作为问题或描述文本的字段名。
            scene_file_suffix: 场景文件名后缀，例如 reason 或 refer。
            dataset_name: 标注缺少 data_type 时使用的数据集名称。
            include_types: 可选的数据类型白名单。
            num_bins: SpatialLM 坐标量化 bin 数。
            world_extent_m: 坐标量化使用的世界范围。
            loc_margin_m: 目标框扩展边距。
            random_voxel_sample: 高层点超限时是否随机采样。
            seed: 随机采样种子。

        Raises:
            ValueError: 配置或字段非法。
            FileNotFoundError: 标注文件或场景根目录不存在。
            TypeError: 标注 JSON 结构不是预期列表。"""
        self.authorized_root = Path(authorized_root).resolve(strict=True)
        annotation_roots = (
            [Path(value).resolve(strict=True) for value in annotation_authorized_roots]
            if annotation_authorized_roots is not None
            else [self.authorized_root]
        )
        self.annotation_path = ensure_within_any(
            remap_renamed_data_path(Path(annotation_path).expanduser()),
            annotation_roots,
            must_exist=True,
        )
        self.scene_root = ensure_within(
            remap_renamed_data_path(Path(scene_root).expanduser()),
            self.authorized_root,
            must_exist=True,
        )
        self.split = split
        self.tuple_fields = tuple(tuple_fields)
        if len(set(self.tuple_fields)) != len(self.tuple_fields):
            raise ValueError(f"tuple_fields contains duplicate names: {self.tuple_fields}")
        self.target_label_field = str(target_label_field)
        if self.target_label_field not in {"semantic_id", "instance_id"}:
            raise ValueError(
                "target_label_field must be 'semantic_id' or 'instance_id', got "
                f"{self.target_label_field!r}"
            )
        self.annotation_text_field = str(annotation_text_field)
        if not self.annotation_text_field:
            raise ValueError("annotation_text_field must not be empty")
        self.scene_file_suffix = str(scene_file_suffix)
        if not self.scene_file_suffix.replace("_", "").isalnum():
            raise ValueError("scene_file_suffix must contain only letters, digits, or underscores")
        self.dataset_name = str(dataset_name)
        self.num_bins = int(num_bins)
        self.world_extent_m = float(world_extent_m)
        self.loc_margin_m = float(loc_margin_m)
        self.random_voxel_sample = bool(random_voxel_sample)
        self.seed = int(seed)
        self.segmentation_labels = bool(segmentation_labels)
        self.require_teacher_answer = bool(require_teacher_answer)
        self.superpoint_config = SuperpointConfig.from_mapping(superpoint_config)
        self.superpoint_config.validate()
        self.superpoint_segmenter = (
            CachedSuperpointSegmenter(self.superpoint_config)
            if self.superpoint_config.enabled
            else None
        )
        self.normal_config = NormalCacheConfig.from_mapping(normal_config)
        self.normal_config.validate()

        if annotation_payload is None:
            with self.annotation_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                payload = payload.get("annotations")
        else:
            payload = annotation_payload
        if include_types and isinstance(payload, list):
            allowed = set(include_types)
            payload = [
                item
                for item in payload
                if item.get("dataset", item.get("type", "unknown")) in allowed
            ]
        if not isinstance(payload, list) or not payload:
            raise ValueError("Annotation JSON must be a non-empty list or {'annotations': [...]}")
        self.annotations = payload
        self._validate_annotation_headers()
        self._validate_superpoint_manifest()
        self._validate_normal_manifest()

    def _unique_scene_paths(self) -> list[Path]:
        unique_annotations: dict[str, dict[str, Any]] = {}
        for annotation in self.annotations:
            key = annotation.get("scene_file") or annotation.get("sp_filename")
            key = str(key) if key else str(annotation["scene_id"])
            unique_annotations.setdefault(key, annotation)
        return [
            self._resolve_scene_path(annotation)
            for annotation in unique_annotations.values()
        ]

    def _validate_normal_manifest(self) -> None:
        validate_normal_manifest(
            self.normal_config, self._unique_scene_paths(), self.scene_root
        )

    def _validate_superpoint_manifest(self) -> None:
        """Fail before model loading when a required superpoint cache is incomplete."""
        if not (
            self.superpoint_config.enabled and self.superpoint_config.require_cache
        ):
            return
        manifest_path = Path(self.superpoint_config.cache_dir) / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing superpoint cache manifest: {manifest_path}. "
                "Run the Reason3D superpoint precompute script first."
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        rows = manifest.get("rows") if isinstance(manifest, dict) else None
        if not isinstance(rows, list):
            raise ValueError(f"Invalid superpoint cache manifest: {manifest_path}")
        expected_parameters = {
            "graph_k": self.superpoint_config.graph_k,
            "normal_k": self.superpoint_config.normal_k,
            "edge_radius_scale": self.superpoint_config.edge_radius_scale,
            "k_thresh": self.superpoint_config.k_thresh,
            "min_vertices": self.superpoint_config.min_vertices,
        }
        if (
            manifest.get("version") != self.superpoint_config.version
            or manifest.get("parameters") != expected_parameters
        ):
            raise ValueError("Superpoint cache manifest parameters do not match the config")
        cached = {
            scene_manifest_key(row["scene_path"])
            for row in rows
            if isinstance(row, dict) and row.get("scene_path")
        }
        required_paths = self._unique_scene_paths()
        required = {scene_manifest_key(path) for path in required_paths}
        missing = sorted(required - cached)
        if missing:
            raise FileNotFoundError(
                f"Superpoint manifest is missing {len(missing)} {self.split} scenes; "
                f"first missing={'/'.join(missing[0])}"
            )

    def _validate_annotation_headers(self) -> None:
        """校验标注必需字段和样本唯一性。

        每条标注必须包含 scene_id、object_id 以及配置的文本字段；同时使用 annotation_sample_id 检查 scene/object/ann 组合是否重复，既兼容 ScanRefer 的 ann_id 复用，又能阻止完全重复样本。"""
        seen: set[str] = set()
        for index, item in enumerate(self.annotations):
            if not isinstance(item, dict):
                raise TypeError(f"Annotation {index} is not a mapping")
            required = (*_REQUIRED_ANNOTATION_KEYS, self.annotation_text_field)
            missing = [key for key in required if key not in item]
            if self.require_teacher_answer and "teacher_answer" not in item:
                missing.append("teacher_answer")
            if missing:
                raise ValueError(f"Annotation {index} is missing {missing}")
            text = item[self.annotation_text_field]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Annotation {index} has empty text in {self.annotation_text_field}"
                )
            teacher_answer = item.get("teacher_answer")
            if teacher_answer is not None and (
                not isinstance(teacher_answer, str) or not teacher_answer.strip()
            ):
                raise ValueError(f"Annotation {index} has an empty teacher_answer")
            object_ids = item["grounding_object_ids"]
            spans_by_object = item["grounding_char_spans"]
            phrases_by_object = item["grounding_phrases"]
            if (
                not isinstance(object_ids, list)
                or not object_ids
                or len(object_ids) != len(spans_by_object)
                or len(object_ids) != len(phrases_by_object)
            ):
                raise ValueError(f"Annotation {index} has invalid grounding rows")
            if int(object_ids[0]) != int(item["object_id"]):
                raise ValueError(f"Annotation {index} target must occupy grounding row zero")
            if len(object_ids) > 50 or len(set(map(int, object_ids))) != len(object_ids):
                raise ValueError(f"Annotation {index} has invalid grounding object ids")
            for spans in spans_by_object:
                if not isinstance(spans, list) or not spans:
                    raise ValueError(f"Annotation {index} has an empty grounding span row")
                for span in spans:
                    if not isinstance(span, list) or len(span) != 2:
                        raise ValueError(f"Annotation {index} has an invalid grounding span")
                    start, end = int(span[0]), int(span[1])
                    if start < 0 or end <= start or end > len(text):
                        raise ValueError(f"Annotation {index} grounding span is outside its text")
            sample_id = annotation_sample_id(item, index)
            if sample_id in seen:
                raise ValueError(f"Duplicate annotation target: {sample_id}")
            seen.add(sample_id)

    def __len__(self) -> int:
        """返回过滤后的标注数量。"""
        return len(self.annotations)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """读取并转换单个训练样本。

        该方法解析场景文件，校验目标字段，生成原始点级二值掩码，执行 SpatialLM 点云预处理，并返回 question、answer、sample_id、inverse_map 等训练所需字段。"""
        annotation = self.annotations[index]
        scene_path = self._resolve_scene_path(annotation)
        scene = self._load_scene(scene_path)

        # tuple_fields 只解释磁盘中五个数组的真实含义；原始场景文件不被改写。
        xyz = _as_tensor(scene["xyz"], torch.float32)
        rgb = _normalize_rgb(_as_tensor(scene["rgb"], torch.float32))
        normals = load_scene_normals(
            scene_path,
            self.scene_root,
            self.normal_config,
            expected_points=xyz.shape[0],
        )
        if not self.segmentation_labels:
            self._validate_point_arrays(scene_path, xyz, rgb)
            voxel_seed = (
                int(np.random.randint(0, np.iinfo(np.uint32).max))
                if self.random_voxel_sample
                else self.seed + index
            )
            prepared = prepare_spatiallm_points(
                xyz,
                rgb,
                normals,
                None,
                num_bins=self.num_bins,
                world_extent_m=self.world_extent_m,
                random_sample=self.random_voxel_sample,
                generator=np.random.default_rng(voxel_seed),
            )
            return {
                "ann_id": str(annotation.get("ann_id", index)),
                "sample_id": annotation_sample_id(annotation, index),
                "scene_id": str(annotation["scene_id"]),
                "scene_path": str(scene_path),
                "object_id": int(annotation["object_id"]),
                "object_name": str(annotation.get("object_name", "unknown")),
                "question": str(annotation[self.annotation_text_field]),
                "teacher_answer": annotation.get("teacher_answer"),
                "data_type": str(annotation.get("dataset", self.dataset_name)),
                "target_char_spans": annotation["grounding_char_spans"][0],
                "target_phrases": list(annotation["grounding_phrases"][0]),
                "grounding_object_ids": list(map(int, annotation["grounding_object_ids"])),
                "grounding_char_spans": annotation["grounding_char_spans"],
                "grounding_phrases": annotation["grounding_phrases"],
                "original_description": str(
                    annotation.get("original_description", annotation[self.annotation_text_field])
                ),
                "point_cloud": prepared["point_cloud"],
            }
        # target_label_field 可选择语义类别或实例编号。
        target_labels = _as_tensor(scene[self.target_label_field], torch.long).reshape(-1)
        if self.superpoint_segmenter is not None:
            generated = self.superpoint_segmenter.segment(
                xyz.cpu().numpy(), source=str(scene_path)
            )
            superpoint_id = torch.from_numpy(generated).long()
        else:
            superpoint_id = _as_tensor(
                scene.get("superpoint_id", torch.arange(xyz.shape[0])), torch.long
            ).reshape(-1)
        extra_labels = {
            name: _as_tensor(scene[name], torch.long).reshape(-1)
            for name in ("semantic_id", "instance_id")
            if name in scene and name != self.target_label_field
        }
        self._validate_scene_arrays(
            scene_path, xyz, rgb, target_labels, superpoint_id, extra_labels
        )

        # Qwen3D supervision keeps the target in row zero and all present anchors.
        object_id = int(annotation["object_id"])
        available_ids = set(map(int, torch.unique(target_labels).tolist()))
        keep_rows = [
            row
            for row, value in enumerate(annotation["grounding_object_ids"])
            if int(value) in available_ids
        ]
        if not keep_rows or keep_rows[0] != 0:
            unique_preview = torch.unique(target_labels)[:20].tolist()
            raise ValueError(
                f"object_id={object_id} has no points in {scene_path}; "
                f"target_label_field={self.target_label_field}; "
                f"first target ids={unique_preview}"
            )
        grounding_object_ids = [
            int(annotation["grounding_object_ids"][row]) for row in keep_rows
        ]
        grounding_char_spans = [
            annotation["grounding_char_spans"][row] for row in keep_rows
        ]
        grounding_phrases = [
            annotation["grounding_phrases"][row] for row in keep_rows
        ]
        gt_point_masks = torch.stack(
            [target_labels.eq(value) for value in grounding_object_ids], dim=0
        ).float()
        gt_point_mask = gt_point_masks[0].bool()
        if not gt_point_mask.any():
            unique_preview = torch.unique(target_labels)[:20].tolist()
            raise ValueError(
                f"object_id={object_id} has no points in {scene_path}; "
                f"target_label_field={self.target_label_field}; "
                f"first target ids={unique_preview}"
            )

        target_xyz = xyz[gt_point_mask]
        bbox_min = target_xyz.amin(dim=0) - self.loc_margin_m
        bbox_max = target_xyz.amax(dim=0) + self.loc_margin_m
        gt_loc_mask = ((xyz >= bbox_min) & (xyz <= bbox_max)).all(dim=1)

        voxel_seed = (
            int(np.random.randint(0, np.iinfo(np.uint32).max))
            if self.random_voxel_sample
            else self.seed + index
        )
        generator = np.random.default_rng(voxel_seed)
        # 保存 inverse_map，decoder 输出才能恢复到原始 N 个点上计算损失。
        if self.superpoint_segmenter is not None:
            prepared = prepare_spatiallm_points(
                xyz,
                rgb,
                normals,
                gt_loc_mask,
                num_bins=self.num_bins,
                world_extent_m=self.world_extent_m,
                random_sample=self.random_voxel_sample,
                generator=generator,
                raw_superpoint_id=superpoint_id,
            )
            superpoint = compact_superpoints(
                prepared["point_to_superpoint"], gt_point_masks
            )
            if not torch.equal(
                superpoint["point_to_superpoint"],
                prepared["point_to_superpoint"],
            ):
                raise RuntimeError("Generated stage-0 superpoints are not compact")
            superpoint_values = prepared["superpoint_values"]
        else:
            # Legacy precomputed superpoints remain raw-point weighted.
            superpoint = compact_superpoints(superpoint_id, gt_point_masks)
            prepared = prepare_spatiallm_points(
                xyz,
                rgb,
                normals,
                gt_loc_mask,
                num_bins=self.num_bins,
                world_extent_m=self.world_extent_m,
                random_sample=self.random_voxel_sample,
                generator=generator,
                point_to_superpoint=superpoint["point_to_superpoint"],
            )
            superpoint_values = superpoint["superpoint_values"]

        return {
            "ann_id": str(annotation.get("ann_id", index)),
            "sample_id": annotation_sample_id(annotation, index),
            "scene_id": str(annotation["scene_id"]),
            "scene_path": str(scene_path),
            "object_id": object_id,
            "target_label_field": self.target_label_field,
            "object_name": str(annotation.get("object_name", "unknown")),
            "data_type": str(
                annotation.get(
                    "dataset", annotation.get("type", self.dataset_name)
                )
            ),
            "question": str(annotation[self.annotation_text_field]),
            "teacher_answer": annotation.get("teacher_answer"),
            "answer_entities": annotation.get("answer_entities", []),
            "answer_entity_indices": [annotation.get("answer_entity_indices", list(range(len(annotation["grounding_object_ids"]))))[row] for row in keep_rows],
            "grounding_roles": [annotation.get("grounding_roles", ["OBJ"] + ["ANC"] * (len(annotation["grounding_object_ids"]) - 1))[row] for row in keep_rows],
            "dropped_missing_entity_count": len(annotation["grounding_object_ids"]) - len(keep_rows),
            "target_char_spans": grounding_char_spans[0],
            "target_phrases": grounding_phrases[0],
            "grounding_object_ids": grounding_object_ids,
            "grounding_char_spans": grounding_char_spans,
            "grounding_phrases": grounding_phrases,
            "original_description": str(
                annotation.get("original_description", annotation[self.annotation_text_field])
            ),
            "point_cloud": prepared["point_cloud"],
            "high_xyz": prepared["high_xyz"],
            "inverse_map": prepared["inverse_map"],
            "gt_loc_high": prepared["gt_loc_high"],
            "gt_point_mask": gt_point_masks[0],
            "gt_point_masks": gt_point_masks,
            "superpoint_id": superpoint_id,
            "point_to_superpoint": superpoint["point_to_superpoint"],
            "superpoint_values": superpoint_values,
            "gt_superpoint_mask": superpoint["gt_superpoint_mask"][0],
            "gt_superpoint_masks": superpoint["gt_superpoint_mask"],
            "superpoint_coord": prepared["superpoint_coord"],
            "superpoint_grid_coord": prepared["superpoint_grid_coord"],
            "stage0_to_superpoint": prepared.get("stage0_to_superpoint"),
        }

    def _resolve_scene_path(self, annotation: dict[str, Any]) -> Path:
        """根据 scene_id 和 scene_file_suffix 定位场景文件。

        解析结果会再次经过授权根目录检查，防止异常 scene_id 或配置把读取范围带到授权目录之外。"""
        explicit = annotation.get("scene_file") or annotation.get("sp_filename")
        if explicit:
            candidate = remap_renamed_data_path(Path(str(explicit)).expanduser())
            if not candidate.is_absolute():
                candidate = self.scene_root / candidate
            return ensure_within(candidate, self.authorized_root, must_exist=True)

        scene_id = str(annotation["scene_id"])
        candidates = (
            self.scene_root / self.split / f"{scene_id}_{self.scene_file_suffix}.pth",
            self.scene_root / self.split / f"{scene_id}.pth",
            self.scene_root / f"{scene_id}_{self.scene_file_suffix}.pth",
            self.scene_root / f"{scene_id}.pth",
        )
        for candidate in candidates:
            if candidate.is_file():
                return ensure_within(candidate, self.authorized_root, must_exist=True)
        tried = "\n  - ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"No scene file for {scene_id}. Tried:\n  - {tried}")

    def _load_scene(self, path: Path) -> dict[str, Any]:
        # 可信的本地预处理场景只加载到 CPU，避免数据进程提前占用 GPU。
        """加载场景点云文件并按 tuple_fields 解包。

        函数只负责可信路径读取和字段映射；数组形状、点数一致性以及目标存在性由后续校验函数处理。"""
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict):
            scene: dict[str, Any] = {}
            for canonical, aliases in _SCENE_ALIASES.items():
                for alias in aliases:
                    if alias in raw:
                        scene[canonical] = raw[alias]
                        break
            required = ("xyz", "rgb")
            if self.segmentation_labels:
                required = (*required, self.target_label_field)
            missing = [key for key in required if key not in scene]
            if missing:
                raise ValueError(f"Scene dictionary {path} is missing {missing}")
            return scene
        if isinstance(raw, (tuple, list)):
            if len(raw) != len(self.tuple_fields):
                raise ValueError(
                    f"Tuple scene {path} has {len(raw)} arrays but tuple_fields has "
                    f"{len(self.tuple_fields)} entries"
                )
            scene = dict(zip(self.tuple_fields, raw))
            required = ("xyz", "rgb")
            if self.segmentation_labels:
                required = (*required, self.target_label_field)
            missing = [key for key in required if key not in scene]
            if missing:
                raise ValueError(f"tuple_fields does not define {missing}")
            return scene
        raise TypeError(f"Unsupported scene payload in {path}: {type(raw).__name__}")

    @staticmethod
    def _validate_point_arrays(
        path: Path,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
    ) -> None:
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must be [N,3] in {path}, got {tuple(xyz.shape)}")
        if rgb.shape != xyz.shape:
            raise ValueError(f"rgb shape {tuple(rgb.shape)} != xyz shape {tuple(xyz.shape)}")
        if xyz.shape[0] == 0:
            raise ValueError(f"Empty point cloud: {path}")
        if not torch.isfinite(xyz).all() or not torch.isfinite(rgb).all():
            raise ValueError(f"Non-finite coordinate or color value in {path}")

    @staticmethod
    def _validate_scene_arrays(
        path: Path,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        target_labels: torch.Tensor,
        superpoint_id: torch.Tensor,
        extra_labels: dict[str, torch.Tensor],
    ) -> None:
        """校验场景数组满足训练前提。

        检查 xyz/rgb 形状、目标字段长度以及指定 object_id 是否存在，失败时抛出包含场景和字段信息的异常，便于定位坏数据。"""
        ReasonSegDataset._validate_point_arrays(path, xyz, rgb)
        if target_labels.numel() != xyz.shape[0] or superpoint_id.numel() != xyz.shape[0]:
            raise ValueError(f"All point arrays must have the same N in {path}")
        for name, labels in extra_labels.items():
            if labels.numel() != xyz.shape[0]:
                raise ValueError(
                    f"{name} has {labels.numel()} values but xyz has {xyz.shape[0]} in {path}"
                )


def prepare_spatiallm_points(
    xyz: torch.Tensor,
    rgb: torch.Tensor,
    normals: torch.Tensor,
    gt_loc_mask: torch.Tensor | None,
    *,
    num_bins: int,
    world_extent_m: float,
    random_sample: bool,
    generator: np.random.Generator,
    point_to_superpoint: torch.Tensor | None = None,
    raw_superpoint_id: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """把原始点云转换为 SpatialLM 使用的高层点云。

    函数完成 RGB 归一化、点云体素化和 inverse_map 构造；如果高层点数超过限制，会同步采样高层点和 inverse_map，保证分割 logits 仍能映射回原始点监督。"""
    if point_to_superpoint is not None and raw_superpoint_id is not None:
        raise ValueError(
            "point_to_superpoint and raw_superpoint_id are mutually exclusive"
        )
    if normals.shape != xyz.shape:
        raise ValueError(
            f"normals shape {tuple(normals.shape)} != xyz shape {tuple(xyz.shape)}"
        )
    if not torch.isfinite(normals).all():
        raise ValueError("normals contain non-finite values")
    # 对齐 SpatialLM 预处理：坐标平移到正空间，再量化到固定三维网格。
    shifted = xyz - xyz.amin(dim=0, keepdim=True)
    grid_size = world_extent_m / num_bins
    grid = torch.floor(shifted / grid_size).long().clamp_(0, num_bins - 1)

    grid_np = grid.cpu().numpy()
    # inverse[i] 表示原始第 i 个点所属体素，是恢复原始分辨率的关键映射。
    _, inverse = np.unique(grid_np, axis=0, return_inverse=True)
    tie_breaker = generator.random(len(inverse)) if random_sample else np.arange(len(inverse))
    order = np.lexsort((tie_breaker, inverse))
    first = np.r_[True, inverse[order][1:] != inverse[order][:-1]]
    # 每个占用体素保留一个代表点；训练可随机选择，验证固定选择以保证可复现。
    selected = order[first]

    # np.unique assigns cluster ids in sorted order; selection is ordered by that same id.
    selected_tensor = torch.from_numpy(selected).long()
    inverse_tensor = torch.from_numpy(inverse).long()
    point_cloud = torch.cat(
        [
            grid[selected_tensor].float(),
            shifted[selected_tensor].float(),
            rgb[selected_tensor].float(),
            normals[selected_tensor].float(),
        ],
        dim=1,
    )
    output = {
        "point_cloud": point_cloud,
        "high_xyz": shifted[selected_tensor].float(),
        "inverse_map": inverse_tensor,
    }
    if raw_superpoint_id is not None:
        raw_labels = raw_superpoint_id.reshape(-1).long()
        if raw_labels.numel() != xyz.shape[0]:
            raise ValueError("raw_superpoint_id must match the raw point count")
        selected_labels = raw_labels[selected_tensor]
        values, stage0_to = torch.unique(
            selected_labels, sorted=True, return_inverse=True
        )
        raw_to = stage0_to[inverse_tensor]
        geometry = superpoint_geometry(
            shifted[selected_tensor], grid[selected_tensor], stage0_to
        )
        output.update(
            {
                "stage0_to_superpoint": stage0_to.long(),
                "point_to_superpoint": raw_to.long(),
                "superpoint_values": values.long(),
                **geometry,
            }
        )
    if point_to_superpoint is not None:
        output.update(
            superpoint_geometry(shifted, grid, point_to_superpoint)
        )
    if gt_loc_mask is not None:
        loc_per_voxel = np.zeros(int(inverse.max()) + 1, dtype=np.uint8)
        np.maximum.at(
            loc_per_voxel, inverse, gt_loc_mask.cpu().numpy().astype(np.uint8)
        )
        output["gt_loc_high"] = torch.from_numpy(loc_per_voxel).float()
    return output


def collate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """拼接 DataLoader batch。

    点云、mask 和 inverse_map 都是变长对象，因此保留为列表；文本和元信息也按列表返回，让模型逐样本计算分割损失。"""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    return {key: [sample[key] for sample in samples] for key in samples[0]}


def teacher_answer_missing_counts(config: dict[str, Any], split: str) -> dict[str, int]:
    """Inspect annotation text only and report missing generated Answer labels."""
    data = config["data"]
    root = Path(config["project"]["authorized_code_root"])
    artifact = Path(config["project"]["artifact_root"])
    if split == "train" and data.get("train_sources"):
        counts: dict[str, int] = {}
        primary_scenes: set[str] | None = None
        for source in data["train_sources"]:
            rows = _load_training_source(
                source,
                authorized_root=root,
                allowed_scenes=primary_scenes,
                additional_annotation_roots=(artifact,),
            )
            if bool(source.get("defines_primary_scenes", False)):
                primary_scenes = {str(row["scene_id"]) for row in rows}
            counts[str(source["name"])] = sum(
                not isinstance(row.get("teacher_answer"), str)
                or not row["teacher_answer"].strip()
                for row in rows
            )
        return counts
    key = "train_annotations" if split == "train" else "val_annotations"
    rows = _load_json_list(
        ensure_within_any(
            Path(data[key]).expanduser(), [root, artifact], must_exist=True
        )
    )
    return {
        str(data.get("dataset_name", split)): sum(
            not isinstance(row.get("teacher_answer"), str)
            or not row["teacher_answer"].strip()
            for row in rows
        )
    }


def build_dataset_from_config(
    config: dict[str, Any], split: str, *, require_teacher_answer: bool | None = None
) -> Dataset:
    """Build an LLM-only dataset; never constructs masks or superpoints."""
    data = config["data"]
    require = split == "train" if require_teacher_answer is None else bool(require_teacher_answer)
    missing = teacher_answer_missing_counts(config, split)
    if require and any(missing.values()):
        detail = ", ".join(f"{name}={count}" for name, count in sorted(missing.items()))
        raise ValueError(
            "teacher_answer is required before model or point-cloud loading; "
            f"missing labels by source: {detail}"
        )

    common = dict(
        scene_root=data["scene_root"],
        authorized_root=config["project"]["authorized_code_root"],
        split=split,
        tuple_fields=data["tuple_fields"],
        target_label_field=data["target_label_field"],
        annotation_text_field=data.get("annotation_text_field", "description"),
        scene_file_suffix=data.get("scene_file_suffix", "refer"),
        num_bins=int(data["num_bins"]),
        world_extent_m=float(data["world_extent_m"]),
        loc_margin_m=float(data.get("loc_margin_m", 0.30)),
        random_voxel_sample=bool(data.get("train_random_voxel_sample", False) and split == "train"),
        seed=int(data.get("seed", 42)),
        segmentation_labels=False,
        superpoint_config={"enabled": False},
        normal_config=data.get("normal"),
        require_teacher_answer=require,
        annotation_authorized_roots=[
            config["project"]["authorized_code_root"],
            config["project"]["artifact_root"],
        ],
    )
    if split == "train" and data.get("train_sources"):
        datasets: list[Dataset] = []
        names: list[str] = []
        probabilities: list[float] = []
        primary_scenes: set[str] | None = None
        for source in data["train_sources"]:
            annotations = _load_training_source(
                source,
                authorized_root=Path(config["project"]["authorized_code_root"]),
                allowed_scenes=primary_scenes,
                additional_annotation_roots=(
                    Path(config["project"]["artifact_root"]),
                ),
            )
            if not annotations:
                raise ValueError(f"Training source {source['name']} produced no valid samples")
            if bool(source.get("defines_primary_scenes", False)):
                primary_scenes = {str(item["scene_id"]) for item in annotations}
            datasets.append(
                ReasonSegDataset(
                    source["annotations"],
                    dataset_name=str(source["name"]),
                    include_types=None,
                    annotation_payload=annotations,
                    **common,
                )
            )
            names.append(str(source["name"]))
            probabilities.append(float(source["sampling_weight"]))
        return GroundingMixtureDataset(
            datasets,
            source_names=names,
            source_probabilities=probabilities,
            epoch_size=int(data["mixture_epoch_size"]),
        )

    key = "train_annotations" if split == "train" else "val_annotations"
    if not data.get(key):
        raise ValueError(f"Set data.{key} before running {split}")
    return ReasonSegDataset(
        data[key],
        dataset_name=data.get("dataset_name", "scanrefer"),
        include_types=data.get("include_types"),
        **common,
    )

def _as_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
    """把数组或张量转换为指定 dtype 的 torch.Tensor。"""
    if isinstance(value, np.ndarray) and not value.flags.writeable:
        value = value.copy()
    return torch.as_tensor(value, dtype=dtype)


def _normalize_rgb(rgb: torch.Tensor) -> torch.Tensor:
    """把 RGB 数据统一到 [0, 1] 浮点范围。"""
    minimum = float(rgb.amin())
    maximum = float(rgb.amax())
    if minimum >= 0.0 and maximum <= 1.5:
        rgb = rgb * 255.0
    elif minimum >= -1.1 and maximum <= 1.1 and minimum < 0.0:
        return rgb.clamp(-1.0, 1.0)
    return (rgb.clamp(0.0, 255.0) / 127.5) - 1.0
