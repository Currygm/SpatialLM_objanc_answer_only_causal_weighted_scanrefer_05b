"""OBJ/ANC mask-target filtering and four-source segmentation datasets."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from .dataset import (
    GroundingMixtureDataset,
    ReasonSegDataset,
    _load_json_list,
    _load_training_source,
)
from ..config import ensure_within_any


def _span_key(entity: dict[str, Any]) -> tuple[int, int, str]:
    span = entity.get("source_char_span") or [-1, -1]
    return int(span[0]), int(span[1]), str(entity.get("phrase", "")).strip().casefold()


def filter_answer_entities(annotation: dict[str, Any]) -> dict[str, Any]:
    """Use only unambiguous Answer entities as mask/Hungarian targets."""
    row = dict(annotation)
    entities = row.get("answer_entities")
    if not isinstance(entities, list) or not entities:
        raise ValueError("Segmentation requires non-empty answer_entities")
    if str(entities[0].get("role", "")).upper() != "OBJ":
        raise ValueError("The first Answer entity must be OBJ")
    raw_ids = [int(x) for x in row["grounding_object_ids"]]
    raw_spans = row["grounding_char_spans"]
    raw_phrases = row["grounding_phrases"]
    candidates: dict[tuple[int, int, str], set[int]] = defaultdict(set)
    for object_id, spans, phrases in zip(raw_ids, raw_spans, raw_phrases):
        for span, phrase in zip(spans, phrases):
            candidates[(int(span[0]), int(span[1]), str(phrase).strip().casefold())].add(object_id)

    valid_entities: list[dict[str, Any]] = []
    valid_indices: list[int] = []
    invalid: list[dict[str, Any]] = []
    for index, entity_value in enumerate(entities):
        entity = dict(entity_value)
        role = str(entity.get("role", "")).upper()
        object_id = int(entity["object_id"])
        grounding_row = int(entity.get("grounding_row", -1))
        reason = None
        if grounding_row < 0 or grounding_row >= len(raw_ids) or raw_ids[grounding_row] != object_id:
            reason = "missing_grounding_row"
        elif role == "ANC" and len(candidates[_span_key(entity)]) != 1:
            reason = "ambiguous_anchor"
        elif role not in {"OBJ", "ANC"}:
            reason = "unknown_role"
        if reason is not None:
            entity["mask_valid"] = False
            entity["mask_invalid_reason"] = reason
            invalid.append(entity)
            continue
        entity["mask_valid"] = True
        valid_entities.append(entity)
        valid_indices.append(index)
    if not valid_entities or str(valid_entities[0]["role"]).upper() != "OBJ":
        raise ValueError("Every segmentation sample requires one valid OBJ target")
    row["grounding_object_ids"] = [int(x["object_id"]) for x in valid_entities]
    row["grounding_char_spans"] = [[list(map(int, x["source_char_span"]))] for x in valid_entities]
    row["grounding_phrases"] = [[str(x["phrase"])] for x in valid_entities]
    row["grounding_roles"] = [str(x["role"]).upper() for x in valid_entities]
    row["answer_entity_indices"] = valid_indices
    row["mask_invalid_answer_entities"] = invalid
    return row


def build_segmentation_dataset(config: dict[str, Any], split: str) -> Dataset:
    data = config["data"]
    code_root = Path(config["project"]["authorized_code_root"])
    artifact_root = Path(config["project"]["artifact_root"])
    source_data_root = Path(config["project"]["source_data_root"])
    common = dict(
        scene_root=data["scene_root"], authorized_root=code_root, split=split,
        tuple_fields=data["tuple_fields"], target_label_field=data["target_label_field"],
        annotation_text_field=data.get("annotation_text_field", "description"),
        scene_file_suffix=data.get("scene_file_suffix", "refer"), num_bins=int(data["num_bins"]),
        world_extent_m=float(data["world_extent_m"]), loc_margin_m=float(data.get("loc_margin_m", .3)),
        random_voxel_sample=bool(data.get("train_random_voxel_sample", False) and split == "train"),
        seed=int(data.get("seed", 42)), segmentation_labels=True,
        superpoint_config=data["superpoint"], normal_config=data["normal"], require_teacher_answer=True,
        annotation_authorized_roots=[code_root, artifact_root, source_data_root],
    )
    if split == "train":
        datasets, names, probabilities = [], [], []
        primary_scenes: set[str] | None = None
        for source in data["train_sources"]:
            annotations = _load_training_source(
                source, authorized_root=code_root, allowed_scenes=primary_scenes,
                additional_annotation_roots=(artifact_root, source_data_root),
            )
            if bool(source.get("defines_primary_scenes", False)):
                primary_scenes = {str(x["scene_id"]) for x in annotations}
            filtered = [filter_answer_entities(x) for x in annotations]
            datasets.append(ReasonSegDataset(source["annotations"], dataset_name=str(source["name"]), annotation_payload=filtered, **common))
            names.append(str(source["name"])); probabilities.append(float(source["sampling_weight"]))
        if data.get("sampling_mode") == "full_dataset_no_replacement":
            if len(datasets) != 1 or names != ["scanrefer"]:
                raise ValueError("Full-dataset mode requires exactly one ScanRefer source")
            if len(datasets[0]) != int(data["expected_train_samples"]):
                raise ValueError(
                    f"ScanRefer sample count changed: {len(datasets[0])} "
                    f"!= {int(data['expected_train_samples'])}"
                )
            return datasets[0]
        return GroundingMixtureDataset(datasets, source_names=names, source_probabilities=probabilities, epoch_size=int(data["mixture_epoch_size"]))
    path = ensure_within_any(Path(data["val_annotations"]), [code_root, artifact_root, source_data_root], must_exist=True)
    annotations = [filter_answer_entities(x) for x in _load_json_list(path)]
    return ReasonSegDataset(path, dataset_name=str(data.get("dataset_name", "scanrefer")), annotation_payload=annotations, **common)
