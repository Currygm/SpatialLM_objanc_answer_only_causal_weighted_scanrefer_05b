"""Deterministic Qwen3D-style ScanEnts3D multi-object derivation."""

from __future__ import annotations

import hashlib
import json
import random
import statistics
import zipfile
from pathlib import Path
from typing import Any, Iterable


OFFICIAL_ARCHIVE_SHA256 = (
    "5b5e5a56511ab32ce43776817e5e1ce4c6965d41408470a0306862b731f91590"
)
MAPPING_VERSION = "scanents3d_target_anchor_entities_v1"
QWEN3D_REFERENCE_COMMIT = "7ef6d01e495290639884878a06e43ba2905e0ef5"
MAX_GROUNDING_OBJECTS = 50
SOURCE_FILES = {
    "train": ("ScanRefer_filtered_train_ScanEnts3D.json", 36_665),
    "val": ("ScanRefer_filtered_val_ScanEnts3D.json", 9_508),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def annotation_key(annotation: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(annotation["scene_id"]),
        str(annotation["object_id"]),
        str(annotation["ann_id"]),
    )


def token_char_offsets(tokens: list[str]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        start = cursor
        end = start + len(token)
        offsets.append((start, end))
        cursor = end + 1
    return offsets


def _entity_object_id(value: Any) -> str:
    descriptor = str(value)
    return descriptor.split("_", 1)[0]


def grounding_entities(
    annotation: dict[str, Any],
    *,
    max_objects: int = MAX_GROUNDING_OBJECTS,
) -> tuple[list[int], list[list[list[int]]], list[list[str]], list[list[list[int]]], int]:
    """Return target-first object ids and all of their ScanEnts3D mentions.

    Object order follows Qwen3D: the ScanRefer target occupies row zero and unique
    anchors follow their first occurrence in ``entities``. A shared entity span is
    copied to every referenced object, while repeated mentions are appended to the
    same object row. The official first/last token interval is intentionally made
    continuous, including tokens omitted from a non-contiguous entity index list.
    """
    if max_objects < 1:
        raise ValueError("max_objects must be positive")
    tokens = annotation.get("token")
    entities = annotation.get("entities")
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise ValueError("ScanEnts3D annotation has an invalid token list")
    if not isinstance(entities, list):
        raise ValueError("ScanEnts3D annotation has an invalid entities list")
    offsets = token_char_offsets(tokens)
    target_id = int(annotation["object_id"])
    anchor_order: list[int] = []
    mentions: dict[int, dict[tuple[int, int], tuple[list[int], str]]] = {}
    for entity in entities:
        if not isinstance(entity, list) or len(entity) != 2:
            raise ValueError("Each ScanEnts3D entity must be [token_indices, objects]")
        raw_indices, objects = entity
        if not isinstance(raw_indices, list) or not isinstance(objects, list):
            raise ValueError("Invalid ScanEnts3D entity fields")
        indices = sorted({int(value) for value in raw_indices})
        if not indices:
            raise ValueError("ScanEnts3D entity has no token indices")
        if indices[0] < 0 or indices[-1] >= len(tokens):
            raise ValueError("ScanEnts3D entity token index is out of range")
        first, last = indices[0], indices[-1]
        char_span = (offsets[first][0], offsets[last][1])
        token_range = [first, last]
        phrase = " ".join(tokens[first : last + 1])
        for descriptor in objects:
            object_id = int(_entity_object_id(descriptor))
            if object_id != target_id and object_id not in anchor_order:
                anchor_order.append(object_id)
            mentions.setdefault(object_id, {})[char_span] = (token_range, phrase)
    if target_id not in mentions:
        return [], [], [], [], 0
    object_ids = [target_id] + [value for value in anchor_order if value != target_id]
    truncated_count = max(len(object_ids) - max_objects, 0)
    object_ids = object_ids[:max_objects]
    char_spans: list[list[list[int]]] = []
    phrases: list[list[str]] = []
    token_ranges: list[list[list[int]]] = []
    for object_id in object_ids:
        ordered = list(mentions[object_id].items())
        char_spans.append([[start, end] for (start, end), _ in ordered])
        phrases.append([value[1] for _, value in ordered])
        token_ranges.append([value[0] for _, value in ordered])
    return object_ids, char_spans, phrases, token_ranges, truncated_count


def target_entity_spans(
    annotation: dict[str, Any],
) -> tuple[list[list[int]], list[str], list[list[int]]]:
    """Return char spans, phrases and official token ranges for the target object.

    ScanEnts3D entities may list more than one object and may contain non-contiguous
    token indices. Qwen3D covers the complete inclusive range from the first to the
    last official token, so this function intentionally follows that convention.
    """
    object_ids, spans, phrases, token_ranges, _ = grounding_entities(annotation)
    if not object_ids:
        return [], [], []
    return spans[0], phrases[0], token_ranges[0]


def derive_annotation(annotation: dict[str, Any]) -> dict[str, Any] | None:
    object_ids, spans, phrases, token_ranges, truncated_count = grounding_entities(
        annotation
    )
    if not object_ids:
        return None
    output = dict(annotation)
    output["original_description"] = str(annotation.get("description", ""))
    output["description"] = " ".join(annotation["token"])
    output["target_char_spans"] = spans[0]
    output["target_phrases"] = phrases[0]
    output["target_token_ranges"] = token_ranges[0]
    output["grounding_object_ids"] = object_ids
    output["grounding_char_spans"] = spans
    output["grounding_phrases"] = phrases
    output["grounding_token_ranges"] = token_ranges
    output["grounding_truncated_count"] = truncated_count
    output["entity_mapping_version"] = MAPPING_VERSION
    return output


def _distribution(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    percentile_index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": ordered[percentile_index],
    }


def derive_split(
    annotations: Iterable[dict[str, Any]], split: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = list(annotations)
    seen: set[tuple[str, str, str]] = set()
    retained: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    positive_token_counts: list[int] = []
    grounding_object_counts: list[int] = []
    repeated_mentions = 0
    truncated_samples = 0
    truncated_objects = 0
    for index, annotation in enumerate(rows):
        key = annotation_key(annotation)
        if key in seen:
            raise ValueError(f"Duplicate ScanEnts3D annotation key: {key}")
        seen.add(key)
        derived = derive_annotation(annotation)
        if derived is None:
            failures.append(
                {
                    "split": split,
                    "index": index,
                    "scene_id": key[0],
                    "object_id": key[1],
                    "ann_id": key[2],
                    "reason": "missing_target_entity",
                }
            )
            continue
        retained.append(derived)
        ranges = derived["target_token_ranges"]
        positive_token_counts.append(
            len({token for first, last in ranges for token in range(first, last + 1)})
        )
        repeated_mentions += int(len(ranges) > 1)
        grounding_object_counts.append(len(derived["grounding_object_ids"]))
        truncated = int(derived["grounding_truncated_count"])
        truncated_samples += int(truncated > 0)
        truncated_objects += truncated
    if not retained:
        raise ValueError(f"ScanEnts3D {split} produced no retained annotations")
    summary = {
        "source_count": len(rows),
        "retained_count": len(retained),
        "dropped_count": len(failures),
        "coverage": len(retained) / len(rows),
        "repeated_target_mention_count": repeated_mentions,
        "positive_official_token_count": _distribution(positive_token_counts),
        "grounding_object_count": _distribution(grounding_object_counts),
        "truncated_sample_count": truncated_samples,
        "truncated_object_count": truncated_objects,
    }
    return retained, failures, summary


def _write_json(path: Path, payload: Any) -> str:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return sha256_file(path)


def prepare_archive(
    archive: Path, output_dir: Path, *, audit_seed: int = 42, audit_size: int = 200
) -> dict[str, Any]:
    archive = archive.resolve(strict=True)
    archive_sha = sha256_file(archive)
    if archive_sha.lower() != OFFICIAL_ARCHIVE_SHA256:
        raise ValueError(
            f"Unexpected ScanEnts3D archive SHA256: {archive_sha}; "
            f"expected {OFFICIAL_ARCHIVE_SHA256}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    split_payloads: dict[str, list[dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    split_summaries: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(archive) as source:
        for split, (member, expected_count) in SOURCE_FILES.items():
            annotations = json.loads(source.read(member))
            if not isinstance(annotations, list) or len(annotations) != expected_count:
                raise ValueError(
                    f"Unexpected {split} annotation count: {len(annotations)}; "
                    f"expected {expected_count}"
                )
            retained, dropped, summary = derive_split(annotations, split)
            if summary["coverage"] < 0.99:
                raise ValueError(f"ScanEnts3D {split} coverage is below 99%")
            split_payloads[split] = retained
            failures.extend(dropped)
            split_summaries[split] = summary

    output_names = {
        "train": "ScanRefer_filtered_train_scanents_multi_v4.json",
        "val": "ScanRefer_filtered_val_scanents_multi_v4.json",
    }
    derived_sha = {
        split: _write_json(output_dir / output_names[split], payload)
        for split, payload in split_payloads.items()
    }
    _write_json(output_dir / "failures.json", failures)
    audit_pool = [
        {"split": split, **row}
        for split, payload in split_payloads.items()
        for row in payload
    ]
    audit = random.Random(audit_seed).sample(
        audit_pool, min(audit_size, len(audit_pool))
    )
    _write_json(output_dir / "audit_200.json", audit)
    summary = {
        "schema_version": "scanents3d_multi_v4_summary_v1",
        "official_url": "https://scanents3d.github.io/ScanEnts3D_ScanRefer.zip",
        "official_archive_sha256": archive_sha,
        "entity_mapping_version": MAPPING_VERSION,
        "qwen3d_reference_commit": QWEN3D_REFERENCE_COMMIT,
        "max_grounding_objects": MAX_GROUNDING_OBJECTS,
        "overflow_policy": "target_then_first_anchors",
        "text_reconstruction": "space_join_official_tokens",
        "matcher_positive_map": "normalized_sum_one",
        "focal_positive_map": "binary_via_gt_zero",
        "derived_annotation_sha256": derived_sha,
        "splits": split_summaries,
        "audit_seed": audit_seed,
        "audit_size": len(audit),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary
