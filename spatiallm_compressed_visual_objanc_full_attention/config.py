"""Configuration loading and strict path-boundary validation."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml

def _required(mapping: dict[str, Any], key: str) -> Any:
    value = mapping.get(key)
    if value is None:
        raise ValueError(f"Missing required configuration value: {key}")
    return value

def _mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = _required(mapping, key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value

def ensure_within(path: str | Path, root: str | Path, *, must_exist: bool) -> Path:
    root_path = Path(root).expanduser().resolve(strict=True)
    candidate = Path(path).expanduser()
    if must_exist:
        resolved = candidate.resolve(strict=True)
    else:
        missing: list[str] = []
        cursor = candidate
        while not cursor.exists():
            missing.append(cursor.name)
            if cursor.parent == cursor:
                raise ValueError(f"Cannot resolve a safe ancestor for {candidate}")
            cursor = cursor.parent
        resolved = cursor.resolve(strict=True).joinpath(*reversed(missing))
    if resolved != root_path and root_path not in resolved.parents:
        raise ValueError(f"Path escapes authorized root {root_path}: {resolved}")
    return resolved

def remap_renamed_data_path(path: Path) -> Path:
    """Resolve the historical Reason3D directory name without writing to it."""
    if path.exists() or not path.is_absolute():
        return path
    parts = list(path.parts)
    for index, part in enumerate(parts):
        if part == "Reason3D":
            candidate = Path(*(parts[:index] + ["Reason3D_base"] + parts[index + 1:]))
            if candidate.exists():
                return candidate
    return path


def ensure_within_any(path: str | Path, roots: list[str | Path], *, must_exist: bool) -> Path:
    errors: list[str] = []
    for root in roots:
        try:
            return ensure_within(path, root, must_exist=must_exist)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))
    raise ValueError(
        f"Path is outside all authorized roots: {path}; "
        + (errors[-1] if errors else "no roots")
    )


def load_segmentation_config(path: str | Path) -> dict[str, Any]:
    """Load the ScanRefer-only teacher-forcing segmentation config."""
    config_path = Path(path).expanduser().resolve(strict=True)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    project = _mapping(config, "project")
    code_root = Path(_required(project, "authorized_code_root")).resolve(strict=True)
    artifact_root = Path(_required(project, "artifact_root")).resolve(strict=True)
    expected = Path("/home/huangxiang/codes/guiming/guiming_codes/research/SpatialLM_objanc_answer_only_causal_weighted_scanrefer_05b_v1")
    if artifact_root != expected:
        raise ValueError(f"artifact_root must be exactly {expected}")
    source_data_root = Path(_required(project, "source_data_root")).resolve(strict=True)
    expected_source = Path("/home/huangxiang/codes/guiming/guiming_codes/research/SpatialLM_compressed_visual_objanc_full_attention_v1")
    if source_data_root != expected_source:
        raise ValueError(f"source_data_root must be exactly {expected_source}")
    project["authorized_code_root"], project["artifact_root"] = str(code_root), str(artifact_root)
    project["source_data_root"] = str(source_data_root)
    for key in ("output_dir", "evaluation_dir", "cache_dir"):
        project[key] = str(ensure_within(_required(project, key), artifact_root, must_exist=False))
    model = _mapping(config, "model")
    model["spatiallm_repo"] = str(ensure_within(model["spatiallm_repo"], "/home/yibo/SpatialLM-pretrain", must_exist=True))
    model["model_name_or_path"] = str(ensure_within(model["model_name_or_path"], "/home/yibo/SpatialLM-pretrain", must_exist=True))
    if model.get("enable_fourier_encode") is not False or model.get("attention_mode") != "causal":
        raise ValueError("Segmentation preserves the causal Answer pass and disables Sonata Fourier encoding")
    initialization = _mapping(config, "initialization")
    if initialization != {"mode": "fresh_05b"}:
        raise ValueError("This project initializes task parameters fresh; no old task checkpoint is allowed")
    if model.get("expected_hidden_size") != 896 or model.get("expected_num_hidden_layers") != 24:
        raise ValueError("Expected Qwen2.5-0.5B with hidden 896 and 24 layers")
    if model.get("structured_tokens") != ["<obj>", "</obj>", "<anc>", "</anc>"]:
        raise ValueError("Expected original four OBJ/ANC tokens")
    data = _mapping(config, "data")
    data["scene_root"] = str(ensure_within(data["scene_root"], code_root, must_exist=True))
    for source in data["train_sources"]:
        source["annotations"] = str(ensure_within_any(source["annotations"], [code_root, artifact_root, source_data_root], must_exist=True))
    data["val_annotations"] = str(ensure_within_any(data["val_annotations"], [code_root, artifact_root, source_data_root], must_exist=True))
    data["normal"]["cache_dir"] = str(ensure_within(data["normal"]["cache_dir"], code_root, must_exist=True))
    data["superpoint"]["cache_dir"] = str(ensure_within(data["superpoint"]["cache_dir"], code_root, must_exist=True))
    names = [x["name"] for x in data["train_sources"]]
    if names != ["scanrefer"] or data.get("sampling_mode") != "full_dataset_no_replacement":
        raise ValueError("Training must use the complete ScanRefer dataset without replacement")
    if int(data.get("expected_train_samples", -1)) != 36569:
        raise ValueError("ScanRefer teacher-forcing training expects 36,569 samples")
    decoder = _mapping(config, "decoder")
    if (int(decoder["num_layers"]), int(decoder["num_queries"]), int(decoder["text_dim"]), int(decoder["max_text_tokens"])) != (9, 50, 896, 384):
        raise ValueError("Decoder must be 9-layer/50-query/896-d text/max384")
    training = _mapping(config, "training")
    if training["objective"] != "spatiallm_objanc_answer_only_causal_weighted_scanrefer_05b_v1" or int(training["epochs"]) != 50:
        raise ValueError("Unexpected segmentation objective or epoch count")
    if int(training.get('answer_pretrain_epochs',-1))!=3:
        raise ValueError('ScanRefer Answer pretraining must run three epochs before joint training')
    if training.get("answer_mode") != "teacher" or bool(training.get("allow_generated_answers", True)):
        raise ValueError("Training must use teacher answers exclusively")
    if int(training["save_every_epochs"]) != 1 or int(training.get("keep_all_checkpoints", 0)) != 1:
        raise ValueError("Segmentation checkpoints must be saved every epoch and all retained")
    if config['point_encoder']['route']!='shared_native_stride_enc4_and_unet':raise ValueError('Original visual route required')
    return config
