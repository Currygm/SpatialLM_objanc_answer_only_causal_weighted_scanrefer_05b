#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/huangxiang/codes/guiming/guiming_codes/research/SpatialLM_objanc_answer_only_causal_weighted_scanrefer_05b_v1
TMP=/home/huangxiang/codes/guiming/.ao05tmp
mkdir -p "$TMP" "$ROOT/cache/huggingface" "$ROOT/cache/torch"
ACTUAL=$(realpath "$TMP")
[[ "$ACTUAL" == "$TMP" && ${#ACTUAL} -lt 65 ]] || { echo 'Unsafe/long TMPDIR' >&2; exit 1; }
export TMPDIR="$TMP" TMP="$TMP" TEMP="$TMP"
export HF_HOME="$ROOT/cache/huggingface" HF_HUB_CACHE="$ROOT/cache/huggingface/hub" TRANSFORMERS_CACHE="$ROOT/cache/huggingface/transformers" TORCH_HOME="$ROOT/cache/torch" XDG_CACHE_HOME="$ROOT/cache"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd "$ROOT"
