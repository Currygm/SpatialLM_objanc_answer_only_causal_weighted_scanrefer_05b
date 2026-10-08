#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 0 ]]; then
  echo 'Usage: bash scripts/train_all.sh (fresh Answer3 then joint50; resume each stage separately)' >&2
  exit 64
fi
DIR=$(cd -- "$(dirname -- "$0")" && pwd)
bash "$DIR/train_answer.sh"
bash "$DIR/train_joint.sh" --answer-checkpoint outputs/open_answer/latest.pt
