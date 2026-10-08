#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime_env.sh"
/home/huangxiang/anaconda3/envs/gmspatialLMseg/bin/python train_segmentation.py configs/scanrefer_teacher_forcing_univlg9.yaml "$@"
