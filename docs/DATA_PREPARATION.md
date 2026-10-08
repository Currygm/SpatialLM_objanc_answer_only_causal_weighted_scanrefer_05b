# Prepare the three missing ScanRefer resources

Run these commands in the cloned repository on the new Linux machine.
These are standalone CPU preprocessing scripts, not training launchers.
They do not import SpatialLM, load LLM weights, or require the old server's paths.

## Dependencies and source data

Use the project's compatible Python environment (Python >=3.10).
Preprocessing needs numpy, scipy and torch; pytest is needed only for tests.
For example, install missing packages into a dedicated preparation environment;
do not upgrade an existing training environment blindly.

Obtain the official ScanEnts3D ScanRefer archive:
https://scanents3d.github.io/ScanEnts3D_ScanRefer.zip
Expected SHA256: 5b5e5a56511ab32ce43776817e5e1ce4c6965d41408470a0306862b731f91590.
The script refuses an archive with a different hash.

Obtain Reason3D's ScanNet referring-segmentation scene files under the applicable
dataset terms: https://github.com/KuanchihHuang/Reason3D#data-preparation.
Extract and inspect the archives so each trusted five-array .pth is at
data/scannetv2/train/sceneXXXX_XX_refer.pth or
data/scannetv2/val/sceneXXXX_XX_refer.pth.
The tuple layout is xyz, rgb, original superpoint id, semantic id, instance id.
Only load trusted .pth files (PyTorch pickle is executable).

## 1. Original OBJ/ANC teacher answers

```bash
mkdir -p data/downloads
wget https://scanents3d.github.io/ScanEnts3D_ScanRefer.zip -O data/downloads/ScanEnts3D_ScanRefer.zip

python prepare_objanc_teacher_answers.py \
  --archive data/downloads/ScanEnts3D_ScanRefer.zip \
  --output-dir data/objanc_teacher_answers_v1
```

The mapping is unchanged: join official tokens with spaces, preserve target-first
grounded objects and all mentions (maximum 50 objects), drop samples with no
target mention, then run the original OBJ-first answer construction. No LLM is
used. Expected counts are train 36,569 and val 9,483. Ambiguous anchors are preserved
in answer annotations and filtered by the existing segmentation target builder.
Files are scanrefer_train.json, scanrefer_val.json, failures.json and manifest.json.
Existing outputs are refused unless --overwrite is supplied.
The raw input JSONs are inside the official ZIP; an unrelated ordinary ScanRefer
annotation file is not an equivalent replacement.

## 2. Normals: CPU radius-neighborhood PCA

```bash
OMP_NUM_THREADS=2 python prepare_scanrefer_normals.py \
  --scene-root data/scannetv2 \
  --annotations data/objanc_teacher_answers_v1/scanrefer_train.json data/objanc_teacher_answers_v1/scanrefer_val.json \
  --output-dir cache/scanrefer_normals --resume
```

This uses the original SciPy cKDTree radius 0.1m / maximum 30 neighbors and
torch.linalg.eigh estimator, explicitly on CPU. The cache keeps the relative scene
path (train/..._refer.npy) and the training loader's required manifest schema and
version scipy-kdtree-cuda-pca-radius0.1-maxnn30-v1. The historical version name
contains cuda, but the estimator also supports CPU; numerical equality across
different devices/libraries is not promised. Coordinates stay in original order.
Resume requires matching coordinate hash, estimator parameters and valid arrays.

## 3. New superpoints: current point-based k-NN algorithm

```bash
OMP_NUM_THREADS=2 python prepare_scanrefer_superpoints.py \
  --scene-root data/scannetv2 \
  --annotations data/objanc_teacher_answers_v1/scanrefer_train.json data/objanc_teacher_answers_v1/scanrefer_val.json \
  --output-dir cache/scanrefer_reason3d_superpoint --resume
```

Uses the existing CachedSuperpointSegmenter on the entire raw float32 xyz array:
graph_k8, normal_k20, edge_radius_scale3, k_thresh0.01, min_vertices20,
reason3d-knn-v1. It estimates its own graph normals; the separate normal cache is
the Sonata input. Original superpoint IDs in the downloaded .pth are not reused.
The .npz coordinate digest and parameter metadata exactly follow the training
loader's current implementation.

## Small checks and continuation

Add --limit 1 to a geometry command to process one scene first. This is a partial
cache, not a full training-ready cache. Rerun without --limit and with --resume to
finish. Outputs may be chosen freely on the new machine; source .pth files are
read-only. Existing correctly validated entries are reused with --resume, invalid
entries are repaired. Default execution is sequential and CPU; full scenes can
still need substantial RAM, especially for k-NN graph construction.

## Training integration and scope

Set data.scene_root, data.train_sources[0].annotations, data.val_annotations,
data.normal.cache_dir and data.superpoint.cache_dir in your training configuration
to these local absolute paths; retain require_cache=true after full preparation.
Generated data/, models/ and cache/ are excluded from Git.

The preparation commands are portable. The current training code and runtime
scripts still retain original node-specific path checks and external SpatialLM
source dependencies. This change does not remove those checks or make the entire
training project portable. That integration remains a separate task.

## Verification

New tests check original answer spans, missing targets, ambiguous anchors,
duplicate annotation rejection, normal loader/manifest compatibility, valid
resume and corrupted-cache repair, exact superpoint cache loading, and scene path
validation. No full scene-cache preprocessing or GPU run is part of this change.

2026-10-09 CPU verification on node224: 48 tests passed (23.85s). The official archive hash and full mapped counts 36569/9483 matched. First 200 train and 200 val answers matched every field of the original annotations. Independent review findings were fixed with regression tests. See data_preparation_cpu_verification.json.
