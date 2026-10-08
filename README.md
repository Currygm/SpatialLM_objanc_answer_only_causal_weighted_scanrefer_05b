# SpatialLM 0.5B Answer-only causal role-weighted ScanRefer

Independent experiment rooted here; original source is read-only. No source output, log, cache or task checkpoint is copied. Official Stage1 checkpoint is `/home/yibo/SpatialLM-pretrain/outputs/spatiallm_qwen25_officialinit_sonata_lora64_4gpu231_v1/stage1_projector/checkpoint-18681/official_format_full` (896 hidden, 24 layers). Read-only preflight found 746 tensors, 452 point-backbone tensors, four projector tensors, zero LoRA tensors and embedding shape [151936,896]. Task parameters are fresh.

Original Sonata enc4 -> frozen original projector -> Qwen remains the language visual path. Joint adds original full PointUnet -> 512-dimensional mean-pooled full superpoints -> unchanged 50-query, nine-layer, 256-hidden, eight-head Decoder. No superpoint sampling or new language visual projector exists.

Answer stage: three complete ScanRefer epochs; only Qwen LoRA r16/alpha32/dropout0.05 and four separate input/output task rows, all LR1e-5. No Sonata LoRA, U-Net or Decoder is constructed. Joint requires completed-three-epoch Answer state, then creates Sonata LoRA LR1e-5, U-Net/Decoder LR1e-4 and trains 50 epochs. Dense Qwen/Sonata/original projector remain frozen. AdamW decay0.01, BF16, checkpointing, batch1/accumulation2, cosine minimum0.1; optimizer and scheduler restart at stage transition. Every epoch persists optimizer/scheduler/tokenizer and per-rank RNG; resume validates objective, architecture, data hashes, frozen-base manifest and exact config.

One causal teacher forward packs visual+question+answer+EOS and returns AnswerOnlyContext. Decoder receives all answer text including tags, using each token's own hidden state; question/prompt/padding/extra EOS are excluded. Phrase positives cover only phrase content; answer roles include tag blocks, retain [QUESTION,OBJ,ANC,OTHER] embedding layout, and QUESTION is unused. CE/accuracy use causal-shift answer+EOS. `pack_causal_batch` is common for training and chosen-answer replay. Generated evaluation uses cached constrained generation then one chosen-answer replay; raw diagnostics separately replay cached generated text. Original fallback and phrase32/answer128 budgets remain.

G(X)=1.5 meanOBJ(X)+0.5 meanANC(X), empty ANC contributes zero. Matched-mask BCE/Dice and final detached hard-raw-point-IoU ranking CE use independent entity role means. Matched-query focal groups independently normalize each role by its own positive phrase-token count over all valid answer tokens. Unmatched-query zero-background focal is retained separately with coefficient1 and global valid-positive denominator. Final and every auxiliary layer use the same reduction. Total =6 span+5 BCE+2 Dice+0.5 sumAux+0.5 ranking+0.1 AnswerCE. Answer stage uses unscaled CE. Hungarian remains -2 meanSigmoidText+15 BCE+6 Dice. Logs retain raw role values, role contributions, background and all component contributions, including each auxiliary layer.

Read-only train/val annotations are reused from SpatialLM_compressed_visual_objanc_full_attention_v1; train36569/val9483, original normal/superpoint caches required. Evaluation exact-shards all9483 by default and reports OBJ/ANC/all selected/matched/oracle, precision/recall/Dice, Qmatch, exact best-query top1/rank, threshold topK, explicit metric denominators, fallback/language counts, visual tokens, forward/replay counts and peak memory. Raw format-invalid outputs have separate unevaluable counts and coverage.

## Commands (GPU runs pending)

```bash
source scripts/runtime_env.sh
# Fresh complete Answer3 -> joint50 workflow: bash scripts/train_all.sh
/home/huangxiang/anaconda3/envs/gmspatialLMseg/bin/python train_answer.py configs/scanrefer_teacher_forcing_univlg9.yaml
/home/huangxiang/anaconda3/envs/gmspatialLMseg/bin/python train_segmentation.py configs/scanrefer_teacher_forcing_univlg9.yaml --answer-checkpoint outputs/open_answer/latest.pt
/home/huangxiang/anaconda3/envs/gmspatialLMseg/bin/python evaluate_segmentation.py configs/scanrefer_teacher_forcing_univlg9.yaml outputs/joint/latest.pt evaluations/joint
```

Use `torchrun --nproc_per_node=N` in place of Python after explicitly selecting authorized GPUs. Source runtime_env first; TMPDIR realpath/length is checked at `/home/huangxiang/codes/guiming/.ao05tmp`; all caches are rooted here. `--resume` resumes only the same stage.

CPU verification command: `mkdir -p .tmp; CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -m pytest -q -s -p no:cacheprovider --basetemp=.tmp/pytest tests`. `-s` avoids this remote filesystem's pytest temporary capture truncate failure. See docs/progress.md for exact executed results. No GPU training/evaluation has run. Real-base CPU initialization/migration checks passed in 100.78s: Answer LoRA 8,798,208 + separate task rows 7,168 parameters; 338 state tensors migrated exactly into joint; joint 29,531,392 trainable parameters; old objective rejected. Synthetic completed-three-epoch metadata verified mechanics only; no actual Answer training occurred. See docs/cpu_real_base_initialization.json. Sparse point backward, BF16 GPU/DDP memory, complete train/eval and checkpoint roundtrips on GPUs remain pending.
