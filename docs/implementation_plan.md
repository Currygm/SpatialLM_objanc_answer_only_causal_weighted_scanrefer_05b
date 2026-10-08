# Authorized implementation plan
1. Create independent repo; snapshot original source hashes; reuse only necessary code.
2. Restore original enc4/projector and shared full-superpoint U-Net route. Implement answer-only causal batch packing and separate input/output task rows.
3. Implement independent OBJ/ANC group losses and explicit background focal on final/aux layers; preserve Hungarian.
4. Configure fresh Answer3 -> joint50, stage migration, checkpoint/RNG/config rejection and short remote TMPDIR.
5. Retain teacher/generated/raw all-validation evaluation with denominators, language/failure/call/memory diagnostics.
6. Execute CPU tests (real tiny Qwen and stage/checkpoint/loss/eval tests), real0.5B initialization controller verification, source-hash comparison and independent review. Commit. GPU runs require later authorization; do not execute now.
