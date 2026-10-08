"""CPU initialization/migration validation; no training or GPU execution."""
import gc, json, sys, time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spatiallm_compressed_visual_objanc_full_attention.config import load_segmentation_config
from spatiallm_compressed_visual_objanc_full_attention.model.answer_stage_3b import AnswerPretrainingStage
from spatiallm_compressed_visual_objanc_full_attention.model.segmentation_model import ObjAncSegmentationModel
from train_segmentation import load_answer_checkpoint,save_checkpoint,scheduler_for,load_resume
from runtime_helpers import checkpoint_manifest,frozen_digest
started=time.time()
cfg=load_segmentation_config(ROOT/"configs/scanrefer_teacher_forcing_univlg9.yaml")
manifest=checkpoint_manifest(cfg["model"]["model_name_or_path"])
answer=AnswerPretrainingStage(cfg)
assert answer.answer_lm.base.config.hidden_size==896
assert answer.answer_lm.base.config.num_hidden_layers==24
assert not hasattr(answer,"point_unet") and not hasattr(answer,"decoder")
groups=answer.optimizer_parameter_groups()
report={"device":"cpu","gpu_validated":False,"hidden_size":896,"layers":24,"answer_groups":{g["name"]:{"lr":g["lr"],"parameters":sum(p.numel() for p in g["params"])} for g in groups},"base_manifest":manifest}
with torch.no_grad():
 for p in answer.parameters():
  if p.requires_grad:p.fill_(0.000123)
expected=answer.trainable_state_dict()
optimizer=torch.optim.AdamW(groups)
scheduler=scheduler_for(optimizer,3,0.1)
scratch=ROOT/"docs/cpu_smoke_scratch"
scratch.mkdir(exist_ok=True)
frozen=frozen_digest(answer)
path=save_checkpoint(answer,optimizer,scheduler,scratch,0,3,cfg,manifest=manifest,frozen_hash=frozen,stage_name="answer")
resume_step,resume_epoch=load_resume(answer,optimizer,scheduler,path,manifest,stage_name="answer")
assert (resume_step,resume_epoch)==(0,3)
assert frozen_digest(answer)==frozen
report["answer_resume"]={"step":resume_step,"epoch":resume_epoch}
del optimizer,scheduler,answer
gc.collect()
joint=ObjAncSegmentationModel(cfg)
parent=load_answer_checkpoint(joint,path,manifest)
destination={**dict(joint.named_parameters()),**dict(joint.named_buffers())}
for n,v in expected.items():
 assert torch.equal(destination[n].detach().cpu(),v),n
report["migrated_tensors"]=len(expected)
groups=joint.optimizer_parameter_groups()
report["joint_groups"]={g["name"]:{"lr":g["lr"],"parameters":sum(p.numel() for p in g["params"])} for g in groups}
report["total_trainable_joint"]=sum(p.numel() for p in joint.parameters() if p.requires_grad)
bad=torch.load(path,map_location="cpu",weights_only=False)
bad["objective"]="spatiallm_objanc_teacher_forcing_scanrefer_univlg9_v1"
badpath=scratch/"old_objective.pt"
torch.save(bad,badpath)
try:
 load_answer_checkpoint(joint,badpath,manifest)
except ValueError:report["old_objective_rejected"]=True
else:raise AssertionError("Old objective accepted")
bad["objective"]=joint.config["training"].get("answer_objective","invalid")
report["elapsed_seconds"]=time.time()-started
for p in scratch.iterdir():
 assert p.is_file() and p.resolve().is_relative_to(ROOT)
 p.unlink()
scratch.rmdir()
(ROOT/"docs/cpu_real_base_initialization.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
