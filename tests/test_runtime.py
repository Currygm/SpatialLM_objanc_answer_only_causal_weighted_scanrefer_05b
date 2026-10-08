import importlib.util
import json
from pathlib import Path
import pytest
import torch
from torch import nn

def helpers():
    assert importlib.util.find_spec('runtime_helpers') is not None, 'runtime helpers missing'
    import runtime_helpers
    return runtime_helpers

def test_frozen_digest_handles_scalars_and_bfloat16():
    h=helpers(); model=nn.Linear(2,2).to(torch.bfloat16)
    model.register_parameter('scalar',nn.Parameter(torch.tensor(0.),requires_grad=False))
    model.weight.requires_grad_(False)
    before=h.frozen_digest(model)
    with torch.no_grad():model.bias.add_(1)
    assert h.frozen_digest(model)==before
    with torch.no_grad():model.scalar.add_(1)
    assert h.frozen_digest(model)!=before

def test_checkpoint_rejects_shapes_before_partial_mutation():
    h=helpers(); model=nn.Linear(2,2)
    before=model.weight.detach().clone()
    bad={'weight':torch.ones(2,2),'bias':torch.ones(3)}
    with pytest.raises(ValueError):h.load_trainable_state(model,bad)
    assert torch.equal(before,model.weight)
    with pytest.raises(ValueError):h.load_trainable_state(model,{'weight':torch.ones(2,2)})

def test_atomic_save_preserves_previous_checkpoint_on_failure(tmp_path,monkeypatch):
    h=helpers(); p=tmp_path/'model.pt'; h.atomic_save({'step':1},p)
    def fail(*args,**kwargs):raise OSError('disk failure')
    monkeypatch.setattr(torch,'save',fail)
    with pytest.raises(OSError):h.atomic_save({'step':2},p)
    assert torch.load(p,weights_only=False)['step']==1

def test_async_log_and_disable_after_io_error(tmp_path):
    h=helpers(); log=h.AsyncJsonLog(tmp_path/'train.jsonl')
    log.submit({'step':1}); log.close(timeout=2)
    assert json.loads((tmp_path/'train.jsonl').read_text())=={'step':1}
    blocked=tmp_path/'directory'; blocked.mkdir()
    log=h.AsyncJsonLog(blocked); log.submit({'step':1}); log.close(timeout=2)
    assert log.disabled

def test_manifest_detects_changed_base_and_missing_shard(tmp_path):
    h=helpers(); (tmp_path/'config.json').write_text('{}')
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'model-1.safetensors'}}))
    (tmp_path/'model-1.safetensors').write_bytes(b'base')
    old=h.checkpoint_manifest(tmp_path)
    (tmp_path/'model-1.safetensors').write_bytes(b'changed')
    assert h.checkpoint_manifest(tmp_path)!=old
    (tmp_path/'model-1.safetensors').unlink()
    with pytest.raises(FileNotFoundError):h.checkpoint_manifest(tmp_path)

def test_resume_rejects_wrong_base_manifest():
    h=helpers()
    with pytest.raises(ValueError):
        h.validate_checkpoint({'objective':'x','architecture_signature':{'h':2048},'base_manifest':{'a':'old'}},
                              'x',{'h':2048},{'a':'new'})

def test_checkpoint_roundtrip_and_latest(tmp_path):
    import train_segmentation as runtime
    class Toy(nn.Linear):
        def architecture_signature(self):return {'hidden':2048}
        def trainable_state_dict(self):return {n:p.detach().cpu() for n,p in self.named_parameters()}
    torch.manual_seed(8); model=Toy(2,2)
    annotation=tmp_path/'annotations.json'; annotation.write_text('[]')
    config={'model':{'model_name_or_path':'base'},'data':{'train_sources':[{'name':'scanrefer','annotations':str(annotation)}]}}
    model.config=config
    optimizer=torch.optim.AdamW(model.parameters()); scheduler=runtime.scheduler_for(optimizer,10,.1)
    x=torch.tensor([[1.,2.]]); expected=model(x).detach().clone()
    path=runtime.save_checkpoint(model,optimizer,scheduler,tmp_path,3,1,config,manifest={'base':'sha'},frozen_hash='hash')
    with torch.no_grad():model.weight.add_(10)
    assert runtime.load_resume(model,optimizer,scheduler,path,{'base':'sha'})==(3,1)
    assert torch.equal(model(x).detach(),expected)
    assert torch.load(tmp_path/'latest.pt',weights_only=False)['step']==3

def test_existing_checkpoints_require_explicit_resume(tmp_path):
    from train_segmentation import check_output_directory
    (tmp_path/'latest.pt').write_bytes(b'existing')
    with pytest.raises(FileExistsError):check_output_directory(tmp_path,None)
    check_output_directory(tmp_path,tmp_path/'latest.pt')

def test_data_manifest_includes_validation(tmp_path):
    from train_segmentation import data_manifest
    annotation=tmp_path/'train.json'; annotation.write_text('[]')
    validation=tmp_path/'val.json'; validation.write_text('[1]')
    result=data_manifest({'data':{'train_sources':[{'name':'scanrefer','annotations':annotation}],
                                  'val_annotations':validation}})
    assert set(result)=={'scanrefer','validation'}
    assert result['scanrefer']!=result['validation']

def test_signature_rejects_shape_compatible_decoder_changes():
    import copy
    import yaml
    from types import SimpleNamespace
    from spatiallm_compressed_visual_objanc_full_attention.model.segmentation_model import ObjAncSegmentationModel,OBJECTIVE
    config=yaml.safe_load((Path(__file__).parents[1]/'configs/scanrefer_teacher_forcing_univlg9.yaml').read_text())
    model=ObjAncSegmentationModel.__new__(ObjAncSegmentationModel); nn.Module.__init__(model)
    model.config=config; model.answer_lm=SimpleNamespace(base=SimpleNamespace(config=SimpleNamespace(hidden_size=2048,num_hidden_layers=36)),lora_module_count=252)
    model.decoder=SimpleNamespace(version='test'); model.answer_lm_weight=.1
    expected=copy.deepcopy(model.architecture_signature())
    for field,value in [('num_heads',4),('text_temperature',.2),('attention_mask_threshold',.3)]:
        old=config['decoder'][field]; config['decoder'][field]=value
        with pytest.raises(ValueError):
            helpers().validate_checkpoint({'objective':OBJECTIVE,'architecture_signature':expected,'base_manifest':{}},
                                          OBJECTIVE,model.architecture_signature(),{})
        config['decoder'][field]=old

def test_signature_rejects_numeric_loss_matcher_ranking_and_threshold_changes():
 import copy,yaml
 from types import SimpleNamespace
 from spatiallm_compressed_visual_objanc_full_attention.model.segmentation_model import ObjAncSegmentationModel,OBJECTIVE
 cfg=yaml.safe_load((Path(__file__).parents[1]/'configs/scanrefer_teacher_forcing_univlg9.yaml').read_text())
 m=ObjAncSegmentationModel.__new__(ObjAncSegmentationModel);nn.Module.__init__(m);m.config=cfg
 m.answer_lm=SimpleNamespace(base=SimpleNamespace(config=SimpleNamespace(hidden_size=896,num_hidden_layers=24)),lora_module_count=168)
 m.decoder=SimpleNamespace(version='test');m.answer_lm_weight=.1
 expected=copy.deepcopy(m.architecture_signature())
 for section,key in [('loss','span_weight'),('loss','mask_bce_weight'),('loss','dice_weight'),('loss','focal_alpha'),('loss','focal_gamma'),('loss','aux_weight'),('matcher','span_weight'),('ranking','temperature'),('ranking','weight'),('validation','mask_threshold')]:
  original=cfg[section][key];cfg[section][key]=float(original)+.123
  with pytest.raises(ValueError):helpers().validate_checkpoint({'objective':OBJECTIVE,'architecture_signature':expected,'base_manifest':{}},OBJECTIVE,m.architecture_signature(),{})
  cfg[section][key]=original
