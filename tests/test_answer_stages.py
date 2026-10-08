"""Catch wrong stage durations, optimizer boundaries and language-only handoff."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import copy
import pytest
import torch
from torch import nn
import yaml

ROOT=Path(__file__).resolve().parents[1]

def stage_module():
    name='spatiallm_compressed_visual_objanc_full_attention.model.answer_stage_3b'
    assert importlib.util.find_spec(name) is not None,'3B Answer-only stage is missing'
    import spatiallm_compressed_visual_objanc_full_attention.model.answer_stage_3b as module
    return module

def config():
    result=yaml.safe_load((ROOT/'configs/scanrefer_teacher_forcing_univlg9.yaml').read_text())
    result['training']['answer_pretrain_epochs']=3
    return result

class SmallLM(nn.Module):
    """Replace only expensive external base loading; exercise actual stage contracts."""
    def __init__(self,config):
        super().__init__()
        self.base=nn.Module(); self.base.config=SimpleNamespace(hidden_size=896,num_hidden_layers=24)
        self.base.point_backbone=nn.Linear(2,2).requires_grad_(False)
        self.base.point_proj=nn.Linear(2,2).requires_grad_(False)
        self.language=nn.Module(); self.language.lora_A=nn.Linear(2,2,bias=False)
        self.structured=nn.Embedding(4,2); self.lora_module_count=252
    def forward(self,cloud,questions,answers,*,visuals=None,return_context=True):
        hidden=self.language.lora_A(torch.ones(1,2))+self.structured(torch.tensor([0]))
        loss=hidden.square().mean()+sum(v.hidden.square().mean() for v in (visuals or []))
        return {'loss':loss,'metrics':{'token_accuracy':loss.detach()*0+1}}
    def encode_points(self,clouds):
        return [SimpleNamespace(hidden=torch.zeros(3,2)) for cloud in clouds]
    def optimizer_parameter_groups(self,*,lora_lr,structured_token_lr,weight_decay):
        return [{'name':'qwen_lora','params':list(self.language.parameters()),'lr':lora_lr,'weight_decay':weight_decay},{'name':'structured_tokens','params':list(self.structured.parameters()),'lr':structured_token_lr,'weight_decay':weight_decay}]
    def trainable_state_dict(self):
        return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if p.requires_grad}

def answer_stage(monkeypatch):
    module=stage_module(); monkeypatch.setattr(module,'CompressedVisualObjAncLM',SmallLM)
    return module.AnswerPretrainingStage(config())

def test_answer_stage_only_updates_language_and_structured_tokens(monkeypatch):
    model=answer_stage(monkeypatch)
    before={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    groups=model.optimizer_parameter_groups()
    assert {g['name'] for g in groups}=={'qwen_lora','structured_tokens'}
    assert all(g['lr']==(1e-4 if g['name'] in {'point_unet','visual_projector'} else 1e-5) for g in groups)
    optimizer=torch.optim.AdamW(groups)
    model({'point_cloud':[torch.zeros(1,12)],'question':['chair?'],'teacher_answer':['<obj>chair</obj>']},
          epoch=0,optimizer_step=0,steps_per_epoch=1,answer_mode='teacher')['loss'].backward()
    optimizer.step()
    assert all(torch.equal(dict(model.named_parameters())[n],value) for n,value in before.items())
    assert not any('decoder' in n or 'sonata_lora' in n for n,p in model.named_parameters() if p.requires_grad)
    assert set(model.trainable_state_dict())=={n for n,p in model.named_parameters() if p.requires_grad}

def test_stage_schedule_is_three_plus_fifty():
    import train_segmentation as runtime
    function=getattr(runtime,'stage_settings',None)
    assert callable(function),'missing stage-aware schedule'
    assert function(config(),'answer')['epochs']==3
    assert function(config(),'joint')['epochs']==50
    assert Path(function(config(),'answer')['output_dir']).name=='open_answer'

def checkpoint(monkeypatch,tmp_path):
    model=answer_stage(monkeypatch)
    module=stage_module()
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:p.fill_(.25)
    data=tmp_path/'data.json'; data.write_text('[]')
    model.config['data']['train_sources'][0]['annotations']=str(data)
    model.config['data']['val_annotations']=str(data)
    import train_segmentation as runtime
    payload={'objective':module.ANSWER_OBJECTIVE,'training_stage':'answer_pretraining','completed_epochs':3,
       'architecture_signature':model.architecture_signature(),'base_manifest':{'base':'sha'},
       'trainable_state':model.trainable_state_dict(),'config':model.config,'data_sha256':runtime.data_manifest(model.config)}
    path=tmp_path/'answer.pt'; torch.save(payload,path)
    return model,payload,path

def test_language_handoff_preserves_fresh_decoder_and_sonata(monkeypatch,tmp_path):
    answer,payload,path=checkpoint(monkeypatch,tmp_path)
    target=nn.Module(); target.config=copy.deepcopy(answer.config); target.answer_lm=SmallLM(target.config)
    target.answer_lm.base.point_backbone.register_parameter('sonata_lora_A',nn.Parameter(torch.ones(2)))
    target.decoder=nn.Linear(2,2); target.point_unet=nn.Linear(2,2);target.visual_adapter=nn.Linear(2,2)
    before={n:p.detach().clone() for n,p in target.named_parameters() if n.startswith('decoder.') or 'sonata_lora' in n}
    import train_segmentation as runtime
    load=getattr(runtime,'load_answer_checkpoint',None)
    assert callable(load),'missing Answer-to-joint handoff'
    parent=load(target,path,{'base':'sha'})
    assert parent['completed_epochs']==3 and parent['path']==str(path)
    assert all(torch.equal(dict(target.named_parameters())[n],value) for n,value in before.items())
    assert all(torch.equal(dict(target.named_parameters())[n],value) for n,value in payload['trainable_state'].items())

@pytest.mark.parametrize('field,value',[('completed_epochs',2),('objective','wrong'),('base_manifest',{'base':'wrong'})])
def test_handoff_rejects_wrong_or_incomplete_answer(monkeypatch,tmp_path,field,value):
    answer,payload,path=checkpoint(monkeypatch,tmp_path); payload[field]=value; torch.save(payload,path)
    target=nn.Module(); target.config=answer.config; target.answer_lm=SmallLM(target.config)
    import train_segmentation as runtime
    load=getattr(runtime,'load_answer_checkpoint',None)
    assert callable(load),'missing Answer-to-joint validation'
    before=target.answer_lm.structured.weight.detach().clone()
    with pytest.raises(ValueError):load(target,path,{'base':'sha'})
    assert torch.equal(before,target.answer_lm.structured.weight)

def test_answer_checkpoint_resumes_its_optimizer_and_schedule(monkeypatch,tmp_path):
    model,_,_=checkpoint(monkeypatch,tmp_path)
    import train_segmentation as runtime
    optimizer=torch.optim.AdamW(model.optimizer_parameter_groups())
    scheduler=runtime.scheduler_for(optimizer,3,.1)
    batch={'point_cloud':[torch.zeros(1,12)],'question':['chair?'],'teacher_answer':['<obj>chair</obj>']}
    model(batch)['loss'].backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step()
    expected=model(batch)['loss'].detach().clone()
    path=runtime.save_checkpoint(model,optimizer,scheduler,tmp_path,1,1,model.config,
                                manifest={'base':'sha'},frozen_hash='hash',stage_name='answer')
    with torch.no_grad():model.answer_lm.structured.weight.add_(10)
    assert runtime.load_resume(model,optimizer,scheduler,path,{'base':'sha'},stage_name='answer')==(1,1)
    assert torch.equal(model(batch)['loss'].detach(),expected)
    assert scheduler.last_epoch==1 and optimizer.state
    with pytest.raises(ValueError):runtime.load_resume(model,optimizer,scheduler,path,{'base':'sha'},stage_name='joint')

def test_joint_entry_requires_answer_handoff_or_joint_resume(monkeypatch):
    import sys
    import train_segmentation as runtime
    monkeypatch.setattr(sys,'argv',['train_segmentation.py',str(ROOT/'configs/scanrefer_teacher_forcing_univlg9.yaml')])
    with pytest.raises(SystemExit) as error:runtime.main()
    assert error.value.code==2

def test_handoff_rejects_bad_state_before_any_partial_mutation(monkeypatch,tmp_path):
    answer,payload,path=checkpoint(monkeypatch,tmp_path)
    key=next(iter(payload['trainable_state'])); payload['trainable_state'][key]=torch.zeros(7)
    torch.save(payload,path)
    target=nn.Module(); target.config=answer.config; target.answer_lm=SmallLM(target.config)
    before={n:p.detach().clone() for n,p in target.named_parameters()}
    import train_segmentation as runtime
    with pytest.raises(ValueError):runtime.load_answer_checkpoint(target,path,{'base':'sha'})
    assert all(torch.equal(p,before[n]) for n,p in target.named_parameters())
