"""Three ScanRefer-only epochs of causal Answer SFT before joint training."""
from __future__ import annotations
import torch
from torch import nn
from .llm import CompressedVisualObjAncLM
def visual_signature(config):
    return {"route":"original_native_enc4_frozen_projector", "answer_hidden":"own_token_causal_answer_only"}

ANSWER_OBJECTIVE='spatiallm_objanc_answer_only_causal_weighted_scanrefer_05b_answer_v1'

def answer_signature(language,config):
    return {'objective':ANSWER_OBJECTIVE,'language_hidden_size':int(language.base.config.hidden_size),
        'language_layers':int(language.base.config.num_hidden_layers),
        'qwen_lora_modules':language.lora_module_count,'qwen_lora':dict(config['model']['lora']),
        'answer_protocol':{key:config['model'][key] for key in
            ('system_prompt','point_prompt','user_prefix','structured_tokens','generation')},
        'visual_interface':visual_signature(config),
        'attention_mode':config['model']['attention_mode'],
        'enable_fourier_encode':config['model']['enable_fourier_encode']}

def is_answer_parameter(name):
    return name.startswith('answer_lm.structured.') or (
        name.startswith('answer_lm.') and 'lora_' in name and 'sonata_lora_' not in name
        and 'point_backbone' not in name)

class AnswerPretrainingStage(nn.Module):
    objective=ANSWER_OBJECTIVE

    def __init__(self,config):
        super().__init__()
        self.config=config; self.answer_lm=CompressedVisualObjAncLM(config)
        for name,parameter in self.named_parameters():
            if parameter.requires_grad and not is_answer_parameter(name):
                raise RuntimeError(f'Unexpected Answer-stage trainable parameter: {name}')

    def forward(self,batch,*,epoch=0,optimizer_step=0,steps_per_epoch=1,answer_mode='teacher'):
        if answer_mode!='teacher':raise ValueError('Answer pretraining is teacher-forced SFT only')
        rows=self.answer_lm.encode_points(batch["point_cloud"])
        output=self.answer_lm(batch['point_cloud'],batch['question'],batch['teacher_answer'],visuals=rows,return_context=True)
        loss=output['loss']
        if self.training:
            loss=loss+sum(p.reshape(-1)[0]*0 for p in self.parameters() if p.requires_grad)
        return {'loss':loss,'losses':{'total':loss.detach(),'answer_lm':output['loss'].detach()},
                'metrics':{'answer_token_accuracy':output['metrics']['token_accuracy'],
                'llm_visual_tokens':loss.new_tensor(sum(len(r.hidden) for r in rows)/len(rows)),
                'causal_forward_calls':loss.new_tensor(1),'replay_calls':loss.new_tensor(0)}}

    def architecture_signature(self):return answer_signature(self.answer_lm,self.config)

    def trainable_state_dict(self):
        from runtime_helpers import task_state
        return task_state(self)

    def optimizer_parameter_groups(self):
        t=self.config['training']
        return self.answer_lm.optimizer_parameter_groups(lora_lr=float(t['qwen_lora_lr']),structured_token_lr=float(t['structured_token_lr']),weight_decay=float(t['weight_decay']))
