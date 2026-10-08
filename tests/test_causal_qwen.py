from types import SimpleNamespace
import torch
from torch import nn
from transformers import Qwen2Config,Qwen2ForCausalLM,PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from spatiallm_compressed_visual_objanc_full_attention.model.llm import CompressedVisualObjAncLM,StructuredTokenRows,TXYZRotaryEmbedding,attach_language_lora
from spatiallm_compressed_visual_objanc_full_attention.model.backbone import CompressedVisualTokens

def tiny_language():
 torch.manual_seed(11)
 vocab={'[UNK]':0,'[EOS]':1,'<|point_start|>':2,'<|point_pad|>':3,'<|point_end|>':4,'Find':5,'chair':6,'beside':7,'table':8,'near':9,'lamp':10,'.':11,'Answer':12,'the':13,'red':14,'blue':15}
 for i in range(16,40):vocab['unused'+str(i)]=i
 backend=Tokenizer(WordLevel(vocab,unk_token='[UNK]'));backend.pre_tokenizer=Whitespace()
 tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]',eos_token='[EOS]',pad_token='[UNK]')
 tokenizer.add_special_tokens({'additional_special_tokens':['<|point_start|>','<|point_pad|>','<|point_end|>','<obj>','</obj>','<anc>','</anc>']})
 dense=Qwen2ForCausalLM(Qwen2Config(vocab_size=40,hidden_size=32,intermediate_size=48,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,_attn_implementation='sdpa'))
 dense.point_backbone=nn.Linear(2,2);dense.point_proj=nn.Linear(2,2)
 dense.requires_grad_(False);dense.config.point_start_token_id=2;dense.config.point_end_token_id=4
 dense.model.rotary_emb=TXYZRotaryEmbedding(8,10000.)
 language,count=attach_language_lora(dense,{'r':4,'alpha':8,'dropout':0.,'bias':'none','target_modules':['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']})
 model=CompressedVisualObjAncLM.__new__(CompressedVisualObjAncLM);nn.Module.__init__(model)
 model.spatiallm=language;model.base=language.get_base_model();model.base.get_model=lambda:model.base.model
 model.tokenizer=tokenizer;model.structured_token_ids={t:tokenizer.convert_tokens_to_ids(t) for t in ['<obj>','</obj>','<anc>','</anc>']}
 model.structured=StructuredTokenRows(32,list(model.structured_token_ids.values()),40);model.base_vocab_size=40;model.eos_token_id=1
 model._render_prompt=lambda q:'<|point_start|><|point_pad|><|point_end|> '+q+' Answer'
 model.generation_config={'max_phrase_tokens':32,'max_new_tokens':128};model.lora_module_count=count
 model.eval();return model

def visuals(n=4):
 x=torch.randn(n,32,requires_grad=True);xyz=torch.randn(n,3)
 return CompressedVisualTokens(x,xyz,xyz/.025)

def test_one_causal_call_and_spans_padding(monkeypatch):
 model=tiny_language();rows=[visuals(4),visuals(7)];calls=[]
 original=model.base.model.forward
 def counted(*a,**kw):calls.append(kw);return original(*a,**kw)
 monkeypatch.setattr(model.base.model,'forward',counted)
 answers=['the <obj>red chair</obj> beside <anc>table</anc> near <anc>blue lamp</anc>.','<obj>chair</obj>.']
 result=model([None,None],['Find chair','Find red chair'],answers,visuals=rows,return_context=True)
 assert len(calls)==1 and calls[0]['attention_mask'].ndim==2
 context=result['context'];assert context.entity_roles==[('OBJ','ANC','ANC'),('OBJ',)]
 for i,answer in enumerate(answers):
  assert len(context.answer_hidden[i])==len(model.tokenizer.encode(answer,add_special_tokens=False))
  assert context.phrase_spans[i].shape==(len(context.entity_roles[i]),len(context.answer_hidden[i]))
  assert context.phrase_spans[i].sum(1).min()>0
 result['loss'].backward()
 assert all(row.hidden.grad is not None and row.hidden.grad.abs().sum()>0 for row in rows)
 assert model.structured.rows.weight.grad.abs().sum()>0
 assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.spatiallm.named_parameters() if 'lora_B' in n)
 assert all(p.grad is None for n,p in model.spatiallm.named_parameters() if 'lora_' not in n)

def test_causal_question_does_not_read_future_answer():
 model=tiny_language();v=visuals();a='<obj>red chair</obj> beside <anc>table</anc>.';b='<obj>blue lamp</obj> near <anc>chair</anc>.'
 x=model([None],['Find chair'],[a],visuals=[v])['context'];y=model([None],['Find chair'],[b],visuals=[v])['context']
 torch.testing.assert_close(x.answer_hidden[0][0],y.answer_hidden[0][0],atol=1e-6,rtol=1e-5)
 assert not hasattr(x,'question_hidden')
 replay=model.encode_causal_context([None],['Find chair'],[a],visuals=[v])
 torch.testing.assert_close(x.answer_hidden[0],replay.answer_hidden[0])

def test_cached_prefix_hidden_matches_causal_replay():
 model=tiny_language();v=visuals();question='Find chair';answer='<obj>red chair</obj> beside <anc>table</anc>.'
 prompt,_=model._prompt_tokens(question);ids=model.tokenizer.encode(answer,add_special_tokens=False)
 embedded,positions,_,_=model._merge(prompt,v);valid=torch.ones(1,len(embedded),dtype=torch.long)
 out=model.base.get_model()(inputs_embeds=embedded[None],attention_mask=valid,position_ids=positions.T[:,None],use_cache=True,return_dict=True)
 prefix=[];textpos=int(positions[:,0].max())+1
 for token in ids:
  valid=torch.cat([valid,valid.new_ones(1,1)],1)
  out=model.base.get_model()(inputs_embeds=model.structured.embed(torch.tensor([[token]]),model.spatiallm.get_input_embeddings()),attention_mask=valid,position_ids=torch.tensor([[[textpos]],[[0.]],[[0.]],[[0.]]]),past_key_values=out.past_key_values,use_cache=True,return_dict=True)
  prefix.append(out.last_hidden_state[0,0]);textpos+=1
 replay=model.encode_causal_context([None],[question],[answer],visuals=[v])
 torch.testing.assert_close(torch.stack(prefix),replay.answer_hidden[0],atol=1e-5,rtol=1e-4)

def test_tokenizer_snapshot_save_reload(tmp_path):
 model=tiny_language();before=model.tokenizer_snapshot();model.tokenizer.save_pretrained(tmp_path)
 loaded=PreTrainedTokenizerFast.from_pretrained(tmp_path,local_files_only=True)
 model.tokenizer=loaded
 assert model.tokenizer_snapshot()==before
 for t,i in model.structured_token_ids.items():assert loaded.encode(t,add_special_tokens=False)==[i]

def test_answer_pack_excludes_extra_eos_prompt_and_padding():
 model=tiny_language();answers=['<obj>chair</obj>.','the <obj>red chair</obj> beside <anc>table</anc> near <anc>lamp</anc>.']
 rows=[visuals(7),visuals(2)];questions=['Find chair','Find red chair']
 embeds,valid,positions,labels,maps,parsed=model.pack_causal_batch([None,None],questions,answers,rows)
 result=model([None,None],questions,answers,visuals=rows)
 for i,answer in enumerate(answers):
  ids=model.tokenizer.encode(answer,add_special_tokens=False);a_idx=maps[i][1]
  assert labels[i,valid[i].bool()].ne(-100).sum()==len(ids)+1
  assert len(a_idx)==len(ids) and int(a_idx[-1])+1==int(valid[i].sum())-1
  assert int(labels[i,int(valid[i].sum())-1])==model.eos_token_id
  assert all(int(x)!=0 for x in result['context'].token_roles[i])
  assert result['context'].phrase_spans[i].shape[1]==len(ids)
  assert bool(labels[i,~valid[i].bool()].eq(-100).all())
 assert model.structured.rows.weight is not model.structured.output_rows.weight
