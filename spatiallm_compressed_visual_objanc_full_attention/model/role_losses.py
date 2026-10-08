"""Independent OBJ/ANC means and separately retained unmatched-query focal."""
import torch
import torch.nn.functional as F

def role_mean(values, roles):
 if len(values)!=len(roles):raise ValueError('Role/value lengths differ')
 zero=values.sum()*0
 means={r:values[torch.tensor([x==r for x in roles],device=values.device)].mean() if r in roles else zero for r in ('OBJ','ANC')}
 return 1.5*means['OBJ']+.5*means['ANC'],means

def role_focal(logits,positive,text_valid,assignments,roles,alpha=.25,gamma=2.):
 target=torch.zeros_like(logits);group={r:[] for r in ('OBJ','ANC')};background=[];allpos=logits.new_zeros(())
 for b,(q,t) in enumerate(assignments):
  target[b,q]=positive[b,t].to(logits);allpos=allpos+target[b,q].sum()
 prob=logits.float().sigmoid();truth=target.float()
 raw=F.binary_cross_entropy_with_logits(logits.float(),truth,reduction='none')
 focal=raw*(1-(prob*truth+(1-prob)*(1-truth))).pow(gamma)*(alpha*truth+(1-alpha)*(1-truth))
 focal=focal*text_valid[:,None].to(focal)
 for b,(q,t) in enumerate(assignments):
  for r in group:
   pick=torch.tensor([roles[b][int(i)]==r for i in t],device=q.device,dtype=torch.bool)
   if pick.any():group[r].append((focal[b,q[pick]].sum(),target[b,q[pick]].sum()))
  unmatched=torch.ones(logits.shape[1],device=q.device,dtype=torch.bool);unmatched[q]=False
  background.append(focal[b,unmatched].sum())
 zero=logits.sum()*0
 means={r:sum(x[0] for x in group[r])/sum(x[1] for x in group[r]).clamp_min(1) if group[r] else zero for r in group}
 bg=sum(background)/allpos.clamp_min(1)
 details={'span_obj':means['OBJ'],'span_anc':means['ANC'],'span_obj_weighted':1.5*means['OBJ'],'span_anc_weighted':.5*means['ANC'],'span_background':bg}
 return details['span_obj_weighted']+details['span_anc_weighted']+bg,details
