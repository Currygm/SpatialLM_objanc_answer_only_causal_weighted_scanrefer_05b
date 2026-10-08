import torch
from causal_runtime import capture_rng,restore_rng

def test_rng_resume_next_random_values():
 import random,numpy as np
 state=capture_rng();a=(random.random(),float(np.random.rand()),torch.rand(4))
 restore_rng(state);b=(random.random(),float(np.random.rand()),torch.rand(4))
 assert a[:2]==b[:2];assert torch.equal(a[2],b[2])

def test_selected_rank_cuda_rng_only(monkeypatch):
 monkeypatch.setattr(torch.cuda,'is_initialized',lambda:True)
 monkeypatch.setattr(torch.cuda,'current_device',lambda:2)
 monkeypatch.setattr(torch.cuda,'get_rng_state',lambda device=None:torch.tensor([17],dtype=torch.uint8))
 monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:(_ for _ in ()).throw(AssertionError('Do not touch other GPUs')))
 assert capture_rng()['cuda']['device']==2

class StochasticDataset(torch.utils.data.Dataset):
 def __len__(self):return 8
 def __getitem__(self,index):
  import numpy as np
  return index,int(np.random.randint(0,100000000))

def test_epoch_boundary_resume_worker_inputs_identical(monkeypatch):
 import train_segmentation as runtime
 monkeypatch.setattr(runtime,'collate_samples',lambda rows:rows)
 config={'data':{'num_workers':2,'batch_size':1}}
 torch.manual_seed(73)
 dataset=StochasticDataset();sampler=torch.utils.data.DistributedSampler(dataset,num_replicas=1,rank=0,seed=42)
 loader=runtime.make_loader(dataset,sampler,config,torch.device('cpu'))
 assert not loader.persistent_workers
 sampler.set_epoch(0);list(loader);saved=capture_rng()
 sampler.set_epoch(1);expected=list(loader)
 restore_rng(saved)
 sampler2=torch.utils.data.DistributedSampler(dataset,num_replicas=1,rank=0,seed=42);sampler2.set_epoch(1)
 resumed=runtime.make_loader(dataset,sampler2,config,torch.device('cpu'))
 assert list(resumed)==expected
