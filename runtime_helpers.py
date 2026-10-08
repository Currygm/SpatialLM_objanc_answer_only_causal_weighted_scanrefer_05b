"""CPU provenance, atomic persistence and nonblocking training logs."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import torch

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def checkpoint_manifest(path):
    root=Path(path).resolve(strict=True)
    index=root/'model.safetensors.index.json'
    names=(set(json.loads(index.read_text())['weight_map'].values()) | {'config.json',index.name}) if index.exists() else {'config.json','model.safetensors'}
    names.update(p.name for p in root.glob('*token*.json'))
    names.update(p.name for p in root.glob('*.txt') if p.name=='merges.txt')
    names.update(p.name for p in root.glob('vocab.json'))
    for name in names:
        if Path(name).name!=name:raise ValueError('Shard filename escapes base directory')
    return {name:sha256(root/name) for name in sorted(names)}

def frozen_digest(model):
    digest=hashlib.sha256()
    for name,parameter in model.named_parameters():
        if parameter.requires_grad:continue
        tensor=parameter.detach().cpu().contiguous()
        digest.update(f'{name}:{tensor.dtype}:{tuple(tensor.shape)}'.encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

def task_state(model):
    values={n:p for n,p in model.named_parameters() if p.requires_grad}
    values.update({n:b for n,b in model.named_buffers() if 'point_backbone' in n})
    return {n:v.detach().cpu().clone() for n,v in values.items()}

def load_trainable_state(model,state):
    expected=task_state(model)
    checked_copy(model,state,expected)

def checked_copy(model,state,expected):
    if not isinstance(state,dict) or set(state)!=set(expected):raise ValueError('Checkpoint task-state keys differ')
    destination={**dict(model.named_parameters()),**dict(model.named_buffers())}
    for n in expected:
        if not isinstance(state[n],torch.Tensor) or state[n].shape!=destination[n].shape or not torch.isfinite(state[n]).all():
            raise ValueError('Checkpoint task-state shapes or values differ')
    with torch.no_grad():
        for n in expected:destination[n].copy_(state[n])


def validate_checkpoint(checkpoint,objective,signature,manifest):
    if checkpoint.get('objective')!=objective or checkpoint.get('architecture_signature')!=signature:
        raise ValueError('Checkpoint objective or architecture differs')
    if checkpoint.get('base_manifest')!=manifest:
        raise ValueError('Frozen base checkpoint SHA256 manifest differs')

def atomic_save(payload,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    try:
        torch.save(payload,temporary)
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()

class AsyncJsonLog:
    """Stalled filesystem never stalls training; overflow disables logging."""
    def __init__(self,path):
        self.path=Path(path); self.disabled=False
        self.queue=queue.Queue(maxsize=128)
        self.worker=threading.Thread(target=self._run,daemon=True)
        self.worker.start()

    def submit(self,record):
        if self.disabled:return
        try:self.queue.put_nowait(json.dumps(record,ensure_ascii=False))
        except queue.Full:self.disabled=True

    def _run(self):
        try:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            with self.path.open('a',encoding='utf-8') as handle:
                while True:
                    item=self.queue.get()
                    if item is None:return
                    handle.write(item+'\n'); handle.flush()
        except OSError as exc:
            self.disabled=True
            print(f'Async metric log disabled: {exc}',flush=True)

    def close(self,timeout=1):
        try:self.queue.put_nowait(None)
        except queue.Full:self.disabled=True
        self.worker.join(timeout=timeout)
