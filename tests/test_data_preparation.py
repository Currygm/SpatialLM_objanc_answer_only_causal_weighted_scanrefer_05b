import importlib,json
from pathlib import Path
import numpy as np
import pytest

def api():
    assert importlib.util.find_spec("data_preparation") is not None, "Missing portable data preparation module"
    return importlib.import_module("data_preparation")

def raw():
    return dict(scene_id="scene0000_00",object_id="1",ann_id="0",description="chair near table",token=["chair","near","table"],entities=[[[0],["1_chair"]],[[2],["2_table"]]])

def test_answers_match_original_rules_and_missing_target(tmp_path):
    p=api()
    row=raw()
    missing={**row,"ann_id":"1","entities":[[[2],["2_table"]]]}
    report=p.prepare_answers([row,missing],[row],tmp_path)
    result=json.loads((tmp_path/"scanrefer_train.json").read_text())["annotations"]
    assert len(result)==1
    assert result[0]["grounding_object_ids"]==[1,2]
    answer=result[0]["teacher_answer"]
    assert answer=="The requested object is <obj>chair</obj>. The description states: chair near <anc>table</anc>"
    for entity in result[0]["answer_entities"]:
        a,b=entity["answer_char_span"]
        assert answer[a:b]==entity["phrase"]
    assert report["splits"]["train"]["dropped_count"]==1
    with pytest.raises(FileExistsError):
        p.prepare_answers([row],[row],tmp_path)

def test_repeated_ambiguous_anchors_and_duplicate_ids(tmp_path):
    p=api();row=raw();row["entities"].append([[2],["3_table"]])
    p.prepare_answers([row],[row],tmp_path)
    data=json.loads((tmp_path/"scanrefer_train.json").read_text())["annotations"][0]
    from spatiallm_compressed_visual_objanc_full_attention.data.segmentation_dataset import filter_answer_entities
    filtered=filter_answer_entities(data)
    assert filtered["grounding_roles"]==["OBJ"]
    with pytest.raises(ValueError,match="Duplicate"):
        p.prepare_answers([raw(),raw()],[raw()],tmp_path/"duplicates")

def scene(tmp_path):
    import torch
    root=tmp_path/"scannetv2";(root/"train").mkdir(parents=True)
    xyz=np.array([[x,y,0] for x in np.linspace(0,.12,5) for y in np.linspace(0,.12,5)],dtype=np.float32)
    path=root/"train/scene0000_00_refer.pth"
    torch.save((xyz,xyz.copy(),np.arange(len(xyz)),np.zeros(len(xyz)),np.ones(len(xyz))),path)
    return root,path,xyz

def test_normal_cache_contract_and_resume(tmp_path):
    p=api();root,path,xyz=scene(tmp_path);out=tmp_path/"normals"
    report=p.prepare_geometry(root,[{"scene_id":"scene0000_00"}],out,"normals")
    from spatiallm_compressed_visual_objanc_full_attention.data.normals import NormalCacheConfig,load_scene_normals,validate_normal_manifest
    cfg=NormalCacheConfig(enabled=True,cache_dir=str(out))
    validate_normal_manifest(cfg,[path],root)
    arr=load_scene_normals(path,root,cfg,len(xyz)).numpy()
    assert np.allclose(np.abs(arr[:,2]),1)
    assert report["scene_count"]==1
    stamp=(out/"train/scene0000_00_refer.npy").stat().st_mtime_ns
    p.prepare_geometry(root,[{"scene_id":"scene0000_00"}],out,"normals",resume=True)
    assert (out/"train/scene0000_00_refer.npy").stat().st_mtime_ns==stamp
    np.save(out/"train/scene0000_00_refer.npy",np.zeros((1,3)))
    p.prepare_geometry(root,[{"scene_id":"scene0000_00"}],out,"normals",resume=True)
    assert np.load(out/"train/scene0000_00_refer.npy").shape==xyz.shape

def test_superpoint_cache_exact_contract_and_partial_manifest(tmp_path):
    p=api();root,path,xyz=scene(tmp_path);out=tmp_path/"sp"
    p.prepare_geometry(root,[{"scene_id":"scene0000_00"}],out,"superpoints",limit=1)
    from spatiallm_compressed_visual_objanc_full_attention.data.superpoint import CachedSuperpointSegmenter,SuperpointConfig
    cfg=SuperpointConfig(enabled=True,cache_dir=str(out),require_cache=True)
    labels=CachedSuperpointSegmenter(cfg).segment(xyz)
    assert labels.shape==(len(xyz),)
    assert len(np.unique(labels))>=1
    report=p.prepare_geometry(root,[{"scene_id":"scene0000_00"}],out,"superpoints",resume=True)
    assert report["scene_count"]==1
    with pytest.raises(FileNotFoundError):
        p.prepare_geometry(root,[{"scene_id":"scene_missing"}],out,"normals")
    with pytest.raises(ValueError,match="scene_id"):
        p.prepare_geometry(root,[{"scene_id":"../escape"}],out,"normals")

def test_superpoint_manifest_passes_training_loader(tmp_path):
    from types import SimpleNamespace
    from spatiallm_compressed_visual_objanc_full_attention.data.dataset import ReasonSegDataset
    from spatiallm_compressed_visual_objanc_full_attention.data.superpoint import SuperpointConfig
    p=api();root,path,xyz=scene(tmp_path);out=tmp_path/"sp"
    p.prepare_geometry(root,[{"scene_id":"scene0000_00"}],out,"superpoints")
    consumer=SimpleNamespace(superpoint_config=SuperpointConfig(enabled=True,require_cache=True,cache_dir=str(out)),split="train",_unique_scene_paths=lambda:[path])
    ReasonSegDataset._validate_superpoint_manifest(consumer)

def test_superpoint_manifest_scene_key_keeps_split(tmp_path):
    from spatiallm_compressed_visual_objanc_full_attention.data.dataset import scene_manifest_key
    p=api();root,path,xyz=scene(tmp_path);out=tmp_path/"sp"
    p.prepare_geometry(root,[{"scene_id":"scene0000_00"}],out,"superpoints")
    manifest=json.loads((out/"manifest.json").read_text())
    assert scene_manifest_key(manifest["rows"][0]["scene_path"])==scene_manifest_key(path)

def test_interrupted_normals_resume_and_limit_preserve_records(tmp_path,monkeypatch):
    import torch
    import preparation.normal_estimation as estimator
    p=api();root,path,xyz=scene(tmp_path);out=tmp_path/"normals"
    second=root/"train/scene0001_00_refer.pth"
    torch.save(torch.load(path,weights_only=False),second)
    rows=[{"scene_id":"scene0000_00"},{"scene_id":"scene0001_00"}]
    original=estimator._estimate_normals;calls=[]
    def fail_second(*args,**kwargs):
        calls.append(1)
        if len(calls)==2:raise RuntimeError("simulated interruption")
        return original(*args,**kwargs)
    monkeypatch.setattr(estimator,"_estimate_normals",fail_second)
    with pytest.raises(RuntimeError,match="interruption"):p.prepare_geometry(root,rows,out,"normals")
    assert (out/"manifest.json").is_file()
    assert json.loads((out/"manifest.json").read_text())["scene_count"]==1
    stamp=(out/"train/scene0000_00_refer.npy").stat().st_mtime_ns
    monkeypatch.setattr(estimator,"_estimate_normals",original)
    p.prepare_geometry(root,rows,out,"normals",resume=True)
    assert (out/"train/scene0000_00_refer.npy").stat().st_mtime_ns==stamp
    p.prepare_geometry(root,rows,out,"normals",resume=True,limit=1)
    assert json.loads((out/"manifest.json").read_text())["scene_count"]==2
