"""Portable CPU preparation for the three ScanRefer training resources."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

def read_rows(path: Path) -> list[dict[str, Any]]:
    value=json.loads(Path(path).read_text(encoding="utf-8"))
    rows=value.get("annotations") if isinstance(value,dict) else value
    if not isinstance(rows,list) or not all(isinstance(row,dict) for row in rows):
        raise ValueError(f"Expected annotation list in {path}")
    return rows

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix="."+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as handle:
            json.dump(value,handle,ensure_ascii=False,indent=2)
            handle.write("\n")
        os.replace(tmp,path)
    finally:
        Path(tmp).unlink(missing_ok=True)

def prepare_answers(train: list[dict[str,Any]], val: list[dict[str,Any]],
                    output_dir: Path, *, overwrite: bool=False) -> dict[str,Any]:
    from preparation.scanents_mapping import derive_split
    from preparation.objanc_rules import prepare_rows,FORMAT_VERSION
    out=Path(output_dir).resolve()
    names=["scanrefer_train.json","scanrefer_val.json","manifest.json","failures.json"]
    if not overwrite:
        for name in names:
            if (out/name).exists(): raise FileExistsError(out/name)
    outputs={};summaries={};failures=[]
    for split,raw in [("train",train),("val",val)]:
        # Raw official ScanEnts3D entities and space-joined token coordinates.
        derived,dropped,mapping=derive_split(raw,split)
        for row in derived: row.setdefault("dataset","scanrefer")
        prepared,stats=prepare_rows(derived,"scanrefer" if split=="train" else "scanrefer_val")
        outputs[split]={"annotations":prepared,"metadata":stats}
        summaries[split]={**mapping,"answer_statistics":stats}
        failures.extend(dropped)
    out.mkdir(parents=True,exist_ok=True)
    for split,payload in outputs.items():write_json(out/f"scanrefer_{split}.json",payload)
    write_json(out/"failures.json",failures)
    report={"format_version":FORMAT_VERSION,"splits":summaries,
            "method":"original ScanEnts3D mapping and OBJ-first rules; no external LLM"}
    write_json(out/"manifest.json",report)
    return report

def scene_paths(root: Path, rows: list[dict[str,Any]]) -> list[Path]:
    root=root.resolve(strict=True);paths=set()
    for row in rows:
        scene=str(row["scene_id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+",scene):raise ValueError(f"Invalid scene_id: {scene}")
        matches=[]
        for sub in ["train","val",""]:
            for suffix in ["_refer.pth",".pth"]:
                path=root/sub/(scene+suffix)
                if path.is_file():
                    resolved=path.resolve(strict=True)
                    if not resolved.is_relative_to(root):raise ValueError(f"Scene escapes root: {path}")
                    matches.append(resolved)
        if not matches:raise FileNotFoundError(f"No scene file for {scene} under {root}")
        if len(set(matches))!=1:raise ValueError(f"Ambiguous scene files for {scene}: {matches}")
        paths.add(matches[0])
    if not paths:raise ValueError("No scenes selected")
    return sorted(paths)

def load_xyz(path: Path):
    import numpy as np
    import torch
    # Only load trusted local .pth files; torch pickle can execute code.
    raw=torch.load(path,map_location="cpu",weights_only=False)
    if isinstance(raw,dict):
        xyz=next((raw[k] for k in ["xyz","coord","coords","coordinates","points"] if k in raw),None)
        if xyz is None:raise ValueError(f"No xyz in {path}")
    elif isinstance(raw,(tuple,list)) and len(raw)==5:xyz=raw[0]
    else:raise ValueError(f"Expected Reason3D five-array scene or xyz dictionary: {path}")
    xyz=np.ascontiguousarray(xyz,dtype=np.float32)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError(f"Invalid coordinates in {path}: {xyz.shape}")
    return xyz

def valid_normals(path: Path, count: int) -> bool:
    import numpy as np
    try:
        arr=np.load(path,allow_pickle=False)
        return arr.shape==(count,3) and bool(np.isfinite(arr).all()) and bool((np.linalg.norm(arr,axis=1)>1e-6).all())
    except (OSError,ValueError):return False

def prepare_geometry(scene_root: Path, rows: list[dict[str,Any]], output_dir: Path,
                     kind: str, *, resume: bool=False, limit: int | None=None) -> dict[str,Any]:
    import numpy as np
    from preparation.normal_estimation import _estimate_normals
    from spatiallm_compressed_visual_objanc_full_attention.data.normals import NORMAL_CACHE_VERSION
    from spatiallm_compressed_visual_objanc_full_attention.data.superpoint import CachedSuperpointSegmenter,SuperpointConfig
    if kind not in {"normals","superpoints"}:raise ValueError(f"Unknown cache kind: {kind}")
    root=Path(scene_root).resolve(strict=True);out=Path(output_dir).resolve()
    paths=scene_paths(root,rows)
    if limit is not None:
        if limit<1:raise ValueError("--limit must be positive")
        selected=paths[:limit]
    else:selected=paths
    out.mkdir(parents=True,exist_ok=True)
    cfg=SuperpointConfig(enabled=True,cache_dir=str(out),require_cache=False)
    segmenter=CachedSuperpointSegmenter(cfg)
    version=NORMAL_CACHE_VERSION if kind=="normals" else cfg.version
    parameters={"radius":.1,"max_nn":30} if kind=="normals" else {k:getattr(cfg,k) for k in ["graph_k","normal_k","edge_radius_scale","k_thresh","min_vertices"]}
    prior={}
    try:
        old=json.loads((out/"manifest.json").read_text(encoding="utf-8"))
        if old.get("version")==version and old.get("parameters")==parameters:
            prior={row["scene_path"]:row for row in old["rows"]}
    except (OSError,ValueError,KeyError,TypeError):pass
    records={}
    if resume:
        for path in paths:
            identity=str(path.relative_to(root)) if kind=="normals" else str(path)
            previous=prior.get(identity)
            if not previous:continue
            target=(out/path.relative_to(root)).with_suffix(".npy") if kind=="normals" else Path(previous["cache_path"])
            if target.resolve().is_relative_to(out) and target.is_file():records[identity]=previous
    def checkpoint():
        report={"version":version,"parameters":parameters,"scene_root":str(root),
                "scene_count":len(records),"total_required_scenes":len(paths),
                "complete":len(records)==len(paths) and len(selected)==len(paths),
                "rows":[records[key] for key in sorted(records)]}
        write_json(out/"manifest.json",report)
        return report
    for i,path in enumerate(selected,1):
        xyz=load_xyz(path);relative=path.relative_to(root)
        digest=hashlib.sha256(xyz.tobytes()).hexdigest();status="generated"
        if kind=="normals":
            dest=(out/relative).with_suffix(".npy")
            cached=prior.get(str(relative),{})
            if resume and cached.get("coordinate_sha256")==digest and valid_normals(dest,len(xyz)):
                status="cached"
            else:
                normals=_estimate_normals(xyz,"cpu",.1,30)
                if not np.isfinite(normals).all() or np.any(np.linalg.norm(normals,axis=1)<=1e-6):
                    raise ValueError(f"Invalid computed normals: {path}")
                dest.parent.mkdir(parents=True,exist_ok=True)
                fd,tmp=tempfile.mkstemp(prefix="."+dest.name,dir=dest.parent)
                try:
                    with os.fdopen(fd,"wb") as handle:np.save(handle,normals)
                    os.replace(tmp,dest)
                finally:Path(tmp).unlink(missing_ok=True)
        else:
            dest=segmenter.cache_path(xyz)
            # segment() validates coordinate digest, parameters and metadata before reuse.
            if not resume and dest.exists():dest.unlink()
            existed=dest.exists()
            segmenter.segment(xyz,source=str(path))
            status="cached_or_repaired" if existed else "generated"
        identity=str(relative) if kind=="normals" else str(path)
        records[identity]={"scene_path":identity,"points":len(xyz),
                           "cache_path":str(dest),"coordinate_sha256":digest}
        checkpoint()
        print(f"[{i}/{len(selected)}] {kind} {status} {relative}",flush=True)
    return checkpoint()

def main(argv=None) -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="kind",required=True)
    answers=sub.add_parser("answers")
    answers.add_argument("--archive",type=Path,required=True)
    answers.add_argument("--output-dir",type=Path,default=Path("data/objanc_teacher_answers_v1"))
    answers.add_argument("--overwrite",action="store_true")
    answers.add_argument("--expect-train",type=int,default=36569)
    answers.add_argument("--expect-val",type=int,default=9483)
    for kind,folder in [("normals","scanrefer_normals"),("superpoints","scanrefer_reason3d_superpoint")]:
        geometry=sub.add_parser(kind)
        geometry.add_argument("--scene-root",type=Path,required=True)
        geometry.add_argument("--annotations",type=Path,nargs="+",required=True)
        geometry.add_argument("--output-dir",type=Path,default=Path("cache")/folder)
        geometry.add_argument("--resume",action="store_true")
        geometry.add_argument("--limit",type=int)
    args=parser.parse_args(argv)
    if args.kind=="answers":
        from preparation.scanents_mapping import sha256_file,OFFICIAL_ARCHIVE_SHA256,SOURCE_FILES,derive_split
        actual=sha256_file(args.archive)
        if actual!=OFFICIAL_ARCHIVE_SHA256:raise ValueError(f"Archive SHA256 mismatch: {actual}")
        with zipfile.ZipFile(args.archive) as archive:
            payloads={split:json.loads(archive.read(member)) for split,(member,count) in SOURCE_FILES.items()}
        for split,(member,count) in SOURCE_FILES.items():
            if len(payloads[split])!=count:raise ValueError(f"Unexpected official {split} count")
            retained,_,_=derive_split(payloads[split],split)
            expected=args.expect_train if split=="train" else args.expect_val
            if len(retained)!=expected:raise ValueError(f"Expected {expected} {split}, got {len(retained)}")
        report=prepare_answers(payloads["train"],payloads["val"],args.output_dir,overwrite=args.overwrite)
    else:
        rows=[row for path in args.annotations for row in read_rows(path)]
        report=prepare_geometry(args.scene_root,rows,args.output_dir,args.kind,resume=args.resume,limit=args.limit)
    print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2))

if __name__=="__main__":main()
