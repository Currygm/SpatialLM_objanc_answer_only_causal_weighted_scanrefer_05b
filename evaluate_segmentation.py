"""Teacher-vs-generated Answer segmentation evaluation."""
from __future__ import annotations

import argparse
import json
import os
import hashlib
import random
from datetime import timedelta
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from spatiallm_compressed_visual_objanc_full_attention.config import ensure_within, load_segmentation_config
from spatiallm_compressed_visual_objanc_full_attention.data import collate_samples
from spatiallm_compressed_visual_objanc_full_attention.data.segmentation_dataset import build_segmentation_dataset
from spatiallm_compressed_visual_objanc_full_attention.model.answer_format import parse_answer
from spatiallm_compressed_visual_objanc_full_attention.model.curriculum import normalize_phrase
from spatiallm_compressed_visual_objanc_full_attention.model.segmentation_model import OBJECTIVE, ObjAncSegmentationModel
from runtime_helpers import checkpoint_manifest, load_trainable_state, validate_checkpoint


def load_checkpoint(model, path: Path, manifest):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint(checkpoint, OBJECTIVE, model.architecture_signature(), manifest)
    from train_segmentation import check_tokenizer
    check_tokenizer(model,checkpoint)
    load_trainable_state(model, checkpoint.get('trainable_state'))
    return checkpoint


def mask_iou(logits, truth, threshold):
    prediction = logits.sigmoid() >= threshold
    truth = truth.to(prediction.device, torch.bool)
    return float(((prediction & truth).sum().float() / (prediction | truth).sum().clamp_min(1).float()).item())


class MetricBook:
    def __init__(self):
        self.sums = defaultdict(float)
        self.counts = Counter()

    def _add_value(self, key, value):
        self.sums[key] += float(value)
        self.counts[key] += 1

    def add(self, role, selected, matched, oracle, selected_q, matched_q, best_q, rank, topk, *, segmentation=None):
        for prefix in ("all", role.lower()):
            self._add_value(f"{prefix}/selected_iou", selected)
            self._add_value(f"{prefix}/matched_iou", matched)
            self._add_value(f"{prefix}/oracle_iou", oracle)
            self._add_value(f"{prefix}/best_query_rank", rank)
            self._add_value(f"{prefix}/qmatch", float(selected_q == matched_q))
            self._add_value(f"{prefix}/best_iou_query_recall", float(selected_q == best_q))
            self._add_value(f"{prefix}/best_query_exact_top1", float(selected_q == best_q))
            for k, value in topk.items():
                self._add_value(f"{prefix}/top{k}_oracle_iou", value)
                for threshold in (.25,.5):self._add_value(f'{prefix}/recall{int(threshold*100)}_top{k}',value>=threshold)
            for key,value in (segmentation or {}).items():self._add_value(f'{prefix}/{key}',value)
            for level in (.25, .5):
                self._add_value(f"{prefix}/selected_acc@{level:.2f}", float(selected >= level))
                self._add_value(f"{prefix}/oracle_acc@{level:.2f}", float(oracle >= level))

    def summary(self):
        return {
            key: self.sums[key] / self.counts[key]
            for key in sorted(self.sums)
            if self.counts[key]
        }

    def state_dict(self):
        return {"sums": dict(self.sums), "counts": dict(self.counts)}

    def merge(self, state):
        for key, value in state["sums"].items():
            self.sums[key] += float(value)
        self.counts.update({key: int(value) for key, value in state["counts"].items()})


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        sample["evaluation_index"] = int(index)
        return sample


class ExactDistributedEvalSampler(Sampler[int]):
    """Shard evaluation without padding or duplicating validation samples."""

    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.world_size - 1) // self.world_size)


def distributed_context():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if torch.cuda.is_available():torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",timeout=timedelta(hours=2))
    return rank, world_size, local_rank, distributed


def measure_result(book, batch, result, threshold):
    decoder=result['decoder_output']
    for b,roles in enumerate(result['entity_roles']):
        mapping=batch['point_to_superpoint'][b].to(decoder['pred_masks'].device,torch.long)
        truths=batch['gt_point_masks'][b][result['target_rows'][b]]
        masks=decoder['pred_masks'][b,:,:batch['gt_superpoint_masks'][b].shape[1]][:,mapping]
        scores=result['query_entity_scores'][b,:,:len(roles)]
        source,target=result['assignments'][b]
        matched_by_target={int(t):int(q) for q,t in zip(source,target)}
        for e,(role,truth) in enumerate(zip(roles,truths)):
            ious=[mask_iou(m,truth,threshold) for m in masks]
            order=torch.argsort(scores[:,e],descending=True).tolist()
            selected,best,matched=order[0],max(range(len(ious)),key=ious.__getitem__),matched_by_target[e]
            prediction=masks[selected].sigmoid()>=threshold; gt=truth.to(prediction.device,torch.bool)
            intersection=(prediction & gt).sum().float()
            segmentation={'precision':float(intersection/prediction.sum().clamp_min(1)),
                'recall':float(intersection/gt.sum().clamp_min(1)),
                'dice':float(2*intersection/(prediction.sum()+gt.sum()).clamp_min(1))}
            for query_kind,q_index in [('matched',matched),('oracle',best)]:
                pred=masks[q_index].sigmoid()>=threshold; inter=(pred & gt).sum().float()
                segmentation.update({query_kind+'/precision':float(inter/pred.sum().clamp_min(1)),query_kind+'/recall':float(inter/gt.sum().clamp_min(1)),query_kind+'/dice':float(2*inter/(pred.sum()+gt.sum()).clamp_min(1))})
            topk={k:max(ious[q] for q in order[:k]) for k in (1,3,5,10,50)}
            book.add(role,ious[selected],ious[matched],ious[best],selected,matched,best,order.index(best)+1,topk,segmentation=segmentation)

def evaluate_mode(model, loader, mode, threshold, examples_limit, *, use_amp):
    book, examples = MetricBook(), []
    raw_book, raw_counts = MetricBook(), Counter()
    answer_counts, fallback = Counter(), Counter()
    visual_counts=[];full_counts=[];calls=Counter()
    if torch.cuda.is_initialized():torch.cuda.reset_peak_memory_stats()
    for batch in loader:
        seed=int(model.config['training']['seed'])+int.from_bytes(hashlib.sha256(str(batch['sample_id'][0]).encode()).digest()[:4],'little')
        random.seed(seed); np.random.seed(seed%2**32); torch.manual_seed(seed)
        device_type = next(model.parameters()).device.type
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=use_amp and device_type == "cuda",
        ):
            result = model(
                batch,
                epoch=0,
                optimizer_step=0,
                steps_per_epoch=1,
                answer_mode=mode,
            )
        measure_result(book,batch,result,threshold)
        answer_counts["missing_anc_mask_count"]+=int(result["metrics"].get("missing_anc_mask_count",0))
        answer_counts["samples_with_missing_anc_masks"]+=int(float(result["metrics"].get("missing_anc_mask_count",0))>0)
        visual_counts.append(float(result['metrics']['llm_visual_tokens']));full_counts.append(float(result['metrics']['full_superpoint_count']))
        for key in ['causal_forward_calls','replay_calls','generation_prefill_calls','generation_incremental_calls']:
            calls[key]+=int(result['metrics'][key])
        if mode=='generated':
            for b,source in enumerate(result['answer_sources']):
                raw_counts['attempted']+=1
                raw_counts['missing_anc_mask_count']+=int(batch.get('dropped_missing_entity_count',[0]*len(batch['question']))[b])
                single_batch={key:[value[b]] for key,value in batch.items()}
                if source=='generated':
                    if len(batch['question'])!=1:raise ValueError('Evaluation requires batch size 1')
                    measure_result(raw_book,batch,result,threshold); raw_counts['evaluable']+=1
                else:
                    answer=result['attempted_generated_answers'][b]
                    parsed=parse_answer(answer)
                    if not parsed.valid or sum(e.role=='OBJ' for e in parsed.entities)!=1:
                        raw_counts['format_unevaluable']+=1; continue
                    with torch.autocast(device_type=device_type,dtype=torch.bfloat16,enabled=use_amp and device_type=='cuda'):
                        raw_result=model(single_batch,epoch=0,optimizer_step=0,steps_per_epoch=1,
                                         answer_mode='raw_generated',generated_answers=[answer])
                    calls['raw_diagnostic_replay_calls']+=1
                    measure_result(raw_book,single_batch,raw_result,threshold); raw_counts['evaluable']+=1
        for source in result["answer_sources"]: fallback[source] += 1
        measured_answers = [
            attempted if mode == "generated" and attempted is not None else chosen
            for attempted, chosen in zip(result["attempted_generated_answers"], result["chosen_answers"])
        ]
        for teacher, answer in zip(batch["teacher_answer"], measured_answers):
            predicted, target = parse_answer(answer), parse_answer(teacher)
            answer_counts["samples"] += 1
            answer_counts["format_valid"] += int(predicted.valid)
            p_obj = next((x for x in predicted.entities if x.role == "OBJ"), None)
            t_obj = next(x for x in target.entities if x.role == "OBJ")
            answer_counts["obj_correct"] += int(p_obj is not None and normalize_phrase(p_obj.phrase) == normalize_phrase(t_obj.phrase))
            p_anc = Counter(normalize_phrase(x.phrase) for x in predicted.entities if x.role == "ANC")
            t_anc = Counter(normalize_phrase(x.phrase) for x in target.entities if x.role == "ANC")
            answer_counts["anc_tp"] += sum((p_anc & t_anc).values()); answer_counts["anc_pred"] += sum(p_anc.values()); answer_counts["anc_true"] += sum(t_anc.values())
        if len(examples) < examples_limit:
            for index in range(min(len(batch["question"]), examples_limit-len(examples))):
                examples.append({"sample_id": batch["sample_id"][index], "question": batch["question"][index], "teacher_answer": batch["teacher_answer"][index],
                                 "evaluation_index": int(batch["evaluation_index"][index]),
                                 "conditioned_answer": result["chosen_answers"][index], "answer_source": result["answer_sources"][index],
                                 "attempted_generated_answer": result["attempted_generated_answers"][index],
                                 "selected_query": int(result["selected_query"][index]), "matched_query": int(result["matched_query"][index])})
    return {
        "metrics": book.state_dict(),
        "answer_counts": dict(answer_counts),
        "fallback": dict(fallback),
        "examples": examples,
        'raw_metrics':raw_book.state_dict(),
        'raw_counts':dict(raw_counts),
        'visual_counts':visual_counts,'full_counts':full_counts,'calls':dict(calls),
        'memory':__import__('causal_runtime').memory_report(),
    }


def merge_mode_payloads(payloads, examples_limit):
    book, answer_counts, fallback, examples = MetricBook(), Counter(), Counter(), []
    raw_book,raw_counts=MetricBook(),Counter()
    visual_counts=[];full_counts=[];calls=Counter();memories=[]
    for payload in payloads:
        book.merge(payload["metrics"])
        visual_counts.extend(payload.get("visual_counts",[]));full_counts.extend(payload.get("full_counts",[]))
        calls.update(payload.get("calls",{}));memories.append(payload.get("memory",{}))
        answer_counts.update(payload["answer_counts"])
        fallback.update(payload["fallback"])
        examples.extend(payload["examples"])
        raw_book.merge(payload.get('raw_metrics',{'sums':{},'counts':{}}))
        raw_counts.update(payload.get('raw_counts',{}))
    examples.sort(key=lambda row: row["evaluation_index"])
    examples = examples[:examples_limit]
    for row in examples:
        row.pop("evaluation_index", None)
    summary = book.summary(); summary['metric_denominators']=dict(book.counts); samples = max(answer_counts["samples"], 1)
    precision = answer_counts["anc_tp"]/max(answer_counts["anc_pred"],1); recall = answer_counts["anc_tp"]/max(answer_counts["anc_true"],1)
    summary.update({"answer/format_valid_rate": answer_counts["format_valid"]/samples, "answer/obj_phrase_accuracy": answer_counts["obj_correct"]/samples,
                    "answer/anc_precision": precision, "answer/anc_recall": recall, "answer/anc_f1": 2*precision*recall/max(precision+recall,1e-12),
                    "answer/fallback_rate": sum(value for key,value in fallback.items() if key not in {"generated","teacher"})/samples,
                    "answer/fallback_reasons": dict(fallback), "samples": answer_counts["samples"],
                    "data/missing_anc_mask_count":answer_counts["missing_anc_mask_count"],
                    "data/samples_with_missing_anc_masks":answer_counts["samples_with_missing_anc_masks"],
                    "data/missing_anc_mask_samples_denominator":answer_counts["samples"]})
    if raw_counts['attempted']:
        summary['raw_generated_no_fallback']={**raw_book.summary(),'samples':raw_counts['evaluable'],
            'attempted':raw_counts['attempted'],'coverage':raw_counts['evaluable']/raw_counts['attempted'],
            'format_unevaluable':raw_counts['format_unevaluable'],'teacher_fallbacks':0,'metric_denominators':dict(raw_book.counts),'missing_anc_mask_count':raw_counts['missing_anc_mask_count'],'missing_anc_mask_samples_denominator':raw_counts['attempted']}
    from causal_runtime import memory_report
    summary['visual_input']={'llm_visual_tokens_mean':sum(visual_counts)/max(len(visual_counts),1),'llm_visual_tokens_max':max(visual_counts,default=0),
        'full_superpoint_mean':sum(full_counts)/max(len(full_counts),1),'full_superpoint_max':max(full_counts,default=0)}
    summary['llm_calls']=dict(calls);summary['memory']={key:max((m.get(key) or 0 for m in memories),default=0) or None for key in ['peak_allocated_bytes','peak_reserved_bytes']}
    return summary, examples


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path); parser.add_argument("checkpoint", type=Path); parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-samples", type=int); parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--val-annotations", type=Path)
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument(
        "--eval-cache-root",
        type=Path,
        help="Dedicated normal/superpoint cache prepared for cross-dataset validation",
    )
    args = parser.parse_args(); config = load_segmentation_config(args.config)
    rank, world_size, local_rank, distributed = distributed_context()
    control=dist.new_group(backend='gloo',timeout=timedelta(hours=2)) if distributed else None
    manifest=[checkpoint_manifest(config['model']['model_name_or_path']) if rank==0 else None]
    if distributed:dist.broadcast_object_list(manifest,src=0,group=control)
    output = ensure_within(args.output_dir, config["project"]["artifact_root"], must_exist=False)
    checkpoint_path = ensure_within(args.checkpoint, config["project"]["artifact_root"], must_exist=True)
    if args.val_annotations is not None:
        val_annotations = ensure_within(
            args.val_annotations, config["project"]["artifact_root"], must_exist=True
        )
        config["data"]["val_annotations"] = str(val_annotations)
    if args.dataset_name is not None:
        config["data"]["dataset_name"] = args.dataset_name
    if args.eval_cache_root is not None:
        cache_root = ensure_within(
            args.eval_cache_root, config["project"]["artifact_root"], must_exist=True
        )
        config["data"]["normal"]["cache_dir"] = str(cache_root / "normals")
        config["data"]["superpoint"]["cache_dir"] = str(cache_root / "superpoints")
    dataset = build_segmentation_dataset(config, "val")
    if args.val_annotations is None and len(dataset)!=9483:raise ValueError("Expected all 9483 validation samples")
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    dataset = IndexedDataset(dataset)
    sampler = ExactDistributedEvalSampler(dataset, rank, world_size)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=int(config["data"]["num_workers"]), collate_fn=collate_samples)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier(group=control)
    model = ObjAncSegmentationModel(config).to(device); checkpoint = load_checkpoint(model, checkpoint_path,manifest[0]); model.eval()
    use_amp = bool(config["training"].get("amp", True))
    local_payload = {
        "teacher": evaluate_mode(model, loader, "teacher", float(config["validation"]["mask_threshold"]), args.examples, use_amp=use_amp),
        "generated": evaluate_mode(model, loader, "generated", float(config["validation"]["mask_threshold"]), args.examples, use_amp=use_amp),
    }
    gathered = [None] * world_size if rank == 0 else None
    if distributed:
        dist.gather_object(local_payload, gathered, dst=0,group=control)
    else:
        gathered = [local_payload]
    if rank == 0:
        teacher, teacher_examples = merge_mode_payloads([item["teacher"] for item in gathered], args.examples)
        generated, generated_examples = merge_mode_payloads([item["generated"] for item in gathered], args.examples)
        summary = {"objective": OBJECTIVE, "checkpoint": str(checkpoint_path), "step": checkpoint["step"],
                   'completed_epochs':checkpoint['completed_epochs'],
                   "distributed_world_size": world_size, "validation_samples": len(dataset),
                   "validation_dataset": str(config["data"].get("dataset_name", "unknown")),
                   "validation_annotations": str(config["data"]["val_annotations"]),
                   "teacher_answer": teacher, "generated_answer": generated,
                   "oracle_generated_gap": teacher.get("obj/oracle_iou",0)-generated.get("obj/oracle_iou",0),
                   "selected_generated_gap": teacher.get("obj/selected_iou",0)-generated.get("obj/selected_iou",0)}
        (output/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
        with (output/"examples.jsonl").open("w", encoding="utf-8") as handle:
            for mode, rows in (("teacher", teacher_examples), ("generated", generated_examples)):
                for row in rows: handle.write(json.dumps({"mode": mode, **row}, ensure_ascii=False)+"\n")
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if distributed:
        dist.barrier(group=control)
        dist.destroy_process_group()


if __name__ == "__main__": main()
