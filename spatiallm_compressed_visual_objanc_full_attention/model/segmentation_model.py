"""OBJ/ANC Qwen conditioning plus nine-layer UniVLG segmentation."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .answer_format import parse_answer
from .curriculum import normalize_phrase
from .llm import CompressedVisualObjAncLM
from .answer_stage_3b import visual_signature
from .role_losses import role_mean,role_focal
from .segmentation_backbone import PointUnetFusion, SharedSonataEncoder
from .segmentation_decoder import ObjAncFullCycleDecoder
from .sonata_lora import SonataLoraConfig, apply_sonata_lora

OBJECTIVE = "spatiallm_objanc_answer_only_causal_weighted_scanrefer_05b_v1"
MAX_MATCH_POINTS = 50_000


def mean_sigmoid_scores(logits: torch.Tensor, positive: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    counts = positive.sum(-1)
    if bool((counts[valid] == 0).any()):
        raise ValueError("Every valid entity needs positive Answer phrase tokens")
    score = (logits.float().sigmoid()[:, :, None] * positive[:, None].float()).sum(-1)
    score = score / counts.clamp_min(1)[:, None]
    return score.masked_fill(~valid[:, None], 0).to(logits.dtype)


def dice_cost(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probability, targets = inputs.float().sigmoid(), targets.float()
    numerator = 2 * torch.einsum("qn,mn->qm", probability, targets)
    denominator = probability.sum(-1)[:, None] + targets.sum(-1)[None]
    return 1 - (numerator + 1) / (denominator + 1)


def bce_cost(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    positive = F.binary_cross_entropy_with_logits(inputs.float(), torch.ones_like(inputs).float(), reduction="none")
    negative = F.binary_cross_entropy_with_logits(inputs.float(), torch.zeros_like(inputs).float(), reduction="none")
    return (torch.einsum("qn,mn->qm", positive, targets.float()) + torch.einsum("qn,mn->qm", negative, 1-targets.float())) / inputs.shape[1]


class ObjAncSegmentationModel(nn.Module):
    objective = OBJECTIVE

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.answer_lm = CompressedVisualObjAncLM(config)
        if config['initialization'] != {'mode': 'fresh_05b'}:
            raise ValueError('Only fresh 0.5B task initialization is supported')
        point_cfg = config["point_encoder"]
        channels = tuple(int(x) for x in self.answer_lm.base.config.point_config["enc_channels"])
        self.point_unet=PointUnetFusion(channels)
        self.shared_encoder=SharedSonataEncoder(self.answer_lm.base,self.point_unet)
        self.sonata_lora_ids = apply_sonata_lora(
            self.answer_lm.base.point_backbone,
            SonataLoraConfig.from_mapping(point_cfg["sonata_lora"]),
        )
        for parameter in self.answer_lm.base.point_backbone.parameters():
            parameter.requires_grad_(id(parameter) in self.sonata_lora_ids)

        decoder = config["decoder"]
        self.decoder = ObjAncFullCycleDecoder(
            visual_dim=int(decoder["visual_dim"]), text_dim=int(decoder["text_dim"]),
            hidden_dim=int(decoder["hidden_dim"]), num_queries=int(decoder["num_queries"]),
            num_layers=int(decoder["num_layers"]), num_heads=int(decoder["num_heads"]),
            feedforward_dim=int(decoder["feedforward_dim"]), dropout=float(decoder["dropout"]),
            max_text_tokens=int(decoder["max_text_tokens"]),
            mask_threshold=float(decoder["attention_mask_threshold"]), text_temperature=float(decoder["text_temperature"]),
        )
        self.loss_cfg, self.matcher_cfg, self.ranking_cfg = config["loss"], config["matcher"], config["ranking"]
        self.mask_threshold = float(config["validation"]["mask_threshold"])
        self.answer_lm_weight = float(config["training"]["answer_lm_weight"])
        self.seed = int(config["training"]["seed"])
        self._checkpoint_names = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        self._assert_boundary()

    def train(self, mode: bool = True):
        super().train(mode)
        self.answer_lm.base.point_backbone.eval()
        stages = list(self.answer_lm.base.point_backbone.enc._modules.values())
        for index in (1, 2, 3, 4):
            stages[index].train(mode)
        self.answer_lm.base.point_proj.eval()
        return self

    def _encode_points(self, batch: dict[str, list[Any]]):
        stage0 = batch.get("stage0_to_superpoint")
        if stage0 is None or any(value is None for value in stage0):
            raise ValueError("Segmentation requires stage0_to_superpoint for every sample")
        return [self.shared_encoder(c,m,x) for c,m,x in zip(batch["point_cloud"],stage0,batch["superpoint_coord"])]

    def _choose_answers(self, batch, point_rows, answer_mode):
        if answer_mode not in {"teacher", "generated"}:
            raise ValueError(f"Unsupported answer_mode: {answer_mode}")
        ratio = 1.0 if answer_mode == "teacher" else 0.0
        answers, target_rows, entity_indices, reasons, attempted = [], [], [], [], []
        for index, (question, teacher, sample_id, features) in enumerate(zip(batch["question"], batch["teacher_answer"], batch["sample_id"], point_rows)):
            use_teacher = answer_mode == "teacher"
            reason = "teacher" if use_teacher else "generated"
            generated = None
            if not use_teacher:
                with torch.no_grad():
                    generated = self.answer_lm.generate_one(question, visual=features.compressed)
                teacher_parsed = parse_answer(teacher)
                predicted = generated["parsed"]
                teacher_obj = next(x for x in teacher_parsed.entities if x.role == "OBJ")
                predicted_obj = next((x for x in predicted.entities if x.role == "OBJ"), None)
                if generated["format_failure"] and generated["budget_forced"]:
                    use_teacher, reason = True, "budget_failure"
                elif generated["format_failure"]:
                    use_teacher, reason = True, "format_failure"
                elif predicted_obj is None or normalize_phrase(predicted_obj.phrase) != normalize_phrase(teacher_obj.phrase):
                    use_teacher, reason = True, "obj_phrase_mismatch"
            if use_teacher:
                answers.append(teacher)
                target_rows.append(list(range(len(batch["grounding_object_ids"][index]))))
                entity_indices.append(list(map(int, batch["answer_entity_indices"][index])))
            else:
                answers.append(generated["answer"])
                valid_teacher = list(zip(batch["grounding_roles"][index], batch["grounding_phrases"][index]))
                parsed = generated["parsed"]
                chosen_targets, chosen_entities = [], []
                for entity_index, entity in enumerate(parsed.entities):
                    matches = [row for row, (role, phrases) in enumerate(valid_teacher) if role == entity.role and normalize_phrase(phrases[0]) == normalize_phrase(entity.phrase)]
                    if len(matches) == 1 and (entity.role == "OBJ" or sum(normalize_phrase(x.phrase) == normalize_phrase(entity.phrase) and x.role == entity.role for x in parsed.entities) == 1):
                        chosen_targets.append(matches[0]); chosen_entities.append(entity_index)
                if 0 not in chosen_targets:
                    raise RuntimeError("Generated OBJ passed validation but did not map to target row zero")
                target_rows.append(chosen_targets); entity_indices.append(chosen_entities)
            reasons.append(reason)
            attempted.append(None if generated is None else generated["answer"])
        return answers, target_rows, entity_indices, reasons, attempted, ratio

    def _choose_raw_answers(self, batch, answers):
        """Measure original generated text against GT, including incorrect OBJ phrases."""
        target_rows, entity_indices = [], []
        for index, answer in enumerate(answers):
            parsed = parse_answer(answer)
            obj = [i for i,e in enumerate(parsed.entities) if e.role == 'OBJ']
            if not parsed.valid or len(obj) != 1:
                raise ValueError('Raw generated answer has no valid single OBJ')
            rows, indices = [0], [obj[0]]
            for i,e in enumerate(parsed.entities):
                if e.role != 'ANC': continue
                matches = [r for r,(role,phrases) in enumerate(zip(batch['grounding_roles'][index],batch['grounding_phrases'][index]))
                           if role == 'ANC' and normalize_phrase(phrases[0]) == normalize_phrase(e.phrase)]
                if len(matches)==1 and matches[0] not in rows:
                    rows.append(matches[0]); indices.append(i)
            target_rows.append(rows); entity_indices.append(indices)
        return list(answers),target_rows,entity_indices,['raw_generated']*len(answers),list(answers),0.

    def _total_loss(self, primary, auxiliary_losses, ranking, lm_loss):
        auxiliary = torch.stack(auxiliary_losses).sum()
        total = primary + float(self.loss_cfg['aux_weight'])*auxiliary + float(self.ranking_cfg['weight'])*ranking + self.answer_lm_weight*lm_loss
        return total, auxiliary

    def _connect_trainable_graph(self, total):
        return total + sum(p.reshape(-1)[0]*0 for p in self.parameters() if p.requires_grad)

    @staticmethod
    def _positive_maps(context, entity_indices):
        text_rows = context.answer_hidden
        max_entities = max(len(x) for x in entity_indices)
        max_text = max(x.shape[0] for x in text_rows)
        device = text_rows[0].device
        positive = torch.zeros((len(text_rows), max_entities, max_text), device=device, dtype=torch.bool)
        valid = torch.zeros((len(text_rows), max_entities), device=device, dtype=torch.bool)
        for batch_index, indices in enumerate(entity_indices):
            spans = context.phrase_spans[batch_index].to(device)
            selected = spans[torch.tensor(indices, device=device, dtype=torch.long)]
            positive[batch_index, :len(indices), :selected.shape[1]] = selected
            valid[batch_index, :len(indices)] = True
        return text_rows, positive, valid

    def _layer_loss(self, output, targets, positive, valid, entity_roles):
        logits, text_valid = output["pred_logits"], output["text_valid_mask"]
        scores = mean_sigmoid_scores(logits, positive.to(logits.device), valid.to(logits.device))
        assignments, bces, dices, roles = [], [], [], []
        for batch_index, target in enumerate(targets):
            length = target.shape[1]
            masks = output["pred_masks"][batch_index, :, :length]
            target = target.to(masks.device, masks.dtype)
            sample = torch.randperm(length, device=masks.device)[:MAX_MATCH_POINTS] if length > MAX_MATCH_POINTS else torch.arange(length, device=masks.device)
            cost = (-float(self.matcher_cfg["span_weight"]) * scores[batch_index, :, :target.shape[0]]
                    + float(self.matcher_cfg["mask_bce_weight"]) * bce_cost(masks[:, sample], target[:, sample])
                    + float(self.matcher_cfg["dice_weight"]) * dice_cost(masks[:, sample], target[:, sample]))
            q, t = linear_sum_assignment(cost.detach().float().cpu().numpy())
            q, t = torch.as_tensor(q, device=masks.device), torch.as_tensor(t, device=masks.device)
            assignments.append((q, t))
            selected, truth = masks[q], target[t]
            bces.append(F.binary_cross_entropy_with_logits(selected.float(), truth.float(), reduction="none").mean(-1))
            prob = selected.float().sigmoid(); dices.append(1-(2*(prob*truth).sum(-1)+1)/(prob.sum(-1)+truth.sum(-1)+1))
            roles.extend(entity_roles[batch_index][int(i)] for i in t)
        span,details=role_focal(logits,positive.to(logits.device),text_valid,assignments,entity_roles,float(self.loss_cfg['focal_alpha']),float(self.loss_cfg['focal_gamma']))
        bce,bce_roles=role_mean(torch.cat(bces),roles)
        dice,dice_roles=role_mean(torch.cat(dices),roles)
        for prefix,means in [('bce',bce_roles),('dice',dice_roles)]:
            for role,coefficient in [('OBJ',1.5),('ANC',.5)]:
                details[prefix+'_'+role.lower()]=means[role]
                details[prefix+'_'+role.lower()+'_weighted']=coefficient*means[role]
        self._last_layer_details=details
        total = float(self.loss_cfg["span_weight"])*span + float(self.loss_cfg["mask_bce_weight"])*bce + float(self.loss_cfg["dice_weight"])*dice
        return total, span, bce, dice, assignments, scores

    def _ranking(self, scores, masks, targets, point_maps, entity_roles):
        terms, best_rows = [], []
        with torch.no_grad():
            for batch_index, target in enumerate(targets):
                mapping = point_maps[batch_index].to(masks.device, torch.long)
                predictions = masks[batch_index, :, :target.shape[1]][:, mapping].sigmoid() >= self.mask_threshold
                sample_best = []
                for entity in target:
                    truth = entity.to(predictions.device, torch.bool)[None]
                    if truth.shape[1] != predictions.shape[1]:
                        raise ValueError("Raw entity mask does not align with point_to_superpoint")
                    iou = (predictions & truth).sum(-1).float() / (predictions | truth).sum(-1).clamp_min(1).float()
                    sample_best.append(iou.argmax())
                best_rows.append(torch.stack(sample_best))
        for batch_index, best in enumerate(best_rows):
            for entity, best_query in enumerate(best):
                terms.append(F.cross_entropy(scores[batch_index, :, entity].float()[None] / float(self.ranking_cfg["temperature"]), best_query[None]))
        value,means=role_mean(torch.stack(terms),[r for rows in entity_roles for r in rows]);self._ranking_details={'ranking_obj':means['OBJ'],'ranking_anc':means['ANC'],'ranking_obj_weighted':1.5*means['OBJ'],'ranking_anc_weighted':.5*means['ANC']}
        return value, best_rows

    def forward(
        self,
        batch: dict[str, list[Any]],
        *,
        epoch: int,
        optimizer_step: int,
        steps_per_epoch: int,
        answer_mode: str = "teacher",
        generated_answers: list[str] | None = None,
    ):
        if self.training and (
            answer_mode != "teacher"
            or bool(self.config["training"].get("allow_generated_answers", False))
        ):
            raise RuntimeError(
                "Training is strictly teacher-forced; generated answers are disabled"
            )
        point_rows = self._encode_points(batch)
        visuals = [row.compressed for row in point_rows]
        self.answer_lm.generation_prefill_calls=0;self.answer_lm.generation_incremental_calls=0
        if answer_mode == 'raw_generated':
            if generated_answers is None:raise ValueError('Raw evaluation requires cached generated answers')
            answers, target_rows, entity_indices, fallback, attempted, ratio = self._choose_raw_answers(batch,generated_answers)
        else:
            answers, target_rows, entity_indices, fallback, attempted, ratio = self._choose_answers(batch, point_rows, answer_mode)
        lm = self.answer_lm(batch["point_cloud"], batch["question"], answers, visuals=visuals, return_context=True)
        context = lm["context"]
        text_rows, positive, valid = self._positive_maps(context, entity_indices)
        target_masks = [batch["gt_superpoint_masks"][i][rows] for i, rows in enumerate(target_rows)]
        raw_targets = [batch["gt_point_masks"][i][rows] for i, rows in enumerate(target_rows)]
        decoder = self.decoder([x.superpoint_hidden for x in point_rows], text_rows, [x.superpoint_xyz for x in point_rows], context.token_roles)
        entity_roles=[[batch["grounding_roles"][i][r] for r in rows] for i,rows in enumerate(target_rows)]
        primary, span, bce, dice, assignments, scores = self._layer_loss(decoder, target_masks, positive, valid,entity_roles)
        loss_details=dict(self._last_layer_details)
        auxiliary_losses=[]
        for layer,x in enumerate(decoder['aux_outputs']):
            auxiliary_losses.append(self._layer_loss(x,target_masks,positive,valid,entity_roles)[0])
            loss_details.update({f'aux_{layer}_{k}':v for k,v in self._last_layer_details.items()})
        ranking, best = self._ranking(scores, decoder["pred_masks"], raw_targets, batch["point_to_superpoint"],entity_roles)
        total, auxiliary = self._total_loss(primary, auxiliary_losses, ranking, lm['loss'])
        if self.training:
            # Conditional text spans and final updates may leave parameters unused on a rank.
            # Zero connections preserve objective values and make DDP reduction complete.
            total = self._connect_trainable_graph(total)
        obj_scores = scores[:, :, 0]; selected = obj_scores.argmax(1)
        matched = []
        for source, target in assignments:
            position = torch.where(target.eq(0))[0]
            matched.append(source[position[0]])
        matched = torch.stack(matched)
        selected_logits = [decoder["pred_masks"][i, selected[i], :target_masks[i].shape[1]] for i in range(len(target_masks))]
        raw_logits = [x[m.to(x.device)] for x, m in zip(selected_logits, batch["point_to_superpoint"])]
        return {"loss": total, "losses": {**{k:v.detach() for k,v in {**loss_details,**self._ranking_details}.items()}, "span_contribution":(6*span).detach(),"bce_contribution":(5*bce).detach(),"dice_contribution":(2*dice).detach(),"ranking_contribution":(.5*ranking).detach(),"answer_contribution":(.1*lm["loss"]).detach(),"aux_contribution":(.5*auxiliary).detach(),"total": total.detach(), "span": span.detach(), "mask_bce": bce.detach(), "dice": dice.detach(), "aux": auxiliary.detach(), "ranking": ranking.detach(), "answer_lm": lm["loss"].detach()},
                "metrics": {"teacher_ratio": total.new_tensor(ratio), "generated_training_samples": total.new_tensor(0 if answer_mode == "teacher" else len(answers)), "fallback_rate": total.new_tensor(sum(x not in {"generated", "teacher"} for x in fallback)/len(fallback)), "missing_anc_mask_count": total.new_tensor(sum(batch.get("dropped_missing_entity_count", [0]))), "answer_token_accuracy": lm["metrics"]["token_accuracy"],
                "llm_visual_tokens":total.new_tensor(sum(len(r.compressed.hidden) for r in point_rows)/len(point_rows)),
                "full_superpoint_count":total.new_tensor(sum(len(r.superpoint_hidden) for r in point_rows)/len(point_rows)),
                "causal_forward_calls":total.new_tensor(1),"replay_calls":total.new_tensor(int(answer_mode!='teacher')),
                "generation_prefill_calls":total.new_tensor(self.answer_lm.generation_prefill_calls),
                "generation_incremental_calls":total.new_tensor(self.answer_lm.generation_incremental_calls)},
                "raw_logits": raw_logits, "query_scores": obj_scores, "query_entity_scores": scores,
                "selected_query": selected, "matched_query": matched, "best_queries_by_entity": best, "assignments": assignments,
                "chosen_answers": answers, "attempted_generated_answers": attempted, "answer_sources": fallback, "target_rows": target_rows, "entity_roles": [[batch["grounding_roles"][i][r] for r in rows] for i, rows in enumerate(target_rows)], "decoder_output": decoder}

    def optimizer_parameter_groups(self):
        training = self.config["training"]
        named = list(self.named_parameters())
        groups = []
        specifications = [
            ("structured_tokens", lambda n,p: n.startswith("answer_lm.structured."), training["structured_token_lr"]),
            ("sonata_lora", lambda n,p: id(p) in self.sonata_lora_ids, training["sonata_lora_lr"]),
            ("qwen_lora", lambda n,p: "lora_" in n and "sonata_lora_" not in n, training["qwen_lora_lr"]),
            ("point_unet", lambda n,p: n.startswith("point_unet.") or n.startswith("shared_encoder.point_unet."), training["unet_lr"]),
            ("decoder", lambda n,p: n.startswith("decoder."), training["decoder_lr"]),
        ]
        used: set[int] = set()
        for name, predicate, lr in specifications:
            values = [p for n,p in named if p.requires_grad and id(p) not in used and predicate(n,p)]
            if values:
                used.update(map(id, values)); groups.append({"name": name, "params": values, "lr": float(lr), "weight_decay": float(training["weight_decay"])})
        expected = {id(p) for _,p in named if p.requires_grad}
        if used != expected:
            raise RuntimeError("Optimizer does not cover the exact trainable boundary")
        return groups

    def _assert_boundary(self):
        for name, parameter in self.named_parameters():
            allowed = (name.startswith("visual_adapter.") or name.startswith("decoder.") or name.startswith("point_unet.") or name.startswith("shared_encoder.point_unet.") or name.startswith("answer_lm.structured.") or "lora_" in name)
            if parameter.requires_grad and not allowed:
                raise RuntimeError(f"Unexpected trainable parameter: {name}")
        self.answer_lm.base.point_proj.requires_grad_(False)

    def trainable_state_dict(self):
        from runtime_helpers import task_state
        return task_state(self)

    def architecture_signature(self):
        return {"objective": OBJECTIVE, "initialization": "fresh_05b", "shared_sonata_forward": True,
                "language_hidden_size": int(self.answer_lm.base.config.hidden_size),
                "language_layers": int(self.answer_lm.base.config.num_hidden_layers),
                "qwen_lora_modules": self.answer_lm.lora_module_count,
                "qwen_lora": self.config['model']['lora'],
                "decoder_config": dict(self.config['decoder']),
                "point_encoder_config": dict(self.config['point_encoder']),
                "answer_protocol": {key:self.config['model'][key] for key in
                    ('system_prompt','point_prompt','user_prefix','structured_tokens','generation')},
                "qwen_passes": visual_signature(self.config),"loss_role_grouping":{"OBJ":1.5,"ANC":.5,"background":1.0,"reduction":"independent_role_mean"},
                "loss_config":dict(self.config["loss"]),"matcher_config":dict(self.config["matcher"]),
                "ranking_config":dict(self.config["ranking"]),"validation_config":dict(self.config["validation"]),
                "decoder": {"version": self.decoder.version, "queries": 50, "layers": 9, "text_dim": int(self.config['decoder']['text_dim']), "visual_dim": 512, "max_text_tokens": 384},
                "roles": ["QUESTION", "OBJ", "ANC", "OTHER"], "matcher": "mean_sigmoid_span_2_15_6", "ranking": "final_hard_raw_point_iou_all_valid_entities",
                "training_answer_policy": "teacher_only", "evaluation_answer_modes": ["teacher", "generated"],
                "answer_lm_weight": self.answer_lm_weight}
