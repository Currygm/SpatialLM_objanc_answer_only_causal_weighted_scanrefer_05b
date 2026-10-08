"""Frozen SpatialLM encoder plus LoRA Qwen OBJ/ANC Answer model."""
from __future__ import annotations
import importlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    LoraConfig = TaskType = get_peft_model = None
from .answer_format import ObjAncGrammar, Phase, ParsedAnswer, parse_answer
from .backbone import CompressedVisualTokens, FrozenCompressedSonata

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
ROLE_QUESTION, ROLE_OBJ, ROLE_ANC, ROLE_OTHER = 0, 1, 2, 3

def validate_language_shape(base_config, cfg):
    if (int(base_config.hidden_size), int(base_config.num_hidden_layers)) != (
        int(cfg['expected_hidden_size']), int(cfg['expected_num_hidden_layers'])
    ):
        raise ValueError('Base language dimensions differ from configured experiment')

def attach_language_lora(base, cfg):
    """Select exact Qwen leaves, never similarly named point-backbone modules."""
    base.requires_grad_(False)
    leaves = set(cfg['target_modules'])
    targets = [name for name, module in base.named_modules()
               if isinstance(module, nn.Linear)
               and re.fullmatch(r'model\.layers\.\d+\.(self_attn|mlp)\.[^.]+', name)
               and name.rsplit('.', 1)[-1] in leaves]
    expected = int(base.config.num_hidden_layers) * len(leaves)
    if len(targets) != expected:
        raise ValueError(f'Expected {expected} Qwen LoRA targets, found {len(targets)}')
    return get_peft_model(base, LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False,
        r=int(cfg['r']), lora_alpha=int(cfg['alpha']),
        lora_dropout=float(cfg['dropout']), bias=str(cfg.get('bias', 'none')),
        target_modules=targets,
    )), len(targets)

@dataclass
class AnswerOnlyContext:
    answer_hidden: list[torch.Tensor]
    token_roles: list[torch.Tensor]
    phrase_spans: list[torch.Tensor]
    entity_roles: list[tuple[str, ...]]
    parsed_answers: list[ParsedAnswer]

class StructuredTokenRows(nn.Module):
    """Four trainable tied input/output rows and output biases."""
    def __init__(self, hidden_size: int, token_ids: list[int], base_vocab_size: int) -> None:
        super().__init__()
        expected = list(range(base_vocab_size, base_vocab_size + 4))
        if token_ids != expected:
            raise ValueError(f"Structured token IDs must be contiguous {expected}, got {token_ids}")
        self.base_vocab_size = int(base_vocab_size)
        self.rows = nn.Embedding(4, int(hidden_size))
        self.output_rows = nn.Embedding(4, int(hidden_size))
        nn.init.normal_(self.rows.weight, mean=0.0, std=float(hidden_size) ** -0.5)
        nn.init.normal_(self.output_rows.weight, mean=0.0, std=float(hidden_size) ** -0.5)

    @property
    def vocab_size(self) -> int:
        return self.base_vocab_size + 4

    def embed(self, ids: torch.Tensor, base_embedding: nn.Module) -> torch.Tensor:
        structured = ids.ge(self.base_vocab_size)
        result = base_embedding(ids.masked_fill(structured, 0))
        if bool(structured.any()):
            rows = ids[structured] - self.base_vocab_size
            if int(rows.min()) < 0 or int(rows.max()) >= 4:
                raise ValueError("Unknown extended-vocabulary token")
            result = result.clone()
            result[structured] = self.rows(rows).to(result.dtype)
        return result

    def logits(self, hidden: torch.Tensor, base_head: nn.Module) -> torch.Tensor:
        base = base_head(hidden)
        extra = F.linear(hidden.float(), self.output_rows.weight.float())
        return torch.cat([base, extra.to(base.dtype)], dim=-1)

class TXYZRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        if head_dim % 8:
            raise ValueError("TXYZ RoPE requires head_dim divisible by 8")
        axis_dim = head_dim // 4
        inv = 1.0 / (float(theta) ** (torch.arange(0, axis_dim, 2).float() / axis_dim))
        self.head_dim = int(head_dim)
        self.register_buffer("inv_freq", inv, persistent=False)

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor):
        if position_ids.ndim != 3 or position_ids.shape[0] != 4:
            raise ValueError("TXYZ positions must be [4,B,L]")
        positions = position_ids.permute(1, 2, 0).float()
        phases = (positions[..., None] * self.inv_freq.view(1, 1, 1, -1)).flatten(-2)
        phases = torch.cat([phases, phases], dim=-1)
        return phases.cos().to(hidden_states.dtype), phases.sin().to(hidden_states.dtype)

class CompressedVisualObjAncLM(nn.Module):
    objective = "spatiallm_compressed_visual_objanc_answer_lm_v1"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        if get_peft_model is None:
            raise ImportError("peft is required")
        self.config = config
        cfg = config["model"]
        repo = str(Path(cfg["spatiallm_repo"]).resolve(strict=True))
        if repo not in sys.path:
            sys.path.insert(0, repo)
        importlib.import_module("spatiallm.model.spatiallm_qwen")
        importlib.import_module("spatiallm.model.spatiallm_llama")
        cache = str(config["project"]["cache_dir"])
        trust = bool(cfg.get("trust_remote_code", True))
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_name_or_path"], cache_dir=cache, trust_remote_code=trust,
            padding_side="right", use_fast=True, token=False,
        )
        if not self.tokenizer.is_fast:
            raise ValueError("A fast tokenizer is required")
        base_config = AutoConfig.from_pretrained(
            cfg["model_name_or_path"], cache_dir=cache, trust_remote_code=trust, token=False
        )
        point_cfg = dict(base_config.point_config)
        point_cfg["enable_fourier_encode"] = False
        base_config.point_config = point_cfg
        dtype = _DTYPES[str(cfg.get("dtype", "bfloat16"))]
        base, loading = AutoModelForCausalLM.from_pretrained(
            cfg["model_name_or_path"], config=base_config, cache_dir=cache,
            trust_remote_code=trust, torch_dtype=dtype, attn_implementation="sdpa",
            token=False, output_loading_info=True,
        )
        bad = {key: loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys") if loading.get(key)}
        if bad:
            raise ValueError(f"Incomplete SpatialLM checkpoint: {bad}")
        validate_language_shape(base.config, cfg)
        if tuple(int(v) for v in base.config.point_config["stride"]) != (2, 2, 2, 2):
            raise ValueError("Base checkpoint must retain native Sonata strides")
        base.requires_grad_(False)
        base.set_point_backbone_dtype(torch.float32)
        lora = cfg["lora"]
        self.spatiallm, self.lora_module_count = attach_language_lora(base, lora)
        self.base = self.spatiallm.get_base_model()
        qwen = self.base.get_model()
        head_dim = int(self.base.config.hidden_size) // int(self.base.config.num_attention_heads)
        qwen.rotary_emb = TXYZRotaryEmbedding(head_dim, float(getattr(self.base.config, "rope_theta", 1_000_000.0)))
        if bool(cfg.get("gradient_checkpointing", True)):
            self.base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.encoder = FrozenCompressedSonata(self.base)
        self.system_prompt = str(cfg["system_prompt"])
        self.point_prompt = str(cfg["point_prompt"])
        self.user_prefix = str(cfg["user_prefix"])
        self.base_vocab_size = int(self.base.config.vocab_size)
        if len(self.tokenizer) > self.base_vocab_size:
            raise ValueError("Tokenizer is larger than the frozen base vocabulary")
        if len(self.tokenizer) < self.base_vocab_size:
            self.tokenizer.add_tokens(
                [f"<|objanc_unused_{i}|>" for i in range(self.base_vocab_size - len(self.tokenizer))],
                special_tokens=True,
            )
        tokens = list(cfg["structured_tokens"])
        native=[row['content'] for row in __import__('json').loads(self.tokenizer.backend_tokenizer.to_str())['added_tokens'] if row.get('special')]
        self.tokenizer.add_special_tokens({'additional_special_tokens':list(dict.fromkeys([*native,*tokens]))})
        self.structured_token_ids = {
            token: int(self.tokenizer.convert_tokens_to_ids(token)) for token in tokens
        }
        self.structured = StructuredTokenRows(
            int(self.base.config.hidden_size),
            [self.structured_token_ids[token] for token in tokens],
            self.base_vocab_size,
        )
        self.eos_token_id = int(self.tokenizer.eos_token_id or self.base.config.eos_token_id)
        self.generation_config = cfg["generation"]
        self._freeze_boundary()

    def tokenizer_snapshot(self):
        return {'backend':self.tokenizer.backend_tokenizer.to_str(),
            'special_tokens_map':self.tokenizer.special_tokens_map,'chat_template':self.tokenizer.chat_template,
            'padding_side':self.tokenizer.padding_side,'task_ids':dict(self.structured_token_ids)}

    def _freeze_boundary(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad_("lora_" in name or name.startswith("structured."))
        self.base.point_backbone.requires_grad_(False)
        self.base.point_proj.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.point_backbone.eval()
        self.base.point_proj.eval()
        return self

    @staticmethod
    def _single_position(ids: torch.Tensor, token_id: int) -> int:
        positions = torch.where(ids.eq(token_id))[0]
        if positions.numel() != 1:
            raise ValueError(f"Expected one point marker {token_id}, got {positions.numel()}")
        return int(positions.item())

    def _render_prompt(self, question: str) -> str:
        content = f"{self.point_prompt}\n{self.user_prefix}{question}"
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True,
        )

    def _prompt_tokens(self, question: str):
        rendered = self._render_prompt(question)
        encoded = self.tokenizer(
            rendered, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt"
        )
        ids, offsets = encoded["input_ids"][0], encoded["offset_mapping"][0]
        start = rendered.rfind(question)
        if start < 0:
            raise ValueError("Rendered prompt lost the question")
        positions = torch.where(
            (offsets[:, 1] > start) & (offsets[:, 0] < start + len(question))
        )[0]
        if positions.numel() == 0:
            raise ValueError("Question mapped to zero tokens")
        return ids, positions

    def encode_points(self, point_clouds: list[torch.Tensor]) -> list[CompressedVisualTokens]:
        return [self.encoder(cloud) for cloud in point_clouds]

    def _merge(self, ids: torch.Tensor, visual: CompressedVisualTokens):
        device = next(self.parameters()).device
        ids = ids.to(device)
        embeddings = self.structured.embed(ids, self.spatiallm.get_input_embeddings())
        start = self._single_position(ids, int(self.base.config.point_start_token_id))
        end = self._single_position(ids, int(self.base.config.point_end_token_id))
        points = visual.hidden.to(device=device, dtype=embeddings.dtype)
        merged = torch.cat([embeddings[:start + 1], points, embeddings[end:]], dim=0)
        point_slice = slice(start + 1, start + 1 + points.shape[0])
        shift = points.shape[0] - (end - start - 1)
        positions = torch.zeros((merged.shape[0], 4), device=device, dtype=torch.float32)
        text_mask = torch.ones(merged.shape[0], device=device, dtype=torch.bool)
        text_mask[point_slice] = False
        positions[text_mask, 0] = torch.arange(int(text_mask.sum()), device=device).float()
        xyz = visual.xyz.to(device=device, dtype=torch.float32)
        if xyz.shape != (points.shape[0], 3):
            raise ValueError("enc4 coordinates and visual tokens do not align")
        positions[point_slice, 1:] = xyz
        return merged, positions, point_slice, int(shift)

    def forward(self, point_clouds, questions, teacher_answers, *, visuals=None, return_context=True):
        if visuals is None:
            visuals = self.encode_points(point_clouds)
        return self._causal_forward(point_clouds, questions, teacher_answers, visuals, return_context=return_context)

    @staticmethod
    def _answer_roles(answer: str, offsets: torch.Tensor, parsed: ParsedAnswer):
        roles = torch.full((offsets.shape[0],), ROLE_OTHER, dtype=torch.long)
        spans = torch.zeros((len(parsed.entities), offsets.shape[0]), dtype=torch.bool)
        tag_pattern = re.compile(r"<(obj|anc)>\s*(.*?)\s*</\1>", re.I | re.S)
        tagged = list(tag_pattern.finditer(answer))
        if len(tagged) != len(parsed.entities):
            raise ValueError("Parsed entities and tag ranges differ")
        for index, (match, entity) in enumerate(zip(tagged, parsed.entities)):
            role_id = ROLE_OBJ if entity.role == "OBJ" else ROLE_ANC
            overlap_block = (offsets[:, 1] > match.start()) & (offsets[:, 0] < match.end())
            overlap_phrase = (
                (offsets[:, 1] > entity.content_start)
                & (offsets[:, 0] < entity.content_end)
            )
            roles[overlap_block] = role_id
            spans[index] = overlap_phrase
            if not bool(overlap_phrase.any()):
                raise ValueError("Entity phrase mapped to zero Answer tokens")
        return roles, spans

    def encode_causal_context(self, point_clouds, questions, answers, *, visuals):
        return self._causal_forward(point_clouds,questions,answers,visuals,return_context=True)['context']

    def pack_causal_batch(self, point_clouds, questions, answers, visuals):
        if not (len(visuals)==len(point_clouds)==len(questions)==len(answers)) or not answers:
            raise ValueError('Causal context batch lengths differ or empty batch')
        embeddings_rows, position_rows, label_rows, maps, parsed_rows=[],[],[],[],[]
        for question,answer,visual in zip(questions,answers,visuals):
            parsed=parse_answer(answer)
            if not parsed.valid:raise ValueError(f'Invalid teacher/chosen Answer: {parsed.errors}')
            prompt_ids,question_indices=self._prompt_tokens(question)
            encoded=self.tokenizer(answer,add_special_tokens=False,return_offsets_mapping=True,return_tensors='pt')
            answer_ids=encoded['input_ids'][0]
            ids=torch.cat([prompt_ids,answer_ids,answer_ids.new_tensor([self.eos_token_id])])
            merged,positions,_,shift=self._merge(ids,visual)
            end=self._single_position(prompt_ids,int(self.base.config.point_end_token_id))
            q_indices=question_indices+(question_indices>=end).long()*shift
            a_indices=torch.arange(len(answer_ids))+len(prompt_ids)+shift
            a_roles,spans=self._answer_roles(answer,encoded['offset_mapping'][0],parsed)
            labels=torch.full((len(merged),),-100,device=merged.device,dtype=torch.long)
            labels[len(prompt_ids)+shift:]=torch.cat([answer_ids,answer_ids.new_tensor([self.eos_token_id])]).to(merged.device)
            embeddings_rows.append(merged);position_rows.append(positions);label_rows.append(labels)
            maps.append((q_indices.to(merged.device),a_indices.to(merged.device),a_roles,spans));parsed_rows.append(parsed)
        embeds=pad_sequence(embeddings_rows,batch_first=True)
        valid=pad_sequence([torch.ones(len(row),device=embeds.device,dtype=torch.long) for row in embeddings_rows],batch_first=True)
        positions=pad_sequence(position_rows,batch_first=True).permute(2,0,1)
        labels=pad_sequence(label_rows,batch_first=True,padding_value=-100)
        return embeds,valid,positions,labels,maps,parsed_rows

    def _causal_forward(self, point_clouds, questions, answers, visuals, *, return_context=True):
        embeds,valid,positions,labels,maps,parsed_rows=self.pack_causal_batch(point_clouds,questions,answers,visuals)
        output=self.base.get_model()(input_ids=None,inputs_embeds=embeds,attention_mask=valid,
            position_ids=positions,use_cache=False,return_dict=True)
        target=labels[:,1:];selected=target.ne(-100)
        logits=self.structured.logits(output.last_hidden_state[:,:-1][selected],self.base.lm_head)
        loss=F.cross_entropy(logits.float(),target[selected])
        result={'loss':loss,'losses':{'answer_lm':loss},'metrics':{'token_accuracy':logits.argmax(-1).eq(target[selected]).float().mean()}}
        if return_context:
            a_hidden=[];roles=[];span_rows=[];entity_roles=[]
            for i,(q_idx,a_idx,a_roles,spans) in enumerate(maps):
                a_hidden.append(output.last_hidden_state[i].index_select(0,a_idx))
                roles.append(a_roles)
                span_rows.append(spans)
                entity_roles.append(tuple(e.role for e in parsed_rows[i].entities))
            result['context']=AnswerOnlyContext(a_hidden,roles,span_rows,entity_roles,parsed_rows)
        return result

    @torch.no_grad()
    def generate_one(self, question: str, point_cloud: torch.Tensor | None = None, *, visual: CompressedVisualTokens | None = None) -> dict[str, Any]:
        if visual is None:
            if point_cloud is None:
                raise ValueError("point_cloud is required without a precomputed visual")
            visual = self.encode_points([point_cloud])[0]
        prompt_ids, _ = self._prompt_tokens(question)
        prompt, positions, _, _ = self._merge(prompt_ids, visual)
        valid = torch.ones((1, prompt.shape[0]), device=prompt.device, dtype=torch.long)
        self.generation_prefill_calls=getattr(self,'generation_prefill_calls',0)+1
        output = self.base.get_model()(
            input_ids=None, inputs_embeds=prompt[None], attention_mask=valid,
            position_ids=positions.T[:, None], use_cache=True, return_dict=True,
        )
        hidden, past = output.last_hidden_state[:, -1], output.past_key_values
        grammar = ObjAncGrammar(
            self.structured_token_ids, self.eos_token_id,
            int(self.generation_config["max_phrase_tokens"]),
        )
        generated: list[int] = []
        budget_forced = False
        forbidden = {
            int(value) for value in self.tokenizer.all_special_ids
            if 0 <= int(value) < self.base_vocab_size and int(value) != self.eos_token_id
        }
        text_position = int(positions[:, 0].max().item()) + 1
        max_new = int(self.generation_config["max_new_tokens"])
        for step in range(max_new):
            logits = self.structured.logits(hidden, self.base.lm_head)[0]
            phrase_text = self.tokenizer.decode(
                grammar.phrase_ids or [], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            allowed = grammar.allowed(logits.numel(), logits.device, phrase_has_text=bool(phrase_text))
            before = allowed.clone()
            allowed = grammar.force_for_budget(
                allowed, max_new - step, phrase_has_text=bool(phrase_text)
            )
            budget_forced = budget_forced or not torch.equal(before, allowed)
            if forbidden:
                allowed[torch.tensor(sorted(forbidden), device=logits.device)] = False
            masked = logits.masked_fill(~allowed, -torch.inf)
            token = int(masked.argmax())
            if grammar.phase in {Phase.IN_OBJ, Phase.IN_ANC} and not phrase_text:
                while not self.tokenizer.decode(
                    [token], skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip():
                    masked[token] = -torch.inf
                    if not bool(torch.isfinite(masked).any()):
                        raise RuntimeError("No visible phrase token remains")
                    token = int(masked.argmax())
            generated.append(token)
            grammar.update(token, phrase_has_text=bool(phrase_text))
            if grammar.phase is Phase.FINISHED:
                break
            token_ids = torch.tensor([[token]], device=prompt.device, dtype=torch.long)
            valid = torch.cat([valid, valid.new_ones((1, 1))], dim=1)
            next_position = torch.tensor(
                [[[float(text_position)]], [[0.0]], [[0.0]], [[0.0]]],
                device=prompt.device,
            )
            self.generation_incremental_calls=getattr(self,'generation_incremental_calls',0)+1
            output = self.base.get_model()(
                input_ids=None,
                inputs_embeds=self.structured.embed(token_ids, self.spatiallm.get_input_embeddings()),
                attention_mask=valid, position_ids=next_position,
                past_key_values=past, use_cache=True, return_dict=True,
            )
            hidden, past = output.last_hidden_state[:, -1], output.past_key_values
            text_position += 1
        answer_ids = [value for value in generated if value != self.eos_token_id]
        answer = self.tokenizer.decode(
            answer_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ).strip()
        parsed = parse_answer(answer)
        return {
            "answer": answer,
            "parsed": parsed,
            "budget_forced": budget_forced,
            "format_failure": not parsed.valid,
            "errors": list(parsed.errors),
        }

    def optimizer_parameter_groups(self, *, lora_lr: float, structured_token_lr: float, weight_decay: float):
        lora, structured = [], []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            (structured if name.startswith("structured.") else lora).append(parameter)
        if not lora or not structured:
            raise RuntimeError("Expected both LoRA and structured-token parameters")
        return [
            {"name": "qwen_lora", "params": lora, "lr": lora_lr, "weight_decay": weight_decay},
            {"name": "structured_tokens", "params": structured, "lr": structured_token_lr, "weight_decay": weight_decay},
        ]

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        names = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        return {name: value.detach().cpu() for name, value in self.state_dict().items() if name in names}

    def trainable_parameter_summary(self) -> dict[str, int]:
        result = {"qwen_lora": 0, "structured_tokens": 0, "total": 0}
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                key = "structured_tokens" if name.startswith("structured.") else "qwen_lora"
                result[key] += parameter.numel()
                result["total"] += parameter.numel()
        return result
