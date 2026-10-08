"""OBJ/ANC answer parsing, validation, and constrained decoding state."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
import re
from typing import Sequence

STRUCTURED_TOKENS = ("<obj>", "</obj>", "<anc>", "</anc>")
_TAG_PATTERN = re.compile(r"</?(?:obj|anc)>", re.IGNORECASE)

@dataclass(frozen=True)
class EntitySpan:
    role: str
    phrase: str
    content_start: int
    content_end: int

@dataclass(frozen=True)
class ParsedAnswer:
    text: str
    entities: tuple[EntitySpan, ...]
    errors: tuple[str, ...]
    @property
    def valid(self) -> bool:
        return not self.errors
    @property
    def obj(self) -> EntitySpan | None:
        rows = [row for row in self.entities if row.role == "OBJ"]
        return rows[0] if len(rows) == 1 else None
    @property
    def anchors(self) -> tuple[EntitySpan, ...]:
        return tuple(row for row in self.entities if row.role == "ANC")

def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())

def parse_answer(answer: str) -> ParsedAnswer:
    text = str(answer or "")
    entities: list[EntitySpan] = []
    errors: list[str] = []
    active_role: str | None = None
    content_start = -1
    for match in _TAG_PATTERN.finditer(text):
        token = match.group(0).lower()
        closing = token.startswith("</")
        role = "OBJ" if "obj" in token else "ANC"
        if not closing:
            if active_role is not None:
                errors.append("nested_tag")
            else:
                active_role = role
                content_start = match.end()
            continue
        if active_role is None:
            errors.append("orphan_close")
            continue
        if active_role != role:
            errors.append("mismatched_close")
            continue
        phrase = " ".join(text[content_start:match.start()].split())
        if not phrase:
            errors.append("empty_span")
        else:
            entities.append(EntitySpan(role, phrase, content_start, match.start()))
        active_role = None
        content_start = -1
    if active_role is not None:
        errors.append("unclosed_tag")
    obj_count = sum(row.role == "OBJ" for row in entities)
    if obj_count != 1:
        errors.append("obj_count_not_one")
    lowered = text.lower()
    for token in STRUCTURED_TOKENS:
        if lowered.count(token) != sum(
            (row.role == "OBJ" and "obj" in token)
            or (row.role == "ANC" and "anc" in token)
            for row in entities
        ):
            if token.startswith("<") and not token.startswith("</"):
                errors.append("unbalanced_tags")
                break
    return ParsedAnswer(text, tuple(entities), tuple(sorted(set(errors))))

def entity_phrase_sets(parsed: ParsedAnswer) -> tuple[str | None, set[str]]:
    obj = normalize_text(parsed.obj.phrase) if parsed.obj is not None else None
    anchors = {normalize_text(row.phrase) for row in parsed.anchors}
    return obj, anchors

class Phase(Enum):
    BEFORE_OBJ = auto()
    IN_OBJ = auto()
    AFTER_OBJ = auto()
    IN_ANC = auto()
    FINISHED = auto()

@dataclass
class ObjAncGrammar:
    token_ids: dict[str, int]
    eos_token_id: int
    max_phrase_tokens: int
    phase: Phase = Phase.BEFORE_OBJ
    phrase_ids: list[int] | None = None

    def __post_init__(self) -> None:
        self.phrase_ids = []

    def allowed(self, vocab_size: int, device, *, phrase_has_text: bool):
        import torch
        allowed = torch.ones(vocab_size, device=device, dtype=torch.bool)
        for token_id in self.token_ids.values():
            if 0 <= token_id < vocab_size:
                allowed[token_id] = False
        allowed[self.eos_token_id] = False
        ids = self.token_ids
        if self.phase is Phase.BEFORE_OBJ:
            allowed[ids["<obj>"]] = True
        elif self.phase is Phase.IN_OBJ:
            if phrase_has_text:
                allowed[ids["</obj>"]] = True
            if len(self.phrase_ids or []) >= self.max_phrase_tokens and phrase_has_text:
                allowed.zero_()
                allowed[ids["</obj>"]] = True
        elif self.phase is Phase.AFTER_OBJ:
            allowed[ids["<anc>"]] = True
            allowed[self.eos_token_id] = True
        elif self.phase is Phase.IN_ANC:
            if phrase_has_text:
                allowed[ids["</anc>"]] = True
            if len(self.phrase_ids or []) >= self.max_phrase_tokens and phrase_has_text:
                allowed.zero_()
                allowed[ids["</anc>"]] = True
        else:
            allowed.zero_()
            allowed[self.eos_token_id] = True
        return allowed

    def force_for_budget(self, allowed, remaining: int, *, phrase_has_text: bool):
        ids = self.token_ids
        if self.phase is Phase.BEFORE_OBJ and remaining <= 4:
            allowed.zero_(); allowed[ids["<obj>"]] = True
        elif self.phase is Phase.IN_OBJ and phrase_has_text and remaining <= 2:
            allowed.zero_(); allowed[ids["</obj>"]] = True
        elif self.phase is Phase.AFTER_OBJ and remaining <= 1:
            allowed.zero_(); allowed[self.eos_token_id] = True
        elif self.phase is Phase.IN_ANC and phrase_has_text and remaining <= 2:
            allowed.zero_(); allowed[ids["</anc>"]] = True
        return allowed

    def update(self, token_id: int, *, phrase_has_text: bool) -> None:
        ids = self.token_ids
        if self.phase is Phase.BEFORE_OBJ and token_id == ids["<obj>"]:
            self.phase, self.phrase_ids = Phase.IN_OBJ, []
        elif self.phase is Phase.IN_OBJ and token_id == ids["</obj>"]:
            if not phrase_has_text:
                raise ValueError("OBJ phrase cannot be empty")
            self.phase, self.phrase_ids = Phase.AFTER_OBJ, []
        elif self.phase is Phase.AFTER_OBJ and token_id == ids["<anc>"]:
            self.phase, self.phrase_ids = Phase.IN_ANC, []
        elif self.phase is Phase.AFTER_OBJ and token_id == self.eos_token_id:
            self.phase = Phase.FINISHED
        elif self.phase is Phase.IN_ANC and token_id == ids["</anc>"]:
            if not phrase_has_text:
                raise ValueError("ANC phrase cannot be empty")
            self.phase, self.phrase_ids = Phase.AFTER_OBJ, []
        elif self.phase in {Phase.IN_OBJ, Phase.IN_ANC}:
            assert self.phrase_ids is not None
            self.phrase_ids.append(int(token_id))
