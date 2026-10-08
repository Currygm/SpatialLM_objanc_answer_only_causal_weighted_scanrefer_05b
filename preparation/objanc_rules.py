"""Original OBJ-first teacher-answer conversion functions."""
from __future__ import annotations
import hashlib
from typing import Any
FORMAT_VERSION = "objanc_description_grounding_v1"
TAGS = ("<obj>", "</obj>", "<anc>", "</anc>")

def clean_phrase(value: str) -> str:
    return " ".join(str(value).strip().split())

def valid_span(span: Any, text: str) -> tuple[int, int] | None:
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return None
    start, end = int(span[0]), int(span[1])
    if start < 0 or end <= start or end > len(text):
        return None
    if not clean_phrase(text[start:end]):
        return None
    return start, end

def choose_target(row: dict[str, Any], text: str) -> tuple[int, int]:
    rows = row.get("grounding_char_spans")
    if not isinstance(rows, list) or not rows:
        raise ValueError("missing target grounding_char_spans row")
    candidates = sorted(
        value for value in (valid_span(span, text) for span in rows[0]) if value is not None
    )
    if not candidates:
        raise ValueError("target has no valid character span")
    return candidates[0]

def choose_anchor_spans(
    row: dict[str, Any], text: str, target_id: int
) -> list[tuple[int, int, int, int]]:
    object_ids = row.get("grounding_object_ids")
    span_rows = row.get("grounding_char_spans")
    if not isinstance(object_ids, list) or not isinstance(span_rows, list):
        return []
    candidates_by_row: list[tuple[int, int, list[tuple[int, int]]]] = []
    seen = {int(target_id)}
    for row_index, (object_id, spans) in enumerate(zip(object_ids[1:], span_rows[1:]), start=1):
        object_id = int(object_id)
        if object_id in seen:
            continue
        seen.add(object_id)
        candidates = sorted(
            value for value in (valid_span(span, text) for span in spans) if value is not None
        )
        if candidates:
            candidates_by_row.append((row_index, object_id, candidates))

    states: list[list[tuple[int, int, int, int]]] = [[]]
    for row_index, object_id, candidates in candidates_by_row:
        expanded = list(states)
        for selected in states:
            for start, end in candidates:
                if any(
                    not (end <= old_start or start >= old_end)
                    for _, _, old_start, old_end in selected
                ):
                    continue
                expanded.append(selected + [(row_index, object_id, start, end)])
        expanded.sort(
            key=lambda selected: (
                -len(selected),
                tuple(item[2] for item in sorted(selected, key=lambda item: item[2])),
            )
        )
        states = expanded[:2048]
    best = states[0] if states else []
    return sorted(best, key=lambda item: item[2])

def tag_anchor_context(
    text: str, selected: list[tuple[int, int, int, int]]
) -> tuple[str, list[dict[str, Any]]]:
    pieces: list[str] = []
    entities: list[dict[str, Any]] = []
    cursor = 0
    answer_cursor = 0
    for row_index, object_id, start, end in selected:
        plain = text[cursor:start]
        pieces.append(plain)
        answer_cursor += len(plain)
        pieces.append("<anc>")
        answer_cursor += len("<anc>")
        content_start = answer_cursor
        phrase = text[start:end]
        pieces.append(phrase)
        answer_cursor += len(phrase)
        content_end = answer_cursor
        pieces.append("</anc>")
        answer_cursor += len("</anc>")
        entities.append({
            "role": "ANC",
            "grounding_row": row_index,
            "object_id": int(object_id),
            "phrase": clean_phrase(phrase),
            "source_char_span": [start, end],
            "context_char_span": [content_start, content_end],
        })
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces), entities

def validate_answer(answer: str) -> None:
    lower = answer.lower()
    if lower.count("<obj>") != 1 or lower.count("</obj>") != 1:
        raise ValueError("generated Answer must contain exactly one OBJ pair")
    if lower.count("<anc>") != lower.count("</anc>"):
        raise ValueError("generated Answer has unbalanced ANC tags")
    stack: list[str] = []
    cursor = 0
    while True:
        positions = [(lower.find(tag, cursor), tag) for tag in TAGS]
        positions = [(position, tag) for position, tag in positions if position >= 0]
        if not positions:
            break
        position, tag = min(positions)
        if tag in {"<obj>", "<anc>"}:
            if stack:
                raise ValueError("generated Answer contains nested tags")
            stack.append(tag[1:4])
        else:
            role = tag[2:5]
            if not stack or stack.pop() != role:
                raise ValueError("generated Answer contains mismatched tags")
        cursor = position + len(tag)
    if stack:
        raise ValueError("generated Answer contains unclosed tags")
    for open_tag, close_tag in (("<obj>", "</obj>"), ("<anc>", "</anc>")):
        start = 0
        while True:
            left = lower.find(open_tag, start)
            if left < 0:
                break
            right = lower.find(close_tag, left + len(open_tag))
            if right < 0 or not clean_phrase(answer[left + len(open_tag):right]):
                raise ValueError("generated Answer contains an empty entity span")
            start = right + len(close_tag)

def build_teacher_answer(row: dict[str, Any]) -> tuple[str, list[dict[str, Any]], int]:
    text = str(row.get("description", "")).strip()
    if not text:
        raise ValueError("empty description")
    if any(tag in text.lower() for tag in TAGS):
        raise ValueError("description already contains reserved OBJ/ANC tags")
    object_ids = row.get("grounding_object_ids")
    if not isinstance(object_ids, list) or not object_ids:
        raise ValueError("missing grounding_object_ids")
    target_start, target_end = choose_target(row, text)
    target_phrase = text[target_start:target_end]
    target_id = int(object_ids[0])
    selected = choose_anchor_spans(row, text, target_id)
    context, anchors = tag_anchor_context(text, selected)
    prefix = "The requested object is <obj>"
    target_answer_start = len(prefix)
    target_answer_end = target_answer_start + len(target_phrase)
    answer = prefix + target_phrase + "</obj>."
    entities = [{
        "role": "OBJ",
        "grounding_row": 0,
        "object_id": target_id,
        "phrase": clean_phrase(target_phrase),
        "source_char_span": [target_start, target_end],
        "answer_char_span": [target_answer_start, target_answer_end],
    }]
    if anchors:
        context_prefix = " The description states: "
        context_offset = len(answer) + len(context_prefix)
        answer += context_prefix + context
        for entity in anchors:
            local_start, local_end = entity.pop("context_char_span")
            entity["answer_char_span"] = [
                context_offset + local_start, context_offset + local_end
            ]
            entities.append(entity)
    validate_answer(answer)
    expected_anchor_rows = max(len(set(map(int, object_ids))) - 1, 0)
    dropped = max(expected_anchor_rows - len(anchors), 0)
    return answer, entities, dropped

def prepare_rows(rows: list[dict[str, Any]], source: str):
    output: list[dict[str, Any]] = []
    dropped_anchors = 0
    for index, original in enumerate(rows):
        row = dict(original)
        try:
            answer, entities, dropped = build_teacher_answer(row)
        except Exception as exc:
            raise ValueError(
                f"{source} annotation {index} cannot produce teacher_answer: {exc}"
            ) from exc
        row["teacher_answer"] = answer
        row["teacher_answer_format"] = FORMAT_VERSION
        row["teacher_answer_source"] = "grounding_span_insertion_no_external_llm"
        row["answer_entities"] = entities
        row["teacher_answer_sha256"] = hashlib.sha256(answer.encode("utf-8")).hexdigest()
        output.append(row)
        dropped_anchors += dropped
    return output, {
        "source": source,
        "input_samples": len(rows),
        "output_samples": len(output),
        "dropped_ambiguous_anchors": dropped_anchors,
        "with_anchors": sum(bool(row["answer_entities"][1:]) for row in output),
    }
