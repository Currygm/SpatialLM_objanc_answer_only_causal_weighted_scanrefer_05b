"""Pure deterministic Answer-conditioning curriculum helpers."""
from __future__ import annotations

import hashlib


def normalize_phrase(value: str) -> str:
    return " ".join(str(value).casefold().split())


def teacher_ratio(epoch: int, optimizer_step: int, steps_per_epoch: int) -> float:
    """Epoch is zero-based; epochs 1-5 teacher, 6-15 linearly decay."""
    if epoch < 5:
        return 1.0
    if epoch >= 15:
        return 0.0
    within = (optimizer_step % max(steps_per_epoch, 1)) / max(steps_per_epoch, 1)
    return max(0.0, 1.0 - ((epoch - 5) + within) / 10.0)


def deterministic_teacher(seed: int, epoch: int, sample_id: str, ratio: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{epoch}:{sample_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return value < ratio

