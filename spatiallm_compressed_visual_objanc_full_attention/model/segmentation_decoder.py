"""Nine-layer single-query UniVLG decoder with Qwen role embeddings."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence


def _pad(rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not rows or any(row.ndim != 2 or row.shape[0] == 0 for row in rows):
        raise ValueError("Expected non-empty [length,channels] rows")
    padded = pad_sequence(rows, batch_first=True)
    valid = torch.zeros(padded.shape[:2], device=padded.device, dtype=torch.bool)
    for index, row in enumerate(rows):
        valid[index, : row.shape[0]] = True
    return padded, valid


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, output_dim)
        ])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            value = layer(value)
            if index < 2:
                value = F.relu(value)
        return value


class FullCycleDecoderBlock(nn.Module):
    """[Query+Text] <- Visual, joint self update, then Visual <- joint."""

    def __init__(self, hidden: int, heads: int, ffn: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.joint_visual = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.joint_visual_norm = nn.LayerNorm(hidden)
        self.joint_self = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.joint_self_norm = nn.LayerNorm(hidden)
        self.joint_ffn = nn.Sequential(nn.Linear(hidden, ffn), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ffn, hidden), nn.Dropout(dropout))
        self.joint_ffn_norm = nn.LayerNorm(hidden)
        self.visual_joint = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.visual_joint_norm = nn.LayerNorm(hidden)
        self.visual_ffn = nn.Sequential(nn.Linear(hidden, ffn), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ffn, hidden), nn.Dropout(dropout))
        self.visual_ffn_norm = nn.LayerNorm(hidden)

    def forward(self, joint, joint_pos, joint_valid, visual, visual_pos, visual_valid, blocked):
        batch, joint_len, visual_len = blocked.shape
        mask = blocked[:, None].expand(batch, self.heads, joint_len, visual_len).reshape(batch * self.heads, joint_len, visual_len)
        update, _ = self.joint_visual(joint + joint_pos, visual + visual_pos, visual, attn_mask=mask, need_weights=False)
        joint = self.joint_visual_norm(joint + update)
        update, _ = self.joint_self(joint + joint_pos, joint + joint_pos, joint, key_padding_mask=~joint_valid, need_weights=False)
        joint = self.joint_self_norm(joint + update)
        joint = self.joint_ffn_norm(joint + self.joint_ffn(joint)).masked_fill(~joint_valid[:, :, None], 0)
        update, _ = self.visual_joint(visual + visual_pos, joint + joint_pos, joint, key_padding_mask=~joint_valid, need_weights=False)
        visual = self.visual_joint_norm(visual + update)
        visual = self.visual_ffn_norm(visual + self.visual_ffn(visual)).masked_fill(~visual_valid[:, :, None], 0)
        return joint, visual


class ObjAncFullCycleDecoder(nn.Module):
    version = "qwen_objanc_single_query_univlg_fullcycle_9layer_v1"

    def __init__(self, *, visual_dim=512, text_dim=896, hidden_dim=256, num_queries=50,
                 num_layers=9, num_heads=8, feedforward_dim=1024, dropout=0.0,
                 max_text_tokens=384, mask_threshold=0.5, text_temperature=0.07) -> None:
        super().__init__()
        if num_layers != 9 or num_queries != 50:
            raise ValueError("This stage requires 50 queries and nine decoder layers")
        self.hidden_dim, self.num_queries, self.num_layers = hidden_dim, num_queries, num_layers
        self.num_heads, self.max_text_tokens = num_heads, max_text_tokens
        self.mask_threshold, self.text_temperature = mask_threshold, text_temperature
        self.visual_connector = nn.Sequential(nn.Linear(visual_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.text_connector = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.role_embedding = nn.Embedding(4, hidden_dim)
        self.xyz_position = nn.Sequential(nn.Linear(3, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.text_position = nn.Embedding(max_text_tokens, hidden_dim)
        self.query_content = nn.Embedding(num_queries, hidden_dim)
        self.query_position = nn.Embedding(num_queries, hidden_dim)
        self.layers = nn.ModuleList(FullCycleDecoderBlock(hidden_dim, num_heads, feedforward_dim, dropout) for _ in range(num_layers))
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.mask_embedding = MLP(hidden_dim, hidden_dim, hidden_dim)
        self.text_query_projection = nn.Linear(hidden_dim, hidden_dim)

    def _predict(self, query, visual, visual_valid, text, text_valid):
        query = self.output_norm(query)
        masks = torch.einsum("bqd,bnd->bqn", self.mask_embedding(query), visual) / math.sqrt(self.hidden_dim)
        masks = masks.masked_fill(~visual_valid[:, None], 0)
        text_logits = torch.einsum("bqd,btd->bqt", F.normalize(self.text_query_projection(query), dim=-1), F.normalize(text, dim=-1))
        text_logits = (text_logits / self.text_temperature).masked_fill(~text_valid[:, None], 0)
        return masks, text_logits

    def _blocked(self, masks, visual_valid, text_valid):
        query = masks.sigmoid().detach() < self.mask_threshold
        query |= ~visual_valid[:, None]
        for batch, index in torch.nonzero(query.all(-1), as_tuple=False).tolist():
            query[batch, index, visual_valid[batch]] = False
        language = (~visual_valid[:, None]).expand(-1, text_valid.shape[1], -1)
        return torch.cat([query, language], dim=1)

    def forward(self, superpoint_hidden: list[torch.Tensor], text_hidden: list[torch.Tensor],
                superpoint_xyz: list[torch.Tensor], token_roles: list[torch.Tensor]) -> dict[str, Any]:
        visual_in, visual_valid = _pad(superpoint_hidden)
        text_in, text_valid = _pad(text_hidden)
        xyz, xyz_valid = _pad(superpoint_xyz)
        if not torch.equal(visual_valid, xyz_valid):
            raise ValueError("Superpoint features and XYZ do not align")
        if text_in.shape[1] > self.max_text_tokens:
            raise ValueError(f"Question+Answer has {text_in.shape[1]} tokens; maximum is {self.max_text_tokens}")
        role_rows = [row.to(text_in.device, torch.long) for row in token_roles]
        role_pad = pad_sequence(role_rows, batch_first=True)
        if role_pad.shape != text_valid.shape or bool((role_pad[text_valid] < 0).any()) or bool((role_pad[text_valid] > 3).any()):
            raise ValueError("Token roles must align and be QUESTION/OBJ/ANC/OTHER")
        visual = self.visual_connector(visual_in).masked_fill(~visual_valid[:, :, None], 0)
        text = (self.text_connector(text_in) + self.role_embedding(role_pad)).masked_fill(~text_valid[:, :, None], 0)
        visual_pos = self.xyz_position(xyz.float()).to(visual.dtype).masked_fill(~visual_valid[:, :, None], 0)
        text_pos = self.text_position(torch.arange(text.shape[1], device=text.device))[None].expand(text.shape[0], -1, -1)
        text_pos = text_pos.masked_fill(~text_valid[:, :, None], 0)
        batch = visual.shape[0]
        query = self.query_content.weight[None].expand(batch, -1, -1)
        query_pos = self.query_position.weight[None].expand(batch, -1, -1)
        query_valid = torch.ones((batch, self.num_queries), device=visual.device, dtype=torch.bool)
        mask_rows, text_rows = [], []
        masks, logits = self._predict(query, visual, visual_valid, text, text_valid)
        mask_rows.append(masks); text_rows.append(logits)
        for layer in self.layers:
            joint = torch.cat([query, text], dim=1)
            joint_pos = torch.cat([query_pos, text_pos], dim=1)
            joint_valid = torch.cat([query_valid, text_valid], dim=1)
            joint, visual = layer(joint, joint_pos, joint_valid, visual, visual_pos, visual_valid, self._blocked(masks, visual_valid, text_valid))
            query, text = joint[:, :self.num_queries], joint[:, self.num_queries:]
            masks, logits = self._predict(query, visual, visual_valid, text, text_valid)
            mask_rows.append(masks); text_rows.append(logits)
        return {
            "pred_masks": mask_rows[-1], "pred_logits": text_rows[-1], "valid_mask": visual_valid,
            "text_valid_mask": text_valid, "updated_queries": query,
            "updated_text_features": text, "updated_visual_features": visual,
            "aux_outputs": [{"pred_masks": m, "pred_logits": t, "text_valid_mask": text_valid} for m, t in zip(mask_rows[:-1], text_rows[:-1])],
        }

