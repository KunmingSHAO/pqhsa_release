"""Quest-style page index + calibrated background channel (2×2 cell).

Page selection matches Quest pages (page-max key scores, top-k pages).
Softmax denominator merges selected-page exact token logits with unselected
pages' calibrated page logits (true max q·k within page) in one softmax — analogous to PQ-HSA
hybrid denom=all_pq. Page **selection** still uses Quest sign-aligned page-max-key scores.
"""
from __future__ import annotations

import math
from types import MethodType
from typing import Any

import torch

import e2e_pq_param_sweep as e2e

QUEST_MIN_PAGES = 3


def quest_aligned_token_budget(p: float, length: int, sink: int = 4, local: int = 128) -> int:
    inner = sink + local + math.ceil(p * max(0, length - sink - local))
    return 16 * math.ceil(inner / 16)


def quest_min_feasible_budget(chunk_size: int = 16) -> int:
    return int(chunk_size) * QUEST_MIN_PAGES


def _quest_calibrated_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    token_budget: int,
    chunk_size: int,
) -> torch.Tensor:
    """Quest page top-k + hybrid softmax (exact selected + page-score background)."""
    bsz, n_heads, _q_len, head_dim = query_states.shape
    seq_length = key_states.shape[-2]
    scale = 1.0 / math.sqrt(head_dim)

    sign = torch.where(
        query_states > 0,
        query_states.new_ones(()),
        query_states.new_full((), -1.0),
    )
    positive_query = query_states * sign
    max_key = key_states * sign
    pad = chunk_size - ((seq_length - 1) % chunk_size + 1)
    if pad:
        fill = torch.finfo(max_key.dtype).min
        max_key = torch.nn.functional.pad(max_key, (0, 0, 0, pad), value=fill)
        value_states = torch.nn.functional.pad(value_states, (0, 0, 0, pad), value=0.0)
    n_pages = max_key.shape[-2] // chunk_size

    keys_signed_paged = max_key.view(bsz, n_heads, n_pages, chunk_size, head_dim)
    page_max = keys_signed_paged.amax(dim=-2)
    page_scores = torch.matmul(positive_query.float(), page_max.transpose(2, 3).float()) * scale
    page_scores_2d = page_scores.squeeze(2)  # [B, H, n_pages] — Quest selector (upper bound)

    if pad:
        keys_orig_paged = torch.nn.functional.pad(key_states, (0, 0, 0, pad), value=0.0)
    else:
        keys_orig_paged = key_states
    keys_orig_paged = keys_orig_paged.view(bsz, n_heads, n_pages, chunk_size, head_dim)

    # Calibrated page logits for the shared softmax: true max q·k within each page.
    page_token_logits = torch.einsum(
        "bhpcd,bhd->bhpc",
        keys_orig_paged.float(),
        query_states.float().squeeze(2),
    ) * scale
    page_logits_2d = page_token_logits.max(dim=-1).values  # [B, H, n_pages]
    page_argmax = page_token_logits.argmax(dim=-1)  # [B, H, n_pages]
    page_token_idx = page_argmax + torch.arange(
        n_pages, device=query_states.device, dtype=page_argmax.dtype
    ).view(1, 1, n_pages) * chunk_size
    page_gather = page_token_idx.unsqueeze(-1).expand(bsz, n_heads, n_pages, head_dim)
    page_values = torch.gather(value_states, 2, page_gather).to(query_states.dtype)

    k_sel = min(max(QUEST_MIN_PAGES, token_budget // chunk_size), n_pages)
    page_scores_4d = page_scores
    _, topk = page_scores_4d.topk(k=k_sel, dim=-1)  # [B, H, 1, k_sel]
    topk_pages = topk.reshape(bsz, n_heads, k_sel)  # [B, H, k_sel]

    token_idx = topk.unsqueeze(-1) * chunk_size + torch.arange(
        chunk_size, device=topk.device, dtype=topk.dtype
    )
    token_idx = token_idx.view(bsz, n_heads, -1)
    valid = token_idx < seq_length
    token_idx = token_idx.clamp(max=seq_length - 1)
    gather_idx = token_idx.unsqueeze(-1).expand(bsz, n_heads, token_idx.shape[-1], head_dim)
    k_sel_t = torch.gather(key_states, 2, gather_idx)
    v_sel_t = torch.gather(value_states, 2, gather_idx)

    sel_logits = torch.matmul(query_states, k_sel_t.transpose(2, 3)) * scale
    sel_logits = sel_logits.masked_fill(~valid.unsqueeze(2), torch.finfo(sel_logits.dtype).min)

    sel_max = sel_logits.float().max(dim=-1).values  # [B, H, 1]
    page_max_logit = page_logits_2d.float().max(dim=-1).values  # [B, H]
    row_max = torch.maximum(sel_max, page_max_logit.unsqueeze(-1))  # [B, H, 1]

    sel_exp = torch.exp(sel_logits.float() - row_max.unsqueeze(-1))
    sel_exp = sel_exp.masked_fill(~valid.unsqueeze(2), 0.0)
    sel_exp_sum = sel_exp.sum(dim=-1)  # [B, H, 1]

    page_exp = torch.exp(page_logits_2d.float() - row_max)
    selected_page_exp = torch.gather(page_exp, 2, topk_pages).sum(dim=2, keepdim=True)
    denom = (page_exp.sum(dim=-1, keepdim=True) - selected_page_exp + sel_exp_sum).clamp_min(
        torch.finfo(page_exp.dtype).tiny
    )

    exact_weights = (sel_exp / denom.unsqueeze(-1)).to(query_states.dtype)
    exact_output = torch.matmul(exact_weights, v_sel_t)

    page_mask = torch.zeros(bsz, n_heads, n_pages, dtype=torch.bool, device=query_states.device)
    page_mask.scatter_(2, topk_pages, True)
    bg_page_exp = page_exp.masked_fill(page_mask, 0.0)
    bg_weight = (bg_page_exp / denom).to(query_states.dtype)
    bg_output = torch.einsum("bhp,bhpd->bhd", bg_weight, page_values).unsqueeze(2)

    return (exact_output + bg_output).to(query_states.dtype)


def _llama_quest_calibrated_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_values: Any = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    del attention_mask, kwargs
    input_shape = hidden_states.shape[:-1]
    layer_idx = int(getattr(self, "layer_idx", getattr(self, "_quest_cal_layer_idx", 0)))
    if len(input_shape) != 2 or input_shape[1] != 1 or past_key_values is None or layer_idx < 2:
        return self._quest_cal_original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=None,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )

    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = e2e.llama_apply_rotary_pos_emb(query_states, key_states, cos, sin)
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = past_key_values.update(
        key_states, value_states, self.layer_idx, cache_kwargs,
    )
    key_states = e2e.llama_repeat_kv(key_states, self.num_key_value_groups)
    value_states = e2e.llama_repeat_kv(value_states, self.num_key_value_groups)
    context = _quest_calibrated_attention(
        query_states,
        key_states,
        value_states,
        token_budget=int(self._quest_cal_token_budget),
        chunk_size=int(self._quest_cal_chunk_size),
    )
    attn_output = context.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def _qwen3_quest_calibrated_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_values: Any = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Quest-calibrated decode path for Qwen3: QK-norm + qwen3 RoPE, same hybrid softmax."""
    del attention_mask, kwargs
    input_shape = hidden_states.shape[:-1]
    layer_idx = int(getattr(self, "layer_idx", getattr(self, "_quest_cal_layer_idx", 0)))
    if len(input_shape) != 2 or input_shape[1] != 1 or past_key_values is None or layer_idx < 2:
        return self._quest_cal_original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=None,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )

    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = e2e.qwen3_apply_rotary_pos_emb(query_states, key_states, cos, sin)
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = past_key_values.update(
        key_states, value_states, self.layer_idx, cache_kwargs,
    )
    key_states = e2e.qwen3_repeat_kv(key_states, self.num_key_value_groups)
    value_states = e2e.qwen3_repeat_kv(value_states, self.num_key_value_groups)
    context = _quest_calibrated_attention(
        query_states,
        key_states,
        value_states,
        token_budget=int(self._quest_cal_token_budget),
        chunk_size=int(self._quest_cal_chunk_size),
    )
    attn_output = context.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def install_quest_calibrated_patch(
    model: torch.nn.Module,
    *,
    token_budget: int,
    chunk_size: int,
) -> list[torch.nn.Module]:
    model_type = getattr(model.config, "model_type", "")
    if model_type not in {"llama", "qwen2", "mistral", "qwen3"}:
        raise ValueError(f"unsupported model_type for quest_calibrated patch: {model_type}")
    if chunk_size <= 0:
        raise ValueError(f"quest_calibrated chunk_size must be positive, got {chunk_size}")
    if token_budget <= 0:
        raise ValueError(f"quest_calibrated token_budget must be positive, got {token_budget}")

    forward = _qwen3_quest_calibrated_forward if model_type == "qwen3" else _llama_quest_calibrated_forward
    patched = []
    for layer_idx, decoder_layer in enumerate(model.model.layers):
        attn = decoder_layer.self_attn
        if hasattr(attn, "_quest_cal_original_forward"):
            raise RuntimeError("quest_calibrated patch already installed")
        attn._quest_cal_original_forward = attn.forward
        attn._quest_cal_token_budget = int(token_budget)
        attn._quest_cal_chunk_size = int(chunk_size)
        attn._quest_cal_layer_idx = layer_idx
        attn.forward = MethodType(forward, attn)
        patched.append(attn)
    return patched


def restore_quest_calibrated_patch(patched: list[torch.nn.Module]) -> None:
    for attn in patched:
        original = getattr(attn, "_quest_cal_original_forward", None)
        if original is not None:
            attn.forward = original
        for name in (
            "_quest_cal_original_forward",
            "_quest_cal_token_budget",
            "_quest_cal_chunk_size",
            "_quest_cal_layer_idx",
        ):
            if hasattr(attn, name):
                delattr(attn, name)


def report_metadata(
    *,
    token_budget: int,
    chunk_size: int,
    budget_p: float | None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "token_budget": int(token_budget),
        "quest_chunk_size": int(chunk_size),
        "quest_first_two_layers_dense": True,
        "quest_calibrated_background": "page_true_max_logit",
        "quest_calibrated_value_mode": "page_argmax_token",
        "quest_calibrated_selector": "quest_page_max_key",
        "quest_calibrated_denominator": "selected_exact_plus_unselected_page_scores",
    }
    if budget_p is not None:
        meta["quest_budget_p"] = float(budget_p)
        meta["quest_budget_mode"] = "grid"
    return meta
