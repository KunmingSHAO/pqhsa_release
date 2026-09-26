"""SnapKV (NeurIPS 2024) baseline for the task-utility harness.

Paper (arXiv:2404.14469): after dense prefill, the last observation window
votes on prefix tokens via attention mass, 1D max-pooling clusters neighbors,
and top-k prefix KV (+ observation window) are kept for decode.

This module is harness-local (do not put it under pq_hsa). Discarded KV are
masked out of decode attention; the full cache may stay in memory.
"""
from __future__ import annotations

import math
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb as llama_apply_rotary_pos_emb,
)
from transformers.models.llama.modeling_llama import repeat_kv as llama_repeat_kv

# Paper LongBench defaults (Sec. 5.3): max-pool kernel 7, observation window 32.
SNAPKV_OBS_WINDOW = 32
SNAPKV_POOL_KERNEL = 7
SNAPKV_POOL = "max"
SNAPKV_SINK_DEFAULT = 4
SNAPKV_LOCAL_DEFAULT = 128
SNAPKV_CLUSTER_SIZE = 16


def aligned_token_budget(
    p: float,
    length: int,
    sink: int = SNAPKV_SINK_DEFAULT,
    local: int = SNAPKV_LOCAL_DEFAULT,
) -> int:
    """Aligned-budget grid formula: 16*ceil((sink+local+ceil(p*max(0,L-sink-local)))/16)."""
    inner = sink + local + math.ceil(p * max(0, length - sink - local))
    return 16 * math.ceil(inner / 16)


def min_feasible_budget(
    sink: int = SNAPKV_SINK_DEFAULT,
    obs_window: int = SNAPKV_OBS_WINDOW,
    local_window: int = SNAPKV_LOCAL_DEFAULT,
) -> int:
    """Named reserved pieces: sink + observation window + local window."""
    return int(sink) + int(obs_window) + int(local_window)


def _crop_cache_to_len(cache: Any, new_len: int) -> None:
    for layer in cache.layers:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None:
            continue
        layer.keys = keys[..., :new_len, :].contiguous()
        layer.values = values[..., :new_len, :].contiguous()


def compute_keep_indices(
    query_obs: torch.Tensor,
    key_full: torch.Tensor,
    *,
    token_budget: int,
    sink: int,
    local_window: int,
    obs_window: int,
    pool_kernel: int,
    num_key_value_groups: int,
) -> torch.Tensor:
    """Vote with obs-window queries, GQA-pool scores, return keep indices.

    Args:
        query_obs: [B, H_q, obs, D] RoPE-applied observation-window queries.
        key_full: [B, H_kv, L, D] full-prompt keys (prefix + observation).
    Returns:
        keep_idx: [B, H_kv, n_keep] positions in [0, L), per KV head.
    """
    bsz, n_q, obs, head_dim = query_obs.shape
    n_kv = key_full.shape[1]
    seq_len = key_full.shape[-2]
    device = query_obs.device
    if obs != obs_window:
        raise ValueError(f"query_obs length {obs} != obs_window {obs_window}")
    if seq_len < obs_window:
        idx = torch.arange(seq_len, device=device)
        return idx.view(1, 1, -1).expand(bsz, n_kv, seq_len).contiguous()
    if token_budget >= seq_len:
        idx = torch.arange(seq_len, device=device)
        return idx.view(1, 1, -1).expand(bsz, n_kv, seq_len).contiguous()

    tail = max(int(local_window), int(obs_window))
    sink = int(sink)
    if sink + tail >= seq_len:
        idx = torch.arange(seq_len, device=device)
        return idx.view(1, 1, -1).expand(bsz, n_kv, seq_len).contiguous()

    groups = int(num_key_value_groups)
    if n_q != n_kv * groups:
        raise ValueError(f"GQA mismatch: H_q={n_q} H_kv={n_kv} groups={groups}")

    prefix_len = seq_len - obs_window
    key_rep = llama_repeat_kv(key_full, groups)
    scale = 1.0 / math.sqrt(head_dim)
    logits = torch.matmul(
        query_obs.float(), key_rep.transpose(2, 3).float()
    ) * scale
    q_pos = torch.arange(prefix_len, seq_len, device=device)
    k_pos = torch.arange(seq_len, device=device)
    causal = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
    logits = logits.masked_fill(causal.view(1, 1, obs, seq_len), torch.finfo(logits.dtype).min)
    attn = torch.softmax(logits, dim=-1)
    vote_q = attn[..., :prefix_len].sum(dim=-2)
    vote = vote_q.view(bsz, n_kv, groups, prefix_len).sum(dim=2)

    pooled = F.max_pool1d(
        vote.reshape(bsz * n_kv, 1, prefix_len),
        kernel_size=int(pool_kernel),
        stride=1,
        padding=int(pool_kernel) // 2,
    ).view(bsz, n_kv, prefix_len)

    cand = torch.ones(prefix_len, dtype=torch.bool, device=device)
    cand[:sink] = False
    extra_tail = max(0, tail - obs_window)
    if extra_tail:
        cand[prefix_len - extra_tail :] = False
    scores = pooled.masked_fill(~cand.view(1, 1, prefix_len), torch.finfo(pooled.dtype).min)

    n_keep_target = min(int(token_budget), seq_len)
    n_always = sink + tail
    n_select = max(0, n_keep_target - n_always)
    n_select = min(n_select, int(cand.sum().item()))

    always = torch.cat(
        (
            torch.arange(sink, device=device),
            torch.arange(seq_len - tail, seq_len, device=device),
        ),
        dim=0,
    )
    always = always.view(1, 1, -1).expand(bsz, n_kv, -1)
    if n_select <= 0:
        keep = always
    else:
        sel = scores.topk(k=n_select, dim=-1).indices
        keep = torch.cat((always, sel), dim=-1)
    keep, _ = keep.sort(dim=-1)
    return keep.contiguous()


def _proj_q(attn: torch.nn.Module, hidden_states: torch.Tensor, hidden_shape: tuple) -> torch.Tensor:
    """q_proj (+ QK-norm when the architecture has it) -> [b, n_q, t, d].

    Llama / Qwen2 / Mistral have no ``q_norm``; for them this is bit-identical to
    the previous ``attn.q_proj(...).view(...).transpose(1, 2)``.
    """
    out = attn.q_proj(hidden_states).view(hidden_shape)
    q_norm = getattr(attn, "q_norm", None)
    if q_norm is not None:
        out = q_norm(out)
    return out.transpose(1, 2)


def _proj_k(attn: torch.nn.Module, hidden_states: torch.Tensor, hidden_shape: tuple) -> torch.Tensor:
    """k_proj (+ QK-norm when the architecture has it) -> [b, n_kv, t, d]."""
    out = attn.k_proj(hidden_states).view(hidden_shape)
    k_norm = getattr(attn, "k_norm", None)
    if k_norm is not None:
        out = k_norm(out)
    return out.transpose(1, 2)


def _record_keep_indices(
    attn: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    past_key_values: Any,
) -> None:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    query_states = _proj_q(attn, hidden_states, hidden_shape)
    cos, sin = position_embeddings
    query_states, _ = llama_apply_rotary_pos_emb(query_states, query_states, cos, sin)
    layer_idx = int(getattr(attn, "layer_idx", getattr(attn, "_snapkv_layer_idx", 0)))
    key_full = past_key_values.layers[layer_idx].keys
    groups = int(attn.num_key_value_groups)
    keep = compute_keep_indices(
        query_states,
        key_full,
        token_budget=int(attn._snapkv_token_budget),
        sink=int(attn._snapkv_sink),
        local_window=int(attn._snapkv_local_window),
        obs_window=int(attn._snapkv_obs_window),
        pool_kernel=int(attn._snapkv_pool_kernel),
        num_key_value_groups=groups,
    )
    attn._snapkv_keep_idx = keep
    attn._snapkv_prefill_len = int(key_full.shape[-2])


def _snapkv_sparse_decode_attn(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attn: torch.nn.Module,
) -> torch.Tensor:
    keep_idx = attn._snapkv_keep_idx
    prefill_len = int(attn._snapkv_prefill_len)
    bsz, n_q, q_len, head_dim = query_states.shape
    del q_len
    n_kv = key_states.shape[1]
    if keep_idx.dim() != 3:
        raise RuntimeError(f"snapkv keep_idx rank {keep_idx.dim()} != 3")
    if keep_idx.shape[0] == 1 and bsz > 1:
        keep_idx = keep_idx.expand(bsz, -1, -1)
    n_keep = keep_idx.shape[-1]
    gather_idx = keep_idx.unsqueeze(-1).expand(bsz, n_kv, n_keep, head_dim)
    k_sel = torch.gather(key_states[:, :, :prefill_len, :], 2, gather_idx)
    v_sel = torch.gather(value_states[:, :, :prefill_len, :], 2, gather_idx)
    if key_states.shape[-2] > prefill_len:
        k_sel = torch.cat((k_sel, key_states[:, :, prefill_len:, :]), dim=2)
        v_sel = torch.cat((v_sel, value_states[:, :, prefill_len:, :]), dim=2)
    groups = int(attn.num_key_value_groups)
    k_sel = llama_repeat_kv(k_sel, groups)
    v_sel = llama_repeat_kv(v_sel, groups)
    return F.scaled_dot_product_attention(
        query_states, k_sel, v_sel, attn_mask=None, dropout_p=0.0, is_causal=False
    )


def cluster_evicted_pages(
    keys: torch.Tensor,
    values: torch.Tensor,
    keep_idx: torch.Tensor,
    cluster_size: int = SNAPKV_CLUSTER_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Page-cluster discarded tokens. Host-native; no PQ.

    Returns cluster_k, cluster_v [B, H_kv, P, D] and valid [B, H_kv, P].
    A page is valid iff it contains at least one evicted token. Cluster key/value
    are the mean of those evicted tokens (SnapKV's cheap leftover representation).
    """
    bsz, n_kv, seq_len, head_dim = keys.shape
    keep_mask = torch.zeros(bsz, n_kv, seq_len, dtype=torch.bool, device=keys.device)
    keep_mask.scatter_(2, keep_idx.to(device=keys.device, dtype=torch.long), True)
    evicted = ~keep_mask
    pad = (cluster_size - (seq_len % cluster_size)) % cluster_size
    if pad:
        evicted = F.pad(evicted, (0, pad), value=False)
        keys = F.pad(keys, (0, 0, 0, pad), value=0.0)
        values = F.pad(values, (0, 0, 0, pad), value=0.0)
    n_pages = evicted.shape[-1] // cluster_size
    evicted_p = evicted.view(bsz, n_kv, n_pages, cluster_size)
    keys_p = keys.view(bsz, n_kv, n_pages, cluster_size, head_dim)
    values_p = values.view(bsz, n_kv, n_pages, cluster_size, head_dim)
    n_e = evicted_p.sum(dim=-1)
    valid = n_e > 0
    denom = n_e.clamp_min(1).unsqueeze(-1).to(dtype=torch.float32)
    evicted_f = evicted_p.unsqueeze(-1).to(dtype=torch.float32)
    cluster_k = (keys_p.float() * evicted_f).sum(dim=-2) / denom
    cluster_v = (values_p.float() * evicted_f).sum(dim=-2) / denom
    return cluster_k.to(keys.dtype), cluster_v.to(values.dtype), valid


def build_and_store_clusters(
    cache: Any,
    patched: list[torch.nn.Module],
    cluster_size: int = SNAPKV_CLUSTER_SIZE,
) -> None:
    """Build evicted-page clusters from the full prefill cache (before compress)."""
    for attn in patched:
        layer_idx = int(getattr(attn, "layer_idx", getattr(attn, "_snapkv_layer_idx", 0)))
        keep = getattr(attn, "_snapkv_keep_idx", None)
        if keep is None:
            raise RuntimeError(f"snapkv cluster missing keep_idx at layer {layer_idx}")
        layer = cache.layers[layer_idx]
        if layer.keys is None or layer.values is None:
            raise RuntimeError(f"snapkv cluster missing cache at layer {layer_idx}")
        cluster_k, cluster_v, valid = cluster_evicted_pages(
            layer.keys, layer.values, keep, cluster_size=int(cluster_size)
        )
        attn._snapkv_cluster_k = cluster_k
        attn._snapkv_cluster_v = cluster_v
        attn._snapkv_cluster_valid = valid
        attn._snapkv_cluster_size = int(cluster_size)


def _snapkv_channel_decode_attn(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attn: torch.nn.Module,
) -> torch.Tensor:
    """Exact attention over the compressed keep-set + evicted-page background.

    After compress, key/value are the selected tokens (+ newly generated).
    Background logits are q · cluster_mean_k (SnapKV's leftover representation),
    not a PQ score.
    """
    bsz, n_q, q_len, head_dim = query_states.shape
    del q_len
    groups = int(attn.num_key_value_groups)
    k_rep = llama_repeat_kv(key_states, groups)
    v_rep = llama_repeat_kv(value_states, groups)
    scale = 1.0 / math.sqrt(head_dim)
    sel_logits = torch.matmul(query_states.float(), k_rep.transpose(2, 3).float()) * scale

    cluster_k = llama_repeat_kv(attn._snapkv_cluster_k, groups)
    cluster_v = llama_repeat_kv(attn._snapkv_cluster_v, groups)
    valid = attn._snapkv_cluster_valid
    if groups > 1:
        valid_q = valid.repeat_interleave(groups, dim=1)
    else:
        valid_q = valid
    bg_logits = torch.matmul(query_states.float(), cluster_k.transpose(2, 3).float()) * scale
    bg_logits = bg_logits.masked_fill(~valid_q.unsqueeze(2), torch.finfo(bg_logits.dtype).min)

    sel_max = sel_logits.max(dim=-1).values
    bg_max = bg_logits.max(dim=-1).values
    # If every cluster is invalid, bg_logits are all -inf; max is -inf.
    bg_max = torch.where(valid_q.any(dim=-1, keepdim=True), bg_max, sel_max)
    row_max = torch.maximum(sel_max, bg_max)

    sel_exp = torch.exp(sel_logits - row_max.unsqueeze(-1))
    bg_exp = torch.exp(bg_logits - row_max.unsqueeze(-1))
    bg_exp = bg_exp.masked_fill(~valid_q.unsqueeze(2), 0.0)
    denom = (sel_exp.sum(dim=-1) + bg_exp.sum(dim=-1)).clamp_min(torch.finfo(sel_exp.dtype).tiny)

    exact_out = torch.matmul((sel_exp / denom.unsqueeze(-1)).to(query_states.dtype), v_rep)
    bg_w = (bg_exp.squeeze(2) / denom).to(query_states.dtype)
    bg_out = torch.einsum("bhp,bhpd->bhd", bg_w, cluster_v).unsqueeze(2)
    return (exact_out + bg_out).to(query_states.dtype)


# Name kept for continuity; now also serves qwen3 / qwen3_moe.
def _llama_snapkv_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_values: Any = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    del kwargs
    input_shape = hidden_states.shape[:-1]
    q_len = input_shape[1] if len(input_shape) == 2 else 0
    keep_idx = getattr(self, "_snapkv_keep_idx", None)
    selecting = bool(getattr(self, "_snapkv_selecting", False))
    if q_len != 1 or past_key_values is None or keep_idx is None or selecting:
        # The mask must be forwarded: with attention_mask=None and q_len>1, sdpa
        # falls back to is_causal with upper-left alignment and transformers then
        # slices K/V down to the first q_len positions, so the observation window
        # would only ever see the start of the prompt.
        if (
            selecting
            and q_len > 1
            and attention_mask is None
            and past_key_values is not None
            and int(past_key_values.get_seq_length()) > 0
        ):
            raise RuntimeError(
                "snapkv observation re-forward got attention_mask=None with "
                f"q_len={q_len} over a non-empty cache; sdpa would truncate KV to "
                f"{q_len} positions. Materialize the causal mask "
                "(create_causal_mask / allow_is_causal_skip=False)."
            )
        out = self._snapkv_original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        if selecting and q_len > 1 and past_key_values is not None:
            _record_keep_indices(self, hidden_states, position_embeddings, past_key_values)
        return out

    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = _proj_q(self, hidden_states, hidden_shape)
    key_states = _proj_k(self, hidden_states, hidden_shape)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = llama_apply_rotary_pos_emb(query_states, key_states, cos, sin)
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = past_key_values.update(
        key_states, value_states, self.layer_idx, cache_kwargs,
    )
    if bool(getattr(self, "_snapkv_use_channel", False)) and getattr(self, "_snapkv_cluster_k", None) is not None:
        context = _snapkv_channel_decode_attn(query_states, key_states, value_states, self)
    else:
        context = _snapkv_sparse_decode_attn(query_states, key_states, value_states, self)
    attn_output = context.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def install_snapkv_patch(
    model: torch.nn.Module,
    *,
    token_budget: int,
    sink_tokens: int = SNAPKV_SINK_DEFAULT,
    local_window: int = SNAPKV_LOCAL_DEFAULT,
    obs_window: int = SNAPKV_OBS_WINDOW,
    pool_kernel: int = SNAPKV_POOL_KERNEL,
    use_denom_channel: bool = False,
) -> list[torch.nn.Module]:
    model_type = getattr(model.config, "model_type", "")
    # qwen3 / qwen3_moe share llama's RoPE and repeat_kv (verified identical source
    # in transformers 5.15.1); the only extra step is QK-norm, handled by _proj_q/_proj_k.
    if model_type not in {"llama", "qwen2", "mistral", "qwen3", "qwen3_moe"}:
        raise ValueError(f"unsupported model_type for snapkv patch: {model_type}")
    if token_budget <= 0:
        raise ValueError(f"snapkv token_budget must be positive, got {token_budget}")
    if obs_window <= 0 or pool_kernel <= 0:
        raise ValueError("snapkv obs_window and pool_kernel must be positive")

    patched: list[torch.nn.Module] = []
    for layer_idx, decoder_layer in enumerate(model.model.layers):
        attn = decoder_layer.self_attn
        if hasattr(attn, "_snapkv_original_forward"):
            raise RuntimeError("snapkv patch already installed")
        attn._snapkv_original_forward = attn.forward
        attn._snapkv_token_budget = int(token_budget)
        attn._snapkv_sink = int(sink_tokens)
        attn._snapkv_local_window = int(local_window)
        attn._snapkv_obs_window = int(obs_window)
        attn._snapkv_pool_kernel = int(pool_kernel)
        attn._snapkv_layer_idx = layer_idx
        attn._snapkv_keep_idx = None
        attn._snapkv_prefill_len = None
        attn._snapkv_selecting = False
        attn._snapkv_use_channel = bool(use_denom_channel)
        attn._snapkv_cluster_k = None
        attn._snapkv_cluster_v = None
        attn._snapkv_cluster_valid = None
        attn._snapkv_cluster_size = SNAPKV_CLUSTER_SIZE
        attn.forward = MethodType(_llama_snapkv_forward, attn)
        patched.append(attn)
    return patched


def restore_snapkv_patch(patched: list[torch.nn.Module]) -> None:
    for attn in patched:
        original = getattr(attn, "_snapkv_original_forward", None)
        if original is not None:
            attn.forward = original
        for name in (
            "_snapkv_original_forward",
            "_snapkv_token_budget",
            "_snapkv_sink",
            "_snapkv_local_window",
            "_snapkv_obs_window",
            "_snapkv_pool_kernel",
            "_snapkv_layer_idx",
            "_snapkv_keep_idx",
            "_snapkv_prefill_len",
            "_snapkv_selecting",
            "_snapkv_use_channel",
            "_snapkv_cluster_k",
            "_snapkv_cluster_v",
            "_snapkv_cluster_valid",
            "_snapkv_cluster_size",
        ):
            if hasattr(attn, name):
                delattr(attn, name)


def _snapshot_cache(cache: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    snap: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer in cache.layers:
        if layer.keys is None or layer.values is None:
            raise RuntimeError("snapkv cache snapshot missing keys/values")
        snap.append((layer.keys.clone(), layer.values.clone()))
    return snap


def _restore_cache(cache: Any, snapshot: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    if len(cache.layers) != len(snapshot):
        raise RuntimeError("snapkv cache snapshot depth mismatch")
    for layer, (keys, values) in zip(cache.layers, snapshot):
        layer.keys = keys
        layer.values = values


def run_observation_selection(
    model: torch.nn.Module,
    cache: Any,
    input_ids: torch.Tensor,
) -> None:
    """Vote with a cropped obs-window re-forward; restore original prefill KV."""
    attn0 = model.model.layers[0].self_attn
    obs = int(attn0._snapkv_obs_window)
    prompt_len = int(input_ids.shape[1])
    n_kv = int(getattr(model.config, "num_key_value_heads", 0) or 1)
    if prompt_len <= obs:
        idx = torch.arange(prompt_len, device=input_ids.device)
        keep = idx.view(1, 1, -1).expand(1, n_kv, prompt_len).contiguous()
        for decoder_layer in model.model.layers:
            attn = decoder_layer.self_attn
            attn._snapkv_keep_idx = keep
            attn._snapkv_prefill_len = prompt_len
        return

    snapshot = _snapshot_cache(cache)
    _crop_cache_to_len(cache, prompt_len - obs)
    for decoder_layer in model.model.layers:
        decoder_layer.self_attn._snapkv_selecting = True
    try:
        with torch.no_grad():
            model(
                input_ids=input_ids[:, -obs:],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
    finally:
        for decoder_layer in model.model.layers:
            decoder_layer.self_attn._snapkv_selecting = False
        _restore_cache(cache, snapshot)

    for decoder_layer in model.model.layers:
        attn = decoder_layer.self_attn
        if attn._snapkv_keep_idx is None:
            raise RuntimeError(
                f"snapkv selection failed at layer {getattr(attn, 'layer_idx', '?')}"
            )


def compress_selected_cache(cache: Any, patched: list[torch.nn.Module]) -> None:
    """Replace each layer's prefill KV with the per-head SnapKV selection.

    Decode then uses the original attention path on the compressed cache, matching
    the paper (discarded KV do not participate). New tokens append as usual.
    """
    for attn in patched:
        layer_idx = int(getattr(attn, "layer_idx", getattr(attn, "_snapkv_layer_idx", 0)))
        keep = getattr(attn, "_snapkv_keep_idx", None)
        if keep is None:
            raise RuntimeError(f"snapkv compress missing keep_idx at layer {layer_idx}")
        layer = cache.layers[layer_idx]
        keys, values = layer.keys, layer.values
        if keys is None or values is None:
            raise RuntimeError(f"snapkv compress missing cache at layer {layer_idx}")
        bsz, n_kv, _, head_dim = keys.shape
        if keep.shape[0] == 1 and bsz > 1:
            keep = keep.expand(bsz, -1, -1)
        keep = keep.to(device=keys.device, dtype=torch.long)
        n_keep = keep.shape[-1]
        gather_idx = keep.unsqueeze(-1).expand(bsz, n_kv, n_keep, head_dim)
        layer.keys = torch.gather(keys, 2, gather_idx).contiguous()
        layer.values = torch.gather(values, 2, gather_idx).contiguous()
        attn._snapkv_prefill_len = int(n_keep)


def generate_after_compression(
    model: torch.nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    cache: Any,
    gen_steps: int,
    *,
    prompt_len: int,
) -> str:
    """Greedy decode on the compressed cache, feeding true prompt RoPE positions.

    Retained keys keep the RoPE phase of their original prompt positions, but
    ``compress_selected_cache`` shrinks the cache, so the default position_ids
    (``arange(q_len) + cache.get_seq_length()``) would place the query thousands
    of positions *before* its own context. Positions here match what the dense
    path would use for the same step, keeping the decode protocol identical.
    """
    device = next(model.parameters()).device
    if input_ids.device != device:
        input_ids = input_ids.to(device)
    current_ids = input_ids[:, prompt_len - 1 : prompt_len]
    gen_ids: list[int] = []
    with torch.no_grad():
        for step in range(int(gen_steps)):
            position_ids = torch.tensor(
                [[prompt_len + step]], dtype=torch.long, device=device
            )
            out = model(
                input_ids=current_ids,
                past_key_values=cache,
                position_ids=position_ids,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_ids.append(int(next_tok[0, 0]))
            current_ids = next_tok
            del out
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def report_metadata(
    *,
    token_budget: int,
    sink: int,
    local_window: int,
    obs_window: int,
    pool_kernel: int,
    budget_p: float | None,
    use_denom_channel: bool = False,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "token_budget": int(token_budget),
        "snapkv_obs_window": int(obs_window),
        "snapkv_pool_kernel": int(pool_kernel),
        "snapkv_pool": SNAPKV_POOL,
        "snapkv_sink": int(sink),
        "snapkv_local_window": int(local_window),
        "snapkv_min_feasible": min_feasible_budget(
            sink=sink, obs_window=obs_window, local_window=local_window
        ),
        "budget_formula": "16*ceil((sink+local+ceil(p*max(0,L-sink-local)))/16)",
        "snapkv_paper": "SnapKV, arXiv:2404.14469",
    }
    if budget_p is not None:
        meta["snapkv_budget_p"] = float(budget_p)
        meta["snapkv_budget_mode"] = "grid"
    if use_denom_channel:
        meta["denom_channel"] = True
        meta["bg_estimator"] = "snapkv_evicted_page_mean_key"
        meta["bg_value"] = "evicted_page_mean"
        meta["bg_source"] = "host_snapkv_evicted_clusters"
        meta["snapkv_cluster_size"] = SNAPKV_CLUSTER_SIZE
        meta["note"] = (
            "Background logits are q·mean(K of evicted tokens in each page). "
            "Selection is unchanged SnapKV. No PQ index."
        )
    return meta
