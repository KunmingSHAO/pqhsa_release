"""RetrievalAttention (arXiv:2409.10516) harness reproduction.

Paper §3.1–§3.3 (not the official CPU-ANN / RoarGraph stack):
  * Static GPU set W: initial sink + local window (paper default 128 + 512 = 640).
  * Dynamic set: remaining KV, retrieved by an OOD-aware index.
  * Index: prefill queries guide construction — exact KNN from sampled queries to
    dynamic keys; decode query → nearest prefill-query anchors → mapped keys,
    plus a torch IVF expansion on keys. Partial attentions on W and Ω are merged
    with FlashAttention-style log-sum-exp (§B.1).
  * Combination is a *token subset* + two-way merge, not a calibrated background
    channel in one softmax (that is PQ-HSA).

This is a quality-only reproduction: GPU-resident torch ANN, no Faiss/RoarGraph,
no CPU offload. Declared in report_metadata() / json.
"""
from __future__ import annotations

import json
import math
from types import MethodType
from typing import Any

import torch

import e2e_pq_param_sweep as e2e

# Paper Table 2 / §3.3: 128 initial + 512 local. Retrieved top-k fills the rest
# of the aligned token_budget (800 @ 0.5%, 1456 @ 1% for L=131072).
RA_SINK = 128
RA_LOCAL = 512
RA_MIN_TOPK = 16
RA_N_QUERIES = 512
RA_KNN = 64
RA_NLIST = 256
RA_NPROBE = 8
RA_NPROBE_Q = 24
RA_IVF_CAP = 2048
RA_CAND_CAP = 4096
RA_KMEANS_ITERS = 1

_ATTRS = (
    "_ra_original_forward",
    "_ra_token_budget",
    "_ra_sink",
    "_ra_local",
    "_ra_n_queries",
    "_ra_knn",
    "_ra_nlist",
    "_ra_nprobe",
    "_ra_nprobe_q",
    "_ra_prompt_len",
    "_ra_layer_idx",
    "_ra_index_ready",
    "_ra_q_chunks",
    "_ra_index",
    "_ra_resolved",
    "_ra_keep_pos",
)


def aligned_token_budget(p: float, length: int, sink: int = 4, local: int = 128) -> int:
    """Aligned-budget grid formula (protocol sink/local, not paper 128+512)."""
    inner = sink + local + math.ceil(p * max(0, length - sink - local))
    return 16 * math.ceil(inner / 16)


def resolve_static_and_topk(
    token_budget: int,
    sink: int = RA_SINK,
    local: int = RA_LOCAL,
    min_topk: int = RA_MIN_TOPK,
) -> tuple[int, int, int]:
    """Split token_budget into paper-style static W + retrieved top-k."""
    budget = int(token_budget)
    sink = max(0, min(int(sink), budget))
    remain = budget - sink
    if remain <= int(min_topk):
        return sink, remain, 0
    local = int(local)
    if sink + local + int(min_topk) > budget:
        local = max(0, budget - sink - int(min_topk))
    topk = budget - sink - local
    return sink, local, max(0, topk)


def min_feasible_budget(min_topk: int = RA_MIN_TOPK) -> int:
    return int(min_topk)


def _partial_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (output, row_max, lse=sum(exp(logit-max))) over the last dim."""
    logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
    if valid is not None:
        logits = logits.masked_fill(~valid.unsqueeze(2), torch.finfo(logits.dtype).min)
    row_max = logits.max(dim=-1, keepdim=True).values
    if valid is not None and not bool(valid.any()):
        out = torch.zeros_like(query)
        lse = torch.zeros_like(row_max)
        return out, row_max, lse
    exp = torch.exp(logits - row_max)
    if valid is not None:
        exp = exp.masked_fill(~valid.unsqueeze(2), 0.0)
    lse = exp.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(exp.dtype).tiny)
    weights = (exp / lse).to(value.dtype)
    out = torch.matmul(weights, value)
    return out.to(query.dtype), row_max, lse


def _merge_partial(
    o1: torch.Tensor,
    m1: torch.Tensor,
    lse1: torch.Tensor,
    o2: torch.Tensor,
    m2: torch.Tensor,
    lse2: torch.Tensor,
) -> torch.Tensor:
    """FlashAttention-style merge of two disjoint partial attentions (paper §B.1)."""
    m = torch.maximum(m1, m2)
    a1 = torch.exp(m1.float() - m.float()) * lse1.float()
    a2 = torch.exp(m2.float() - m.float()) * lse2.float()
    denom = (a1 + a2).clamp_min(torch.finfo(a1.dtype).tiny)
    merged = (a1 / denom) * o1.float() + (a2 / denom) * o2.float()
    return merged.to(o1.dtype)


def _static_indices(seq_len: int, sink: int, local: int, device: torch.device) -> torch.Tensor:
    sink = max(0, min(int(sink), seq_len))
    local_start = max(sink, seq_len - max(0, int(local)))
    sink_idx = torch.arange(0, sink, device=device)
    local_idx = torch.arange(local_start, seq_len, device=device)
    return torch.cat([sink_idx, local_idx], dim=0)


def _sample_keep_local(start: int, end: int, keep_pos: torch.Tensor) -> torch.Tensor:
    mask = (keep_pos >= start) & (keep_pos < end)
    return keep_pos[mask]


def _planned_anchor_positions(prompt_len: int, n_queries: int, local: int, device: torch.device) -> torch.Tensor:
    prompt_len = max(1, int(prompt_len))
    n_q = max(1, min(int(n_queries), prompt_len))
    stride = torch.linspace(0, prompt_len - 1, n_q, device=device).round().long()
    loc_n = max(0, min(int(local), prompt_len, 256))
    local_pos = torch.arange(prompt_len - loc_n, prompt_len, device=device, dtype=torch.long)
    return torch.unique(torch.cat([stride, local_pos], dim=0), sorted=True)


def _capture_prefill_queries(
    attn: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    past_key_values: Any,
) -> None:
    q_len = int(hidden_states.shape[1])
    if q_len <= 1:
        return
    hidden_shape = (*hidden_states.shape[:-1], -1, attn.head_dim)
    query = attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query, _ = e2e.llama_apply_rotary_pos_emb(query, query, cos, sin)

    layer_idx = int(getattr(attn, "layer_idx", getattr(attn, "_ra_layer_idx", 0)))
    if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
        seq_after = int(past_key_values.get_seq_length(layer_idx))
    elif past_key_values is not None:
        seq_after = int(past_key_values.layers[layer_idx].keys.shape[-2])
    else:
        seq_after = q_len
    start = max(0, seq_after - q_len)
    keep_all = attn._ra_keep_pos.to(device=query.device, dtype=torch.long)
    keep = _sample_keep_local(start, start + q_len, keep_all)
    if keep.numel() == 0:
        return
    local = keep - start
    q_keep = query.index_select(2, local).detach()
    chunks: list[tuple[torch.Tensor, torch.Tensor]] = getattr(attn, "_ra_q_chunks")
    chunks.append((keep.detach(), q_keep))


def _concat_captured_queries(attn: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    chunks: list[tuple[torch.Tensor, torch.Tensor]] = getattr(attn, "_ra_q_chunks", [])
    if not chunks:
        raise RuntimeError("retrieval_attention captured no prefill queries")
    pos = torch.cat([c[0] for c in chunks], dim=0)
    qs = torch.cat([c[1] for c in chunks], dim=2)
    # Dedup positions (stride ∩ last-local).
    uniq_pos, inv = torch.unique(pos, sorted=True, return_inverse=True)
    if uniq_pos.numel() == pos.numel():
        return pos, qs
    out = torch.zeros(
        qs.shape[0], qs.shape[1], uniq_pos.numel(), qs.shape[-1],
        device=qs.device, dtype=qs.dtype,
    )
    out.index_copy_(2, inv, qs)
    return uniq_pos, out


def _kmeans_ivf(
    keys: torch.Tensor,
    nlist: int,
    iters: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """keys: [B, H, N, D] → centroids [B, H, nlist, D], assign [B, H, N]."""
    bsz, n_h, n_vec, dim = keys.shape
    nlist = max(1, min(int(nlist), n_vec))
    pick = torch.randperm(n_vec, generator=generator, device=keys.device)[:nlist]
    centroids = keys.index_select(2, pick).float().contiguous()
    keys_f = keys.float()
    assign = torch.zeros(bsz, n_h, n_vec, dtype=torch.long, device=keys.device)
    remain = int(iters)
    for _ in range(max(0, remain) + 1):
        scores = torch.matmul(keys_f, centroids.transpose(-1, -2))
        assign = scores.argmax(dim=-1)
        if remain <= 0:
            break
        flat = bsz * n_h
        kf = keys_f.reshape(flat, n_vec, dim)
        af = assign.reshape(flat, n_vec)
        sums = torch.zeros(flat, nlist, dim, device=keys.device, dtype=keys_f.dtype)
        sums.scatter_add_(1, af.unsqueeze(-1).expand_as(kf), kf)
        counts = torch.zeros(flat, nlist, device=keys.device, dtype=keys_f.dtype)
        counts.scatter_add_(1, af, torch.ones_like(af, dtype=keys_f.dtype))
        updated = sums / counts.clamp_min(1.0).unsqueeze(-1)
        empty = (counts < 1.0).unsqueeze(-1)
        centroids = torch.where(
            empty.view(bsz, n_h, nlist, 1),
            centroids,
            updated.view(bsz, n_h, nlist, dim),
        )
        remain -= 1
    return centroids.to(keys.dtype), assign


def _csr_from_assign(
    assign: torch.Tensor,
    dyn_pos: torch.Tensor,
    nlist: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """assign [B,H,N], dyn_pos [N] → list_idx [B,H,N], ptr [B,H,nlist+1]."""
    bsz, n_h, n_vec = assign.shape
    order = assign.argsort(dim=-1)
    list_idx = dyn_pos.to(assign.device).expand_as(assign).gather(-1, order)
    counts = torch.zeros(bsz, n_h, nlist, dtype=torch.long, device=assign.device)
    counts.scatter_add_(2, assign, torch.ones_like(assign))
    ptr = torch.zeros(bsz, n_h, nlist + 1, dtype=torch.long, device=assign.device)
    ptr[:, :, 1:] = counts.cumsum(dim=-1)
    return list_idx, ptr


def _exact_q2k(
    queries: torch.Tensor,
    keys_dyn: torch.Tensor,
    dyn_pos: torch.Tensor,
    knn: int,
    groups: int,
) -> torch.Tensor:
    """Per Q-head exact top-m keys among dynamic KV. queries [B,Hq,Nq,D], keys [B,Hkv,N,D]."""
    bsz, n_q, n_anchor, dim = queries.shape
    n_kv = keys_dyn.shape[1]
    n_dyn = keys_dyn.shape[2]
    knn = max(1, min(int(knn), n_dyn))
    q2k = torch.empty(bsz, n_q, n_anchor, knn, dtype=torch.long, device=queries.device)
    dyn = dyn_pos.to(queries.device)
    for kv in range(n_kv):
        h0 = kv * max(groups, 1)
        h1 = min(n_q, h0 + max(groups, 1))
        if h0 >= n_q:
            break
        qg = queries[:, h0:h1].float()
        kg = keys_dyn[:, kv]
        scores = torch.matmul(qg, kg.float().transpose(-1, -2))
        _, topi = scores.topk(knn, dim=-1)
        q2k[:, h0:h1] = dyn[topi]
        del scores
    return q2k


def build_one_index(
    attn: torch.nn.Module,
    cache: Any,
) -> None:
    layer_idx = int(getattr(attn, "layer_idx", getattr(attn, "_ra_layer_idx", 0)))
    layer = cache.layers[layer_idx]
    if layer.keys is None or layer.values is None:
        raise RuntimeError(f"retrieval_attention missing cache at layer {layer_idx}")
    keys = layer.keys
    bsz, n_kv, seq_len, dim = keys.shape
    sink, local, topk = attn._ra_resolved
    device = keys.device
    dyn_start = min(int(sink), seq_len)
    dyn_end = max(dyn_start, seq_len - int(local))
    n_dyn = dyn_end - dyn_start
    groups = int(attn.num_key_value_groups)

    if n_dyn <= 0 or topk <= 0:
        attn._ra_index = None
        attn._ra_q_chunks = []
        return

    dyn_pos = torch.arange(dyn_start, dyn_end, device=device, dtype=torch.long)
    keys_dyn = keys.index_select(2, dyn_pos)
    _, queries = _concat_captured_queries(attn)

    nlist = max(8, min(int(attn._ra_nlist), max(8, n_dyn // 64), n_dyn))
    gen = torch.Generator(device=device)
    gen.manual_seed(0xA11E + layer_idx)
    centroids, assign = _kmeans_ivf(keys_dyn, nlist, RA_KMEANS_ITERS, gen)
    list_idx, ptr = _csr_from_assign(assign, dyn_pos, nlist)
    q2k = _exact_q2k(queries, keys_dyn, dyn_pos, int(attn._ra_knn), groups)

    attn._ra_index = {
        "q_anchor": queries.contiguous(),
        "q2k": q2k.contiguous(),
        "centroids": centroids.contiguous(),
        "list_idx": list_idx.contiguous(),
        "ptr": ptr.contiguous(),
        "nlist": int(nlist),
        "n_dyn": int(n_dyn),
        "groups": groups,
        "n_kv": int(n_kv),
    }
    attn._ra_q_chunks = []
    if layer_idx == 0:
        print(
            json.dumps(
                {
                    "event": "ra_index_built",
                    "layer": layer_idx,
                    "seq_len": int(seq_len),
                    "n_dyn": int(n_dyn),
                    "n_anchors": int(queries.shape[2]),
                    "nlist": int(nlist),
                    "knn": int(q2k.shape[-1]),
                    "topk": int(topk),
                    "sink": int(sink),
                    "local": int(local),
                },
                sort_keys=True,
            ),
            flush=True,
        )


def build_indexes(patched: list[torch.nn.Module], cache: Any) -> None:
    for attn in patched:
        build_one_index(attn, cache)
        attn._ra_index_ready = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _gather_ivf(
    lists: torch.Tensor,
    list_idx: torch.Tensor,
    ptr: torch.Tensor,
    cap: int,
) -> torch.Tensor:
    """lists [B,H,nprobe], list_idx/ptr on KV heads already expanded to H."""
    bsz, n_h, nprobe = lists.shape
    starts = torch.gather(ptr, 2, lists)
    ends = torch.gather(ptr, 2, lists + 1)
    lens = (ends - starts).clamp_min(0)
    max_len = int(lens.max().item()) if lens.numel() else 1
    max_len = max(1, min(max_len, max(1, cap // max(nprobe, 1))))
    off = torch.arange(max_len, device=lists.device)
    pos = starts.unsqueeze(-1) + off.view(1, 1, 1, max_len)
    ok = off.view(1, 1, 1, max_len) < lens.unsqueeze(-1)
    n_full = list_idx.shape[-1]
    pos = pos.clamp(0, max(0, n_full - 1))
    gathered = torch.gather(list_idx, 2, pos.reshape(bsz, n_h, -1))
    gathered = gathered.masked_fill(~ok.reshape(bsz, n_h, -1), -1)
    return gathered


def _search_candidates(
    query: torch.Tensor,
    index: dict[str, Any],
    seq_len: int,
    sink: int,
    local: int,
    nprobe_q: int,
    nprobe: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cand_idx [B,H,C], valid [B,H,C]) in the dynamic region."""
    bsz, n_h, _, dim = query.shape
    q_anchor = index["q_anchor"]
    q2k = index["q2k"]
    groups = int(index["groups"])
    n_kv = int(index["n_kv"])
    kv_ids = torch.arange(n_h, device=query.device) // groups
    kv_ids = kv_ids.clamp(max=n_kv - 1)

    qf = query.float()
    nq_keep = max(1, min(int(nprobe_q), q_anchor.shape[2]))
    sim_q = torch.matmul(qf, q_anchor.float().transpose(-1, -2))
    _, nq = sim_q.topk(nq_keep, dim=-1)
    nq = nq.squeeze(2)
    mapped = torch.gather(
        q2k, 2, nq.unsqueeze(-1).expand(bsz, n_h, nq_keep, q2k.shape[-1]),
    ).reshape(bsz, n_h, -1)

    centroids = index["centroids"][:, kv_ids]
    nlist = int(index["nlist"])
    npk = max(1, min(int(nprobe), nlist))
    sim_c = torch.matmul(qf, centroids.float().transpose(-1, -2))
    _, lists = sim_c.topk(npk, dim=-1)
    lists = lists.squeeze(2)
    list_idx = index["list_idx"][:, kv_ids]
    ptr = index["ptr"][:, kv_ids]
    ivf = _gather_ivf(lists, list_idx, ptr, RA_IVF_CAP)

    cands = torch.cat([mapped, ivf], dim=-1)
    local_start = max(int(sink), seq_len - int(local))
    valid = (cands >= int(sink)) & (cands < local_start) & (cands < seq_len) & (cands >= 0)
    cands_sorted, order = cands.sort(dim=-1)
    valid = torch.gather(valid, 2, order)
    dup = torch.zeros_like(valid)
    dup[:, :, 1:] = cands_sorted[:, :, 1:] == cands_sorted[:, :, :-1]
    valid = valid & ~dup & (cands_sorted >= 0)
    if cands_sorted.shape[-1] > RA_CAND_CAP:
        cands_sorted = cands_sorted[:, :, :RA_CAND_CAP]
        valid = valid[:, :, :RA_CAND_CAP]
    return cands_sorted, valid


def _retrieval_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attn: torch.nn.Module,
) -> torch.Tensor:
    bsz, n_heads, q_len, head_dim = query_states.shape
    del q_len
    seq_len = int(key_states.shape[-2])
    scale = 1.0 / math.sqrt(head_dim)
    sink, local, topk = attn._ra_resolved
    device = query_states.device
    groups = int(attn.num_key_value_groups)
    k_rep = e2e.llama_repeat_kv(key_states, groups)
    v_rep = e2e.llama_repeat_kv(value_states, groups)

    static_idx = _static_indices(seq_len, sink, local, device)
    k_w = k_rep.index_select(2, static_idx)
    v_w = v_rep.index_select(2, static_idx)
    o_w, m_w, lse_w = _partial_attn(query_states, k_w, v_w, None, scale)

    index = getattr(attn, "_ra_index", None)
    if index is None or topk <= 0:
        return o_w

    cands, valid = _search_candidates(
        query_states,
        index,
        seq_len,
        sink,
        local,
        int(attn._ra_nprobe_q),
        int(attn._ra_nprobe),
    )
    n_c = cands.shape[-1]
    safe = cands.clamp(0, max(0, seq_len - 1))
    gather = safe.unsqueeze(-1).expand(bsz, n_heads, n_c, head_dim)
    k_c = torch.gather(k_rep, 2, gather)
    scores = torch.matmul(query_states.float(), k_c.float().transpose(-1, -2)) * scale
    scores = scores.masked_fill(~valid.unsqueeze(2), torch.finfo(scores.dtype).min)
    k_use = min(int(topk), n_c)
    _, topi = scores.topk(k_use, dim=-1)
    retr = torch.gather(cands, 2, topi.squeeze(2))
    retr_valid = torch.gather(valid, 2, topi.squeeze(2))
    retr_safe = retr.clamp(0, max(0, seq_len - 1))
    g2 = retr_safe.unsqueeze(-1).expand(bsz, n_heads, k_use, head_dim)
    k_o = torch.gather(k_rep, 2, g2)
    v_o = torch.gather(v_rep, 2, g2)
    o_o, m_o, lse_o = _partial_attn(query_states, k_o, v_o, retr_valid, scale)
    if not bool(retr_valid.any()):
        return o_w
    return _merge_partial(o_w, m_w, lse_w, o_o, m_o, lse_o)


def _llama_retrieval_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_values: Any = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    input_shape = hidden_states.shape[:-1]
    q_len = int(input_shape[1]) if len(input_shape) == 2 else 0
    ready = bool(getattr(self, "_ra_index_ready", False))
    if q_len != 1 or past_key_values is None or not ready:
        out = self._ra_original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        if q_len > 1 and past_key_values is not None and not ready:
            _capture_prefill_queries(self, hidden_states, position_embeddings, past_key_values)
        return out

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
    context = _retrieval_attention(query_states, key_states, value_states, self)
    attn_output = context.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def install_retrieval_attention_patch(
    model: torch.nn.Module,
    *,
    token_budget: int,
    prompt_len: int,
    sink: int = RA_SINK,
    local_window: int = RA_LOCAL,
    n_queries: int = RA_N_QUERIES,
    knn: int = RA_KNN,
    nlist: int = RA_NLIST,
    nprobe: int = RA_NPROBE,
    nprobe_q: int = RA_NPROBE_Q,
) -> list[torch.nn.Module]:
    model_type = getattr(model.config, "model_type", "")
    if model_type not in {"llama", "qwen2", "mistral"}:
        raise ValueError(f"unsupported model_type for retrieval_attention patch: {model_type}")
    if token_budget <= 0:
        raise ValueError(f"retrieval_attention token_budget must be positive, got {token_budget}")
    resolved = resolve_static_and_topk(token_budget, sink, local_window)
    keep_device = next(model.parameters()).device
    keep_pos = _planned_anchor_positions(prompt_len, n_queries, resolved[1], device=keep_device)

    patched: list[torch.nn.Module] = []
    for layer_idx, decoder_layer in enumerate(model.model.layers):
        attn = decoder_layer.self_attn
        if hasattr(attn, "_ra_original_forward"):
            raise RuntimeError("retrieval_attention patch already installed")
        attn._ra_original_forward = attn.forward
        attn._ra_token_budget = int(token_budget)
        attn._ra_sink, attn._ra_local, _ = resolved
        attn._ra_resolved = resolved
        attn._ra_n_queries = int(n_queries)
        attn._ra_knn = int(knn)
        attn._ra_nlist = int(nlist)
        attn._ra_nprobe = int(nprobe)
        attn._ra_nprobe_q = int(nprobe_q)
        attn._ra_prompt_len = int(prompt_len)
        attn._ra_layer_idx = layer_idx
        attn._ra_index_ready = False
        attn._ra_q_chunks = []
        attn._ra_index = None
        attn._ra_keep_pos = keep_pos
        attn.forward = MethodType(_llama_retrieval_attention_forward, attn)
        patched.append(attn)
    return patched


def restore_retrieval_attention_patch(patched: list[torch.nn.Module]) -> None:
    for attn in patched:
        original = getattr(attn, "_ra_original_forward", None)
        if original is not None:
            attn.forward = original
        for name in _ATTRS:
            if hasattr(attn, name):
                delattr(attn, name)


def report_metadata(
    *,
    token_budget: int,
    sink: int,
    local_window: int,
    budget_p: float | None,
    n_queries: int = RA_N_QUERIES,
    knn: int = RA_KNN,
    nlist: int = RA_NLIST,
    nprobe: int = RA_NPROBE,
    nprobe_q: int = RA_NPROBE_Q,
) -> dict[str, Any]:
    rs, rl, rk = resolve_static_and_topk(token_budget, sink, local_window)
    meta: dict[str, Any] = {
        "token_budget": int(token_budget),
        "ra_sink": int(rs),
        "ra_local": int(rl),
        "ra_retrieve_topk": int(rk),
        "ra_sink_requested": int(sink),
        "ra_local_requested": int(local_window),
        "ra_n_queries": int(n_queries),
        "ra_knn": int(knn),
        "ra_nlist": int(nlist),
        "ra_nprobe": int(nprobe),
        "ra_nprobe_q": int(nprobe_q),
        "ra_paper": "RetrievalAttention, arXiv:2409.10516",
        "implementation": "harness reproduction, not official",
        "ra_merge": "flashattention_logsumexp_two_partial",
        "ra_first_two_layers_dense": False,
        "budget_formula": "16*ceil((4+128+ceil(p*max(0,L-132)))/16)",
        "simplifications": [
            "no official RetrievalAttention / RoarGraph / Faiss stack (offline, not installed)",
            "torch IVF + sampled-query KNN mapping instead of RoarGraph projection (query anchors kept)",
            "prefill queries subsampled (stride + last-local), not all prefill Q",
            "IVF centroids: random init + 1 k-means iter, not Faiss IVF",
            "KV and index stay on GPU (quality-only; no CPU offload / PCIe path)",
            "index is built once after prefill and not updated during short decode",
            "retrieved set is a token subset; no PQ background logits in the softmax denominator",
        ],
    }
    if budget_p is not None:
        meta["ra_budget_p"] = float(budget_p)
        meta["ra_budget_mode"] = "grid"
    return meta
