"""STS-harness (arXiv 2605.15508v3, "Efficient Sparse Attention with Speculative
Token Sparsity") harness reproduction.

Paper recipe (no public code exists; described only): a same-family smaller
"draft" model (Llama-3.2-1B-Instruct) runs alongside the target
(Llama-3.1-8B-Instruct). At each decode step the draft's per-head attention
top-k over the KV cache selects which target tokens get exact attention. An
offline head map (Jaccard overlap of top-k index sets on a short calibration
prefix) assigns each target head to one draft head; because the draft is
shallower than the target, one draft layer's map would in principle be reused
by a proportional band of target layers. This is a *pure selector* baseline:
tokens not selected are dropped, there is no background/compensation term
(contrast PQ-HSA, which keeps a PQ-score background in the same softmax).

Like `baselines_retrieval_attention.py`, this is a quality-only harness
reproduction inside this repo's HF eval loop, not the paper's own system.

Two draft modes (see report_metadata()): the real cross-model draft
(``--sts-draft-model``, Llama-3.2-1B-Instruct; see generate_with_real_draft)
and a **proxy draft** fallback for environments without the 1B checkpoint:
a single shallow layer of the *target* model itself
(`STS_DRAFT_LAYER`, default index 2, 0-indexed) stands in for the 1B draft.
That layer is always computed densely (it must scan the whole cache to
produce a top-k, exactly as a real draft would); its per-head top-k indices,
remapped through the offline Jaccard head map, gate every later target
layer's attention to an exact-only subset (no background — same truncation
family as Quest/SnapKV/sink_local, just with a learned/mapped selector
instead of a heuristic one). Consequences of the proxy, stated up front:
  * Layers 0..STS_DRAFT_LAYER (inclusive) cannot be sparsified — the draft
    signal for step t only exists *after* layer STS_DRAFT_LAYER has run
    within that same forward pass, so earlier layers stay dense (this
    mirrors Quest's own "first two layers dense" convention, but for a
    different reason). A real separate draft model completes its entire
    forward before the target starts, so all 32 target layers would be
    sparse; here only layers (STS_DRAFT_LAYER+1)..31 are.
  * Because the "draft" is a layer of the target network, not a smaller
    network, its attention op is not cheaper than the target's own dense
    attention. The draft-overhead number recorded here is a structural
    placeholder, not a faithful stand-in for real 1B-vs-8B draft cost —
    flagged n.m. for the speed axis. Quality is the axis the proxy mode is
    meant for.
  * Head mapping is still computed for real via Jaccard overlap (not
    identity-assumed), even though draft and target share a head-count
    convention (32 query / 8 KV heads) because they are literally the same
    network; head semantics still drift across layers, so the mapping is a
    non-trivial permutation, not a no-op — see reported head-map stats.
"""
from __future__ import annotations

import math
import time
from types import MethodType
from typing import Any

import torch

import e2e_pq_param_sweep as e2e

STS_DRAFT_LAYER = 2       # proxy draft layer index (0-indexed) inside the target net
STS_CALIB_TOKENS = 4096   # calibration window (paper: "a short text, e.g. 4K")
STS_CALIB_TOPK = 64       # top-k size used only for the offline head-map Jaccard step

_ATTRS = (
    "_sts_original_forward",
    "_sts_layer_idx",
    "_sts_role",
    "_sts_token_budget",
    "_sts_calib_len",
    "_sts_calib_topk",
    "_sts_calib_chunks",
    "_sts_head_map",
    "_sts_index_ready",
    "_sts_state",
)


def aligned_token_budget(p: float, length: int, sink: int = 4, local: int = 128) -> int:
    """Aligned-budget grid formula (protocol sink/local; STS itself has no sink/local
    structure — the resulting count is used directly as the draft top-k size)."""
    inner = sink + local + math.ceil(p * max(0, length - sink - local))
    return 16 * math.ceil(inner / 16)


def min_feasible_budget(min_topk: int = 16) -> int:
    return int(min_topk)


def _dense_attn_with_topk(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact dense attention (this is the "draft" step: must see the full cache
    to produce a top-k). Returns (output, topk_idx[B,H,k])."""
    logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale  # [B,H,1,N]
    weights = torch.softmax(logits, dim=-1)
    out = torch.matmul(weights.to(value.dtype), value)
    n = key.shape[-2]
    kk = max(1, min(int(k), n))
    _, topi = logits.squeeze(2).topk(kk, dim=-1)  # [B,H,kk]
    return out.to(query.dtype), topi


def _exact_topk_attn(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, idx: torch.Tensor, scale: float
) -> torch.Tensor:
    """Target-side exact attention restricted to the draft-selected token subset.
    No background term (pure selector, matches the paper)."""
    bsz, n_h, _, hd = query.shape
    k = idx.shape[-1]
    gather = idx.unsqueeze(-1).expand(bsz, n_h, k, hd)
    k_sel = torch.gather(key, 2, gather)
    v_sel = torch.gather(value, 2, gather)
    logits = torch.matmul(query.float(), k_sel.float().transpose(-1, -2)) * scale
    weights = torch.softmax(logits, dim=-1).to(value.dtype)
    return torch.matmul(weights, v_sel).to(query.dtype)


def _capture_calib_query(
    attn: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    past_key_values: Any,
) -> None:
    """During prefill only: stash this chunk's post-RoPE query vectors for the
    positions that fall inside the calibration window [0, calib_len)."""
    q_len = int(hidden_states.shape[1])
    if q_len <= 1:
        return
    layer_idx = int(attn.layer_idx)
    if hasattr(past_key_values, "get_seq_length"):
        seq_after = int(past_key_values.get_seq_length(layer_idx))
    else:
        seq_after = int(past_key_values.layers[layer_idx].keys.shape[-2])
    start = max(0, seq_after - q_len)
    calib_len = int(attn._sts_calib_len)
    if start >= calib_len:
        return
    n_take = min(calib_len, start + q_len) - start
    if n_take <= 0:
        return
    hidden_shape = (*hidden_states.shape[:-1], -1, attn.head_dim)
    q_full = attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    q_full, _ = e2e.llama_apply_rotary_pos_emb(q_full, q_full, cos, sin)
    q_slice = q_full[:, :, 0:n_take, :].detach()
    attn._sts_calib_chunks.append(q_slice)


def _calib_topk_sets(attn: torch.nn.Module, cache: Any) -> torch.Tensor:
    """Jaccard signature per head: top-k(calib_topk) key positions attended by
    the *last* calibration-window query row. Returns [n_heads, calib_topk]."""
    chunks = attn._sts_calib_chunks
    if not chunks:
        raise RuntimeError(f"sts: no calibration queries captured at layer {attn._sts_layer_idx}")
    q_calib = torch.cat(chunks, dim=2)
    calib_len = int(attn._sts_calib_len)
    q_calib = q_calib[:, :, :calib_len, :]
    last_q = q_calib[:, :, -1:, :]  # [B,H,1,D]
    layer = cache.layers[int(attn.layer_idx)]
    k_calib = layer.keys[:, :, :calib_len, :]
    groups = int(attn.num_key_value_groups)
    k_calib_rep = e2e.llama_repeat_kv(k_calib, groups)
    scale = 1.0 / math.sqrt(attn.head_dim)
    logits = torch.matmul(last_q.float(), k_calib_rep.float().transpose(-1, -2)) * scale
    logits = logits.squeeze(2).squeeze(0)  # [n_heads, calib_len] (bsz==1)
    kk = max(1, min(int(attn._sts_calib_topk), logits.shape[-1]))
    _, topi = logits.topk(kk, dim=-1)  # [n_heads, kk]
    return topi


def _jaccard_matrix(target_idx: torch.Tensor, draft_idx: torch.Tensor, calib_len: int) -> torch.Tensor:
    """Pairwise Jaccard(target_head, draft_head) via boolean-membership matmul."""
    n_t, k_t = target_idx.shape
    n_d, k_d = draft_idx.shape
    device = target_idx.device
    mem_t = torch.zeros(n_t, calib_len, dtype=torch.float32, device=device)
    mem_t.scatter_(1, target_idx, 1.0)
    mem_d = torch.zeros(n_d, calib_len, dtype=torch.float32, device=device)
    mem_d.scatter_(1, draft_idx, 1.0)
    inter = mem_t @ mem_d.t()
    union = (k_t + k_d) - inter
    return inter / union.clamp_min(1.0)


def build_head_maps(patched: list[torch.nn.Module], cache: Any) -> dict[str, Any]:
    """Offline step 1: compute the draft head signature once, then Jaccard-map
    every target layer's heads onto it. Marks all patched layers ready."""
    draft_attn = next(a for a in patched if a._sts_role == "draft")
    calib_len = int(draft_attn._sts_calib_len)
    draft_sets = _calib_topk_sets(draft_attn, cache)  # [n_heads, kk]
    n_heads = draft_sets.shape[0]
    per_layer: dict[str, Any] = {}
    for attn in patched:
        if attn._sts_role == "target":
            tgt_sets = _calib_topk_sets(attn, cache)
            jacc = _jaccard_matrix(tgt_sets, draft_sets, calib_len)
            head_map = jacc.argmax(dim=1)
            best = jacc.gather(1, head_map.view(-1, 1)).squeeze(1)
            identity = torch.arange(n_heads, device=head_map.device)
            attn._sts_head_map = head_map.detach()
            per_layer[str(int(attn._sts_layer_idx))] = {
                "mean_best_jaccard": float(best.mean().item()),
                "min_best_jaccard": float(best.min().item()),
                "frac_identity_map": float((head_map == identity).float().mean().item()),
            }
        attn._sts_index_ready = True
        attn._sts_calib_chunks = []
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    all_best = [v["mean_best_jaccard"] for v in per_layer.values()]
    return {
        "draft_layer": int(draft_attn._sts_layer_idx),
        "calib_len": calib_len,
        "calib_topk": int(draft_attn._sts_calib_topk),
        "n_target_layers_mapped": len(per_layer),
        "mean_best_jaccard_overall": float(sum(all_best) / len(all_best)) if all_best else None,
        "per_layer": per_layer,
    }


def _sts_llama_forward(
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
    ready = bool(getattr(self, "_sts_index_ready", False))
    if q_len != 1 or past_key_values is None or not ready:
        out = self._sts_original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        if q_len > 1 and past_key_values is not None and not ready:
            _capture_calib_query(self, hidden_states, position_embeddings, past_key_values)
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
    groups = int(self.num_key_value_groups)
    k_rep = e2e.llama_repeat_kv(key_states, groups)
    v_rep = e2e.llama_repeat_kv(value_states, groups)
    scale = 1.0 / math.sqrt(self.head_dim)

    if self._sts_role == "draft":
        out, topk_idx = _dense_attn_with_topk(query_states, k_rep, v_rep, scale, int(self._sts_token_budget))
        self._sts_state["draft_topk_idx"] = topk_idx
    else:
        draft_idx = self._sts_state.get("draft_topk_idx")
        if draft_idx is None:
            # Should not happen: the draft layer runs earlier in the same
            # sequential forward pass. Fail safe to dense rather than crash a
            # long-running evaluation.
            out, _ = _dense_attn_with_topk(query_states, k_rep, v_rep, scale, k_rep.shape[2])
        else:
            head_map = self._sts_head_map
            bsz, n_h, k = draft_idx.shape
            idx_map = head_map.view(1, n_h, 1).expand(bsz, n_h, k).to(draft_idx.device)
            sel_idx = torch.gather(draft_idx, 1, idx_map)
            sel_idx = sel_idx.clamp(0, k_rep.shape[2] - 1)
            out = _exact_topk_attn(query_states, k_rep, v_rep, sel_idx, scale)

    attn_output = out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def install_sts_patch(
    model: torch.nn.Module,
    *,
    token_budget: int,
    prompt_len: int,
    draft_layer: int = STS_DRAFT_LAYER,
    calib_tokens: int = STS_CALIB_TOKENS,
    calib_topk: int = STS_CALIB_TOPK,
) -> list[torch.nn.Module]:
    model_type = getattr(model.config, "model_type", "")
    if model_type not in {"llama", "qwen2", "mistral"}:
        raise ValueError(f"unsupported model_type for sts patch: {model_type}")
    if token_budget <= 0:
        raise ValueError(f"sts token_budget must be positive, got {token_budget}")
    layers = list(model.model.layers)
    n_layers = len(layers)
    draft_layer = max(0, min(int(draft_layer), n_layers - 1))
    calib_len = max(1, min(int(calib_tokens), int(prompt_len)))
    shared_state: dict[str, Any] = {}

    patched: list[torch.nn.Module] = []
    for layer_idx in range(draft_layer, n_layers):
        attn = layers[layer_idx].self_attn
        if hasattr(attn, "_sts_original_forward"):
            raise RuntimeError("sts patch already installed")
        attn._sts_original_forward = attn.forward
        attn._sts_layer_idx = layer_idx
        attn._sts_role = "draft" if layer_idx == draft_layer else "target"
        attn._sts_token_budget = int(token_budget)
        attn._sts_calib_len = calib_len
        attn._sts_calib_topk = int(calib_topk)
        attn._sts_calib_chunks = []
        attn._sts_head_map = None
        attn._sts_index_ready = False
        attn._sts_state = shared_state
        attn.forward = MethodType(_sts_llama_forward, attn)
        patched.append(attn)
    return patched


def restore_sts_patch(patched: list[torch.nn.Module]) -> None:
    for attn in patched:
        original = getattr(attn, "_sts_original_forward", None)
        if original is not None:
            attn.forward = original
        for name in _ATTRS:
            if hasattr(attn, name):
                delattr(attn, name)


def report_metadata(
    *,
    token_budget: int,
    budget_p: float | None,
    draft_layer: int = STS_DRAFT_LAYER,
    calib_tokens: int = STS_CALIB_TOKENS,
    calib_topk: int = STS_CALIB_TOPK,
    head_map_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "token_budget": int(token_budget),
        "sts_draft_layer": int(draft_layer),
        "sts_calib_tokens": int(calib_tokens),
        "sts_calib_topk": int(calib_topk),
        "sts_paper": "arXiv:2605.15508v3, Efficient Sparse Attention with Speculative Token Sparsity",
        "implementation": "harness reproduction, not official (no public code)",
        "sts_selector": "pure top-k selector, no background term (unlike PQ-HSA)",
        "sts_draft_model_requested": "meta-llama/Llama-3.2-1B-Instruct",
        "sts_draft_model_available": False,
        "sts_draft_proxy": "single shallow layer of the target model itself (gated HF download, no token in this env)",
        "budget_formula": "16*ceil((4+128+ceil(p*max(0,L-132)))/16)",
        "simplifications": [
            "no real Llama-3.2-1B-Instruct draft: HF repo is gated, 401 without a token in this environment",
            "proxy draft = target's own layer `sts_draft_layer` (dense every decode step, since it must scan the full cache to produce a top-k)",
            "consequence: target layers 0..sts_draft_layer stay dense (causal — the proxy signal for step t only exists after that layer runs within the same forward pass); a real separate draft model would let all target layers be sparse",
            "head map computed once after prefill via Jaccard overlap of top-k index sets on the first sts_calib_tokens positions (last calibration query row's attention), not per-decode-step",
            "layer mapping is 1 proxy layer -> all downstream target layers (no proportional multi-layer draft depth, since there is only one proxy layer)",
            "draft overhead is not comparable to a real 1B-vs-8B cost delta: the proxy layer's attention op costs the same as the target's own dense attention (n.m. for the speed axis; quality is the axis judged here)",
            "no speculative decode-step overlap: the proxy's 'draft forward' is simply this step's target forward through the shared early layers, not a separate model call",
        ],
    }
    if budget_p is not None:
        meta["sts_budget_p"] = float(budget_p)
        meta["sts_budget_mode"] = "grid"
    if head_map_stats is not None:
        meta["sts_head_map_stats"] = head_map_stats
    return meta


# ---------------------------------------------------------------------------
# Real cross-model draft (Llama-3.2-1B-Instruct -> Llama-3.1-8B-Instruct)
#
# Replaces the proxy (a single shallow layer of the *target* network
# standing in for the draft, see install_sts_patch/_sts_llama_forward above,
# which is kept unmodified for backward compatibility / as a fallback if the
# draft weights are unavailable in some environment). Here the draft is an
# actual smaller same-family model with its own weights and its own KV cache:
#   - draft = Llama-3.2-1B-Instruct: 16 layers, 32 query / 8 KV heads, head_dim
#     64 (from the model's config.json).
#   - target = Llama-3.1-8B-Instruct: 32 layers, 32 query / 8 KV heads,
#     head_dim 128.
# Query-head COUNT matches exactly (32 == 32) even though head_dim differs --
# that's fine, because only draft top-k *indices* (token positions) cross the
# model boundary, never K/V values or head_dim-shaped tensors. The offline
# head map is still a full Jaccard permutation over the 32x32 pairing (not
# assumed identity), same as the proxy.
#
# Per decode step, the draft's *entire* forward pass (all 16 layers, its own
# KV cache) completes before the target's forward starts -- unlike the proxy,
# there is no same-forward-pass causality constraint, so all 32 target layers
# can be sparsified (not just layers 3..31).
#
# Layer mapping: proportional depth band (paper: "draft:target depth ratio"),
# target layer t -> draft layer floor(t * n_draft / n_target). With 16 draft
# / 32 target layers the ratio is exactly 2, i.e. each draft layer's map
# gates a contiguous band of 2 target layers (see layer_band_map()).
# ---------------------------------------------------------------------------

STS_REAL_DRAFT_MODEL_DEFAULT = "meta-llama/Llama-3.2-1B-Instruct"

_REAL_ATTRS = (
    "_sts_original_forward",
    "_sts_own_layer_idx",
    "_sts_role",
    "_sts_draft_layer_idx",
    "_sts_token_budget",
    "_sts_calib_len",
    "_sts_calib_topk",
    "_sts_calib_chunks",
    "_sts_head_map",
    "_sts_index_ready",
    "_sts_shared",
)


def layer_band_map(n_draft_layers: int, n_target_layers: int) -> list[int]:
    """target layer t -> draft layer floor(t * n_draft_layers / n_target_layers),
    clipped to [0, n_draft_layers-1]. When n_target is an integer multiple of
    n_draft (16 vs 32 here, ratio 2) this is exactly a contiguous 1:many band
    per draft layer, matching the paper's "proportional to depth" layer map."""
    out = []
    for t in range(n_target_layers):
        d = (t * n_draft_layers) // max(1, n_target_layers)
        out.append(max(0, min(d, n_draft_layers - 1)))
    return out


def _sts_real_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_values: Any = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Same shape/contract as _sts_llama_forward, generalized to (a) a
    per-own-layer-idx draft top-k store (every draft layer produces one,
    not just a single proxy layer) and (b) a per-target-layer draft-layer
    lookup via the layer band map, instead of one fixed proxy source."""
    input_shape = hidden_states.shape[:-1]
    q_len = int(input_shape[1]) if len(input_shape) == 2 else 0
    ready = bool(getattr(self, "_sts_index_ready", False))
    if q_len != 1 or past_key_values is None or not ready:
        out = self._sts_original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        if q_len > 1 and past_key_values is not None and not ready:
            _capture_calib_query(self, hidden_states, position_embeddings, past_key_values)
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
    groups = int(self.num_key_value_groups)
    k_rep = e2e.llama_repeat_kv(key_states, groups)
    v_rep = e2e.llama_repeat_kv(value_states, groups)
    scale = 1.0 / math.sqrt(self.head_dim)

    if self._sts_role == "draft":
        out, topk_idx = _dense_attn_with_topk(query_states, k_rep, v_rep, scale, int(self._sts_token_budget))
        self._sts_shared["draft_topk"][int(self._sts_own_layer_idx)] = topk_idx
        self._sts_shared["draft_layer_hits"] = self._sts_shared.get("draft_layer_hits", 0) + 1
    else:
        draft_idx = self._sts_shared["draft_topk"].get(int(self._sts_draft_layer_idx))
        if draft_idx is None:
            # Should not happen: the full draft forward (all layers) runs to
            # completion before the target forward starts each step (see
            # generate_with_real_draft). Fail safe to dense rather than crash
            # a long-running evaluation.
            out, _ = _dense_attn_with_topk(query_states, k_rep, v_rep, scale, k_rep.shape[2])
            self._sts_shared["target_fallback_hits"] = self._sts_shared.get("target_fallback_hits", 0) + 1
        else:
            head_map = self._sts_head_map
            bsz, n_h, k = draft_idx.shape
            idx_map = head_map.view(1, n_h, 1).expand(bsz, n_h, k).to(draft_idx.device)
            sel_idx = torch.gather(draft_idx, 1, idx_map)
            sel_idx = sel_idx.clamp(0, k_rep.shape[2] - 1)
            out = _exact_topk_attn(query_states, k_rep, v_rep, sel_idx, scale)
            self._sts_shared["target_layer_hits"] = self._sts_shared.get("target_layer_hits", 0) + 1

    attn_output = out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def install_sts_real(
    target_model: torch.nn.Module,
    draft_model: torch.nn.Module,
    *,
    token_budget: int,
    prompt_len: int,
    calib_tokens: int = STS_CALIB_TOKENS,
    calib_topk: int = STS_CALIB_TOPK,
) -> dict[str, Any]:
    """Patch every layer of both models (draft: dense attn + per-layer top-k
    store; target: exact top-k attn gated by the mapped draft layer's top-k,
    through the offline head map). `prompt_len` is shared: both models see
    the identical tokenized input_ids in this harness (same tokenizer
    family, vocab_size 128256 confirmed for both checkpoints)."""
    t_type = getattr(target_model.config, "model_type", "")
    d_type = getattr(draft_model.config, "model_type", "")
    if t_type not in {"llama", "qwen2", "mistral"} or d_type not in {"llama", "qwen2", "mistral"}:
        raise ValueError(f"unsupported model_type(s) for sts real patch: target={t_type} draft={d_type}")
    if token_budget <= 0:
        raise ValueError(f"sts token_budget must be positive, got {token_budget}")
    t_layers = list(target_model.model.layers)
    d_layers = list(draft_model.model.layers)
    n_t, n_d = len(t_layers), len(d_layers)
    t_heads = int(target_model.config.num_attention_heads)
    d_heads = int(draft_model.config.num_attention_heads)
    if t_heads != d_heads:
        raise ValueError(
            f"sts real draft requires matching query-head counts (target={t_heads} "
            f"draft={d_heads}); an unequal count would need an additional head-count "
            "remap not implemented here"
        )
    layer_map = layer_band_map(n_d, n_t)
    calib_len = max(1, min(int(calib_tokens), int(prompt_len)))
    shared: dict[str, Any] = {
        "draft_topk": {},
        # Activation evidence (review requirement): proves the
        # real draft forward actually ran every decode step, rather than
        # silently falling back. draft_layer_hits increments once per draft
        # layer per step (expect gen_steps * n_draft_layers); target_layer_hits
        # once per target layer per step using a real mapped draft top-k
        # (expect gen_steps * n_target_layers); target_fallback_hits should
        # stay 0 -- it only fires if a target layer ran before its mapped
        # draft layer wrote a top-k this step (would indicate a causality bug).
        "draft_layer_hits": 0,
        "target_layer_hits": 0,
        "target_fallback_hits": 0,
    }

    draft_patched: list[torch.nn.Module] = []
    for layer_idx, layer in enumerate(d_layers):
        attn = layer.self_attn
        if hasattr(attn, "_sts_original_forward"):
            raise RuntimeError("sts real patch already installed on draft model")
        attn._sts_original_forward = attn.forward
        attn._sts_own_layer_idx = layer_idx
        attn._sts_role = "draft"
        attn._sts_token_budget = int(token_budget)
        attn._sts_calib_len = calib_len
        attn._sts_calib_topk = int(calib_topk)
        attn._sts_calib_chunks = []
        attn._sts_head_map = None
        attn._sts_index_ready = False
        attn._sts_shared = shared
        attn.forward = MethodType(_sts_real_forward, attn)
        draft_patched.append(attn)

    target_patched: list[torch.nn.Module] = []
    for layer_idx, layer in enumerate(t_layers):
        attn = layer.self_attn
        if hasattr(attn, "_sts_original_forward"):
            raise RuntimeError("sts real patch already installed on target model")
        attn._sts_original_forward = attn.forward
        attn._sts_own_layer_idx = layer_idx
        attn._sts_role = "target"
        attn._sts_draft_layer_idx = int(layer_map[layer_idx])
        attn._sts_token_budget = int(token_budget)
        attn._sts_calib_len = calib_len
        attn._sts_calib_topk = int(calib_topk)
        attn._sts_calib_chunks = []
        attn._sts_head_map = None
        attn._sts_index_ready = False
        attn._sts_shared = shared
        attn.forward = MethodType(_sts_real_forward, attn)
        target_patched.append(attn)

    return {
        "target_patched": target_patched,
        "draft_patched": draft_patched,
        "shared": shared,
        "layer_map": layer_map,
        "n_target_layers": n_t,
        "n_draft_layers": n_d,
        "n_heads": t_heads,
        "calib_len": calib_len,
    }


def build_head_maps_real(handle: dict[str, Any], target_cache: Any, draft_cache: Any) -> dict[str, Any]:
    """Offline step 1 for the real draft: one calib signature per draft layer
    (computed once against the draft's own KV cache), then each target
    layer's signature is Jaccard-matched against its mapped draft layer's
    signature (not a fixed single proxy source)."""
    calib_len = int(handle["calib_len"])
    draft_sets: dict[int, torch.Tensor] = {}
    for attn in handle["draft_patched"]:
        draft_sets[int(attn._sts_own_layer_idx)] = _calib_topk_sets(attn, draft_cache)

    per_layer: dict[str, Any] = {}
    for attn in handle["target_patched"]:
        t_idx = int(attn._sts_own_layer_idx)
        d_idx = int(attn._sts_draft_layer_idx)
        tgt_sets = _calib_topk_sets(attn, target_cache)
        d_sets = draft_sets[d_idx]
        jacc = _jaccard_matrix(tgt_sets, d_sets, calib_len)
        head_map = jacc.argmax(dim=1)
        best = jacc.gather(1, head_map.view(-1, 1)).squeeze(1)
        identity = torch.arange(head_map.shape[0], device=head_map.device)
        attn._sts_head_map = head_map.detach()
        per_layer[str(t_idx)] = {
            "draft_layer": d_idx,
            "mean_best_jaccard": float(best.mean().item()),
            "min_best_jaccard": float(best.min().item()),
            "frac_identity_map": float((head_map == identity).float().mean().item()),
        }

    for attn in handle["target_patched"] + handle["draft_patched"]:
        attn._sts_index_ready = True
        attn._sts_calib_chunks = []
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_best = [v["mean_best_jaccard"] for v in per_layer.values()]
    return {
        "draft_model": "real",
        "n_draft_layers": handle["n_draft_layers"],
        "n_target_layers": handle["n_target_layers"],
        "layer_map": handle["layer_map"],
        "calib_len": calib_len,
        "calib_topk": int(handle["target_patched"][0]._sts_calib_topk) if handle["target_patched"] else None,
        "n_target_layers_mapped": len(per_layer),
        "mean_best_jaccard_overall": float(sum(all_best) / len(all_best)) if all_best else None,
        "per_layer": per_layer,
    }


def restore_sts_real(handle: dict[str, Any]) -> None:
    for attn in handle["target_patched"] + handle["draft_patched"]:
        original = getattr(attn, "_sts_original_forward", None)
        if original is not None:
            attn.forward = original
        for name in _REAL_ATTRS:
            if hasattr(attn, name):
                delattr(attn, name)


def generate_with_real_draft(
    target_model: torch.nn.Module,
    draft_model: torch.nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    target_cache: Any,
    draft_cache: Any,
    *,
    context_tokens: int,
    gen_steps: int,
    shared: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Greedy decode, mirroring e2e._greedy_generate_text's re-feed-last-
    context-token convention exactly (same trick used by every other arm in
    this harness -- dense/PQ-HSA/Quest/SnapKV/RA all call it via _generate,
    so this keeps the STS arm's cache-continuation semantics identical, not
    a separate/inconsistent convention). Runs the draft model's *entire*
    forward to completion before the target forward starts, each step --
    this is the real speculative-style pipeline the proxy could not do
    (draft and target were the same forward pass there). Wall-clock timed
    per model with explicit CUDA syncs so ms/token is comparable to a real
    1B-vs-8B cost delta (not measurable in the proxy mode)."""
    t_device = e2e._model_input_device(target_model, fallback=input_ids.device)
    d_device = e2e._model_input_device(draft_model, fallback=input_ids.device)
    t_ids = input_ids[:, context_tokens - 1 : context_tokens].to(t_device)
    d_ids = input_ids[:, context_tokens - 1 : context_tokens].to(d_device)
    gen_ids: list[int] = []
    draft_ms = 0.0
    target_ms = 0.0
    cuda = torch.cuda.is_available()
    with torch.no_grad():
        for _ in range(gen_steps):
            if cuda:
                torch.cuda.synchronize(d_device)
            t0 = time.perf_counter()
            d_out = draft_model(
                input_ids=d_ids, past_key_values=draft_cache, use_cache=True, logits_to_keep=1,
            )
            draft_cache = d_out.past_key_values
            if cuda:
                torch.cuda.synchronize(d_device)
            draft_ms += (time.perf_counter() - t0) * 1000.0
            del d_out

            if cuda:
                torch.cuda.synchronize(t_device)
            t1 = time.perf_counter()
            t_out = target_model(
                input_ids=t_ids, past_key_values=target_cache, use_cache=True, logits_to_keep=1,
            )
            target_cache = t_out.past_key_values
            if cuda:
                torch.cuda.synchronize(t_device)
            target_ms += (time.perf_counter() - t1) * 1000.0

            next_tok = t_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_ids.append(int(next_tok[0, 0]))
            t_ids = next_tok.to(t_device)
            d_ids = next_tok.to(d_device)
            del t_out
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    n = max(1, gen_steps)
    stats = {
        "draft_ms_per_token": draft_ms / n,
        "target_ms_per_token": target_ms / n,
        "draft_wall_ms_total": draft_ms,
        "target_wall_ms_total": target_ms,
        "gen_steps": int(gen_steps),
    }
    if shared is not None:
        stats["draft_layer_hits"] = int(shared.get("draft_layer_hits", 0))
        stats["target_layer_hits"] = int(shared.get("target_layer_hits", 0))
        stats["target_fallback_hits"] = int(shared.get("target_fallback_hits", 0))
    return text, stats


def report_metadata_real(
    *,
    token_budget: int,
    budget_p: float | None,
    draft_model_path: str,
    n_draft_layers: int,
    n_target_layers: int,
    calib_tokens: int = STS_CALIB_TOKENS,
    calib_topk: int = STS_CALIB_TOPK,
    head_map_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "token_budget": int(token_budget),
        "sts_paper": "arXiv:2605.15508v3, Efficient Sparse Attention with Speculative Token Sparsity",
        "implementation": "harness reproduction, not official (no public code)",
        "sts_selector": "pure top-k selector, no background term (unlike PQ-HSA)",
        "sts_draft_model_requested": "meta-llama/Llama-3.2-1B-Instruct",
        "sts_draft_model_available": True,
        "sts_draft_model_path": draft_model_path,
        "sts_draft_model_source": "meta-llama/Llama-3.2-1B-Instruct",
        "sts_draft_real": True,
        "sts_n_draft_layers": int(n_draft_layers),
        "sts_n_target_layers": int(n_target_layers),
        "sts_calib_tokens": int(calib_tokens),
        "sts_calib_topk": int(calib_topk),
        "budget_formula": "16*ceil((4+128+ceil(p*max(0,L-132)))/16)",
        "simplifications": [
            "real Llama-3.2-1B-Instruct draft (16 layers, 32/8 heads, head_dim 64) alongside "
            "real Llama-3.1-8B-Instruct target (32 layers, 32/8 heads, head_dim 128); "
            "both share the Llama-3 tokenizer (vocab_size 128256 confirmed equal)",
            "layer map: target layer t -> draft layer floor(t*n_draft/n_target) (proportional "
            "depth band; ratio exactly 2 here, so every draft layer gates 2 target layers)",
            "head map: offline Jaccard overlap of top-k index sets on a calibration prefix, "
            "computed once per (target layer, its mapped draft layer) pair -- not assumed identity",
            "per decode step the draft's full forward (all draft layers, its own KV cache) "
            "completes before the target forward starts, so every target layer (0..n_target-1) "
            "can be sparse -- unlike the proxy draft, which could only sparsify layers after its "
            "single proxy layer within the same forward pass",
            "draft and target are separate model instances/caches; draft overhead is measured "
            "as real wall-clock ms/token (CUDA-synced) and is comparable to a genuine 1B-vs-8B cost delta",
        ],
    }
    if budget_p is not None:
        meta["sts_budget_p"] = float(budget_p)
        meta["sts_budget_mode"] = "grid"
    if head_map_stats is not None:
        meta["sts_head_map_stats"] = head_map_stats
    return meta
