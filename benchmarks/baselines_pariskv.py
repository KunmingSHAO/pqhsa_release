"""ParisKV harness reproduction (arXiv 2602.07721v3).

The official ParisKV repository has no Llama adapter (Qwen-only release) and
ships no RULER/niah driver, so an official-stack apples-to-apples run is
infeasible. This file is a **quality-only harness reproduction**: it reproduces the paper's *selection* logic (which
tokens get exact attention) inside this repo's own HF eval loop — the same
pattern already used for Quest/SnapKV/STS (`baselines_snapkv.py`,
`baselines_sts.py`) — without the paper's CUDA kernels, UVA/CPU-offload, or
decode-time incremental buffer update (none of those change *which tokens are
selected*, only how fast the selection runs).

Paper recipe (Sec. 4, Appendix B of the ParisKV paper):
  1. L2-normalize K/Q per head, apply a shared orthogonal rotation onto the
     unit hypersphere (paper: SRHT; here: exact Haar-random rotation via QR
     of a Gaussian matrix — see DEVIATIONS).
  2. Split the rotated D-dim vector into B contiguous subspaces of dim m=D/B.
  3. Stage I (coarse candidate generation): in each subspace, assign every
     key to the nearest "sign-pattern" centroid (data-independent, evenly
     covers the hypersphere) — nearest-centroid assignment to an
     equal-magnitude sign-pattern set reduces exactly to sign(u); the query
     scores each key's assigned centroid (cheap dot product), only the
     top-rho fraction per subspace contributes a non-zero bonus, and within
     that top-rho window keys are split into L=6 percentile tiers with
     weights {6,5,4,3,2,1} at cutoffs {5%,15%,30%,50%,75%,100%}. Summing the
     per-subspace bonus gives an integer collision score in [0,6B]; the
     top-beta fraction of all keys (typically beta=5-10%) becomes the
     Stage-I candidate pool.
  4. Stage II (RSQ-IP reranking): candidates are reranked by an
     alignment-corrected estimate of the raw <k,q> score from a compact 4-bit
     (1 sign + 3 magnitude) direction code plus the exact subspace radius;
     the final top-k (`final_topk`) candidates are kept.
  5. Exact attention (Eq. 3) is restricted to sink + local + the Stage-II
     top-k — a pure *selector*, no background/residual compensation term
     (unlike PQ-HSA's PQ-score background) — matches the "truncation" family
     (Quest/SnapKV/STS) structurally, differing only in the selector.

DEVIATIONS from the paper (stated explicitly, quality-only reproduction):
  * **Rotation**: SRHT (SubsampledRandomizedHadamardTransform, an O(D log D)
    fast near-isotropic transform) is replaced by an exact Haar-random
    orthogonal matrix (QR decomposition of a Gaussian matrix), generated
    once and shared across all layers/heads/decode-steps (matching the
    paper's "shared rotation"). The paper's own Remark (Appendix B.1.1)
    states SRHT is used only for speed and "empirically induces
    near-isotropic coordinate statistics" that are treated as a stable
    approximation to the Haar-random analysis (Prop. 4.1/B.1). Using true
    Haar-random rotation here is a strictly *more* faithful choice for a
    quality-only reproduction (exact Beta-prior isotropy, no SRHT
    approximation error) at the cost of O(D^2) instead of O(D log D) per
    rotation — acceptable since CUDA-kernel speed is explicitly out of scope
    for this task.
  * **Centroid codebook**: never materializes the 2^m centroid set. Because
    Omega = all equal-magnitude sign patterns, "nearest centroid to unit
    direction u" is *exactly* sign(u) coordinate-wise (provable: maximizing
    sum_j sign(u_j) c_j over c in {+-1/sqrt(m)}^m picks c_j = sign(u_j)).
    This is not an approximation, just an algebraic simplification.
  * **RSQ-IP reranking formula**: the paper's Eq. (7)/(19)-(22) were not
    available as transcribed text/LaTeX. The estimator implemented here
    (`_rsqip_rerank_scores`) is reconstructed from the surrounding prose —
    radius-direction (polar) decomposition with exact subspace radius
    (paper: "we do not quantize radii... retain exact r_i,b", Appendix
    B.1.3), a 4-bit (1 sign + 3-bit magnitude) direction code with
    Lloyd-Max-quantized reconstruction levels, and a per-subspace alignment
    correction alpha_i,b = <u_i,b, v_hat_i,b> (RaBitQ-style, as the paper
    cites Gao & Long 2024) — not transcribed equation-for-equation. The
    *ranking* this produces (which candidates make the final top-k) is the
    faithful target; the exact numeric estimator is a reconstruction.
  * **Quantization levels**: derived by Monte-Carlo Lloyd-Max quantization
    of |u_j| samples drawn from random unit vectors in R^m (this is exactly
    the paper's claimed *data-independent* target distribution — Beta(1/2,
    (m-1)/2) after the sqrt — so this reproduces the "offline, once, shared
    across layers/subspaces" property without needing a closed-form Beta
    quantile solver (avoids a scipy dependency; only numpy/torch used).
  * **No decode-time incremental buffer/CPU-offload/UVA**: this harness
    keeps the full-precision KV cache resident (like Quest/SnapKV's harness
    reproductions) and recomputes normalize+rotate+select from scratch each
    decode step directly from the live HF cache, rather than maintaining the
    paper's GPU-resident summary + CPU-offloaded full KV + sliding
    sink/local/update-buffer regions. This is a systems-only simplification
    (memory/latency), not a selection-quality difference: the *quality* of
    which tokens get selected is unaffected because every decode step still
    sees the same up-to-date K/Q history the paper's incrementally-updated
    summaries would represent.
  * **Final softmax**: computed on full-precision K/V for the selected
    token set (matches the paper exactly — "Full-precision KV are fetched
    only for the final selected Top-k tokens"; RSQ-IP is only used to decide
    *which* tokens are fetched, never to compute the attention itself).
  * **Budget conversion**: the paper's budget is a fixed absolute
    `final_topk` (default 100), not a percentage.
    for apples-to-apples comparison with PQ-HSA/Quest/SnapKV/STS on the same
    x-axis, `final_topk = token_budget - sink - local_window` where
    `token_budget` comes from the shared aligned-budget formula
    (`aligned_token_budget` below, identical formula to
    `baselines_sts.py`/`baselines_snapkv.py`).
"""
from __future__ import annotations

import math
from types import MethodType
from typing import Any

import torch

import e2e_pq_param_sweep as e2e

# ---- paper-derived hyperparameters ----------------------------------------
PARISKV_B = 16                 # number of subspaces (paper's D=128 example: B=16, m=8)
PARISKV_RHO = 0.30             # top-rho fraction per subspace that can score (rho >= beta, paper constraint)
PARISKV_BETA = 0.08            # top-beta fraction of all keys kept as Stage-I candidates (paper: "typically 5-10%")
PARISKV_TIER_WEIGHTS = (6, 5, 4, 3, 2, 1)              # L=6 tiers, paper Appendix B.2.1
PARISKV_TIER_PERCENTILES = (0.05, 0.15, 0.30, 0.50, 0.75, 1.00)  # cumulative, within top-rho
PARISKV_ROTATION_SEED = 20260918  # fixed shared rotation seed
PARISKV_MIN_FINAL_TOPK = 16

_ATTRS = (
    "_pariskv_original_forward",
    "_pariskv_layer_idx",
    "_pariskv_sink",
    "_pariskv_local",
    "_pariskv_final_topk",
    "_pariskv_B",
    "_pariskv_rho",
    "_pariskv_beta",
    "_pariskv_rotation",
    "_pariskv_stats_buf",
)

_ROTATION_CACHE: dict[tuple[int, int], torch.Tensor] = {}
_QUANT_CACHE: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def aligned_token_budget(p: float, length: int, sink: int = 4, local: int = 128) -> int:
    """Aligned-budget grid formula (same as baselines_sts.py / baselines_snapkv.py).
    The result is the *total* exact-attention
    budget; final_topk = token_budget - sink - local (see install_pariskv_patch)."""
    inner = sink + local + math.ceil(p * max(0, length - sink - local))
    return 16 * math.ceil(inner / 16)


def min_feasible_budget(sink: int = 4, local_window: int = 128, min_topk: int = PARISKV_MIN_FINAL_TOPK) -> int:
    return int(sink) + int(local_window) + int(min_topk)


# ---- offline, data-independent building blocks -----------------------------

def _get_rotation(dim: int, *, seed: int = PARISKV_ROTATION_SEED, device=None, dtype=torch.float32) -> torch.Tensor:
    """Shared Haar-random orthogonal DxD rotation (stand-in for SRHT — see
    module docstring DEVIATIONS). Cached per (dim, seed); generated on CPU for
    reproducibility, then cast to the caller's device/dtype."""
    key = (int(dim), int(seed))
    if key not in _ROTATION_CACHE:
        g = torch.Generator().manual_seed(int(seed))
        a = torch.randn(dim, dim, generator=g, dtype=torch.float64)
        q, r = torch.linalg.qr(a)
        d = torch.sign(torch.diagonal(r))
        d = torch.where(d == 0, torch.ones_like(d), d)
        q = (q * d.unsqueeze(0)).to(torch.float32)  # uniform (Haar) orthogonal, not just QR-biased
        _ROTATION_CACHE[key] = q.detach()
    rot = _ROTATION_CACHE[key]
    return rot.to(device=device, dtype=dtype)


def _derive_quant_levels(
    m: int, *, n_samples: int = 200_000, n_iter: int = 40, seed: int = 12345,
    device=None, dtype=torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Offline Lloyd-Max scalar quantizer for coordinate magnitude
    X=|u_j|, u uniform on S^{m-1} (paper: X=sqrt(Y), Y~Beta(1/2,(m-1)/2),
    Appendix B.1.2). Derived once per m via Monte-Carlo sampling of random
    unit vectors (data-independent — does not touch any real key/query).
    Returns (tau[7] thresholds, levels[8] reconstruction values)."""
    if m in _QUANT_CACHE:
        tau, levels = _QUANT_CACHE[m]
        return tau.to(device=device, dtype=dtype), levels.to(device=device, dtype=dtype)
    g = torch.Generator().manual_seed(int(seed) + m)
    gauss = torch.randn(n_samples, m, generator=g, dtype=torch.float64)
    u = gauss / gauss.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x = u.abs().reshape(-1)
    qs = torch.linspace(1.0 / 16.0, 15.0 / 16.0, 8, dtype=torch.float64)
    levels_t = torch.quantile(x, qs)
    levels_t, _ = torch.sort(levels_t)
    for _ in range(n_iter):
        tau_t = (levels_t[:-1] + levels_t[1:]) / 2.0
        idx = torch.bucketize(x, tau_t)
        new_levels = levels_t.clone()
        for t in range(8):
            mask = idx == t
            if mask.any():
                new_levels[t] = x[mask].mean()
        levels_t = new_levels
    tau_t = (levels_t[:-1] + levels_t[1:]) / 2.0
    tau32 = tau_t.to(torch.float32).detach().clone()
    levels32 = levels_t.to(torch.float32).detach().clone()
    _QUANT_CACHE[m] = (tau32, levels32)
    return tau32.to(device=device, dtype=dtype), levels32.to(device=device, dtype=dtype)


def _quantize_direction(u: torch.Tensor, tau: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """1-bit sign + 3-bit magnitude quantization of a unit-subspace direction
    (paper Appendix B.2.2 Step 1). u: [..., m] -> v: [..., m] reconstructed."""
    sign = torch.where(u >= 0, torch.ones_like(u), -torch.ones_like(u))
    mag = u.abs()
    idx = torch.bucketize(mag.contiguous(), tau)
    a = levels[idx]
    return sign * a


def _l2n(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


# ---- core selection logic (unit-testable in isolation) --------------------

def _stage1_collision_scores(
    q_rot: torch.Tensor,   # [H, B, m] rotated unit query, split into subspaces
    k_rot: torch.Tensor,   # [H, N, B, m] rotated unit-*direction* keys (need not be pre-normalized: only sign() is used per-subspace)
    *,
    rho: float,
    tier_weights: tuple[int, ...],
    tier_percentiles: tuple[float, ...],
) -> torch.Tensor:
    """Paper Appendix B.2.1: per subspace, centroid-of-key = sign(direction);
    proxy score = q^T centroid; only the top-rho fraction of keys (by proxy
    score) contributes a non-zero bonus, tiered into L=len(tier_weights)
    percentile bands. Returns the summed integer collision score [H, N] in
    [0, sum(tier_weights) * n_subspaces]."""
    h, n, n_subspaces, m = k_rot.shape
    device = k_rot.device
    sign_k = torch.where(k_rot >= 0, torch.ones_like(k_rot), -torch.ones_like(k_rot))  # centroid = sign(u), any radius
    proxy = torch.einsum("hbm,hnbm->hnb", q_rot, sign_k)  # [H,N,B]

    top_n = max(1, min(n, math.ceil(rho * n)))
    collision = torch.zeros(h, n, dtype=torch.int64, device=device)
    for b in range(n_subspaces):
        col = proxy[:, :, b]
        _, idx = col.topk(top_n, dim=-1, largest=True, sorted=True)  # [H, top_n], best first
        bonus_b = torch.zeros(h, n, dtype=torch.int64, device=device)
        prev_cut = 0
        for weight, pct in zip(tier_weights, tier_percentiles):
            cut = min(top_n, max(prev_cut, math.ceil(pct * top_n)))
            if cut > prev_cut:
                sel = idx[:, prev_cut:cut]
                bonus_b.scatter_(1, sel, int(weight))
                prev_cut = cut
            if prev_cut >= top_n:
                break
        collision += bonus_b
    return collision


def _stage2_rerank_scores(
    q_rot: torch.Tensor,        # [H, B, m] rotated unit query, split into subspaces
    k_rot_cand: torch.Tensor,   # [H, C, B, m] rotated (unnormalized-per-subspace) candidate keys
    k_norm_cand: torch.Tensor,  # [H, C] exact ||k_i|| for each candidate
) -> torch.Tensor:
    """Paper Appendix B.2.2 (RSQ-IP, reconstructed from prose -- see module
    docstring DEVIATIONS): exact per-subspace radius + 4-bit quantized
    direction + alignment correction, summed over subspaces and rescaled by
    the exact key norm. Returns estimated raw <k_i, q> score, [H, C]."""
    m = k_rot_cand.shape[-1]
    device = k_rot_cand.device
    calc_dtype = k_rot_cand.dtype
    tau, levels = _derive_quant_levels(m, device=device, dtype=calc_dtype)
    r = k_rot_cand.norm(dim=-1)                                  # exact subspace radius (not quantized, per B.1.3)
    u = k_rot_cand / r.unsqueeze(-1).clamp_min(1e-12)
    v = _quantize_direction(u, tau, levels)                      # 4-bit reconstructed direction
    v_norm = v.norm(dim=-1).clamp_min(1e-12)
    v_hat = v / v_norm.unsqueeze(-1)
    alpha = (u * v_hat).sum(-1).clamp_min(1e-3)                  # alignment correction (RaBitQ-style)
    qk_hat = torch.einsum("hbm,hcbm->hcb", q_rot, v_hat)
    subspace_est = r * qk_hat / alpha
    return k_norm_cand * subspace_est.sum(-1)                    # [H, C]


def select_pariskv_indices(
    q: torch.Tensor,   # [H, D] query, one decode step, per head (post-RoPE)
    k: torch.Tensor,   # [H, N, D] full-precision keys (post-RoPE), the "retrieval zone"
    *,
    final_topk: int,
    n_subspaces: int = PARISKV_B,
    rho: float = PARISKV_RHO,
    beta: float = PARISKV_BETA,
    tier_weights: tuple[int, ...] = PARISKV_TIER_WEIGHTS,
    tier_percentiles: tuple[float, ...] = PARISKV_TIER_PERCENTILES,
    rotation: torch.Tensor | None = None,
    rotation_seed: int = PARISKV_ROTATION_SEED,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Two-stage ParisKV selection (Sec. 4.2.2 / Appendix B.2). Returns
    (selected_idx[H, kk] into the N axis of k, stats). kk = min(final_topk, N)."""
    h, n, d = k.shape
    if q.shape != (h, d):
        raise ValueError(f"q shape {tuple(q.shape)} != {(h, d)}")
    if n <= final_topk:
        idx = torch.arange(n, device=k.device).view(1, n).expand(h, n).clone()
        return idx, {"n": n, "stage2_skipped": True, "n_candidates": n}
    if d % n_subspaces != 0:
        raise ValueError(f"head_dim {d} not divisible by n_subspaces {n_subspaces}")
    m = d // n_subspaces
    device = k.device
    calc_dtype = torch.float32
    if rotation is not None:
        # Always re-home the caller-supplied rotation onto k's device/dtype:
        # install_pariskv_patch caches one rotation per process and shares it
        # across all layers/decode-steps, so it must not be silently stuck on
        # whatever device it was first materialized on (a CPU-cached rotation
        # vs CUDA keys would otherwise fail). `.to()` is a cheap no-op once already co-located.
        rot = rotation.to(device=device, dtype=calc_dtype)
    else:
        rot = _get_rotation(d, seed=rotation_seed, device=device, dtype=calc_dtype)

    k32 = k.to(calc_dtype)
    q32 = q.to(calc_dtype)
    k_norm = k32.norm(dim=-1)              # [H, N] exact magnitude ||k_i||
    k_hat = _l2n(k32, dim=-1)
    q_hat = _l2n(q32, dim=-1)
    k_rot = torch.einsum("hnd,ed->hne", k_hat, rot).view(h, n, n_subspaces, m)
    q_rot = torch.einsum("hd,ed->he", q_hat, rot).view(h, n_subspaces, m)

    collision = _stage1_collision_scores(
        q_rot, k_rot, rho=rho, tier_weights=tier_weights, tier_percentiles=tier_percentiles,
    )

    n_cand = max(final_topk, math.ceil(beta * n))
    n_cand = min(n_cand, n)
    _, cand_idx = collision.topk(n_cand, dim=-1, largest=True, sorted=False)  # [H, n_cand]

    # -- Stage II: RSQ-IP-style alignment-corrected rerank -------------------
    gather_idx = cand_idx.unsqueeze(-1).unsqueeze(-1).expand(h, n_cand, n_subspaces, m)
    k_rot_cand = torch.gather(k_rot, 1, gather_idx)             # [H,n_cand,B,m]
    k_norm_cand = torch.gather(k_norm, 1, cand_idx)             # [H,n_cand]
    raw_est = _stage2_rerank_scores(q_rot, k_rot_cand, k_norm_cand)  # [H,n_cand]

    kk = min(final_topk, n_cand)
    _, top_local = raw_est.topk(kk, dim=-1, largest=True, sorted=False)
    final_idx = torch.gather(cand_idx, 1, top_local)             # [H, kk] into N

    stats = {
        "n": n,
        "n_subspaces": n_subspaces,
        "subspace_dim": m,
        "n_candidates": int(n_cand),
        "final_topk": int(kk),
        "mean_collision_score_selected": float(
            torch.gather(collision, 1, final_idx).float().mean().item()
        ),
        "collision_score_max_possible": int(sum(tier_weights) * n_subspaces),
    }
    return final_idx, stats


def pariskv_attention(
    query_states: torch.Tensor,  # [1, H, 1, D]
    key_states: torch.Tensor,    # [1, H, N, D] full precision, GQA-repeated, post-RoPE
    value_states: torch.Tensor,  # [1, H, N, D]
    *,
    sink: int,
    local_window: int,
    final_topk: int,
    n_subspaces: int = PARISKV_B,
    rho: float = PARISKV_RHO,
    beta: float = PARISKV_BETA,
    rotation: torch.Tensor | None = None,
    rotation_seed: int = PARISKV_ROTATION_SEED,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Sink + Stage-I/II-selected + local exact attention, no background term
    (Eq. 3, restricted softmax — same truncation family as Quest/SnapKV/STS)."""
    bsz, h, q_len, d = query_states.shape
    if bsz != 1 or q_len != 1:
        raise ValueError("pariskv_attention: harness assumes bsz=1, q_len=1 decode step")
    n = key_states.shape[2]
    device = key_states.device
    scale = 1.0 / math.sqrt(d)

    sink_n = max(0, min(int(sink), n))
    local_n = max(0, min(int(local_window), max(0, n - sink_n)))
    zone_start = sink_n
    zone_end = max(sink_n, n - local_n)
    zone_len = zone_end - zone_start

    if zone_len <= 0:
        sel_idx = None
        stats: dict[str, Any] = {"zone_len": 0}
    elif zone_len <= final_topk:
        zone_idx = torch.arange(zone_start, zone_end, device=device)
        sel_idx = zone_idx.view(1, -1).expand(h, -1)
        stats = {"zone_len": zone_len, "stage2_skipped": True, "n_candidates": zone_len}
    else:
        q_vec = query_states[0, :, 0, :]
        k_zone = key_states[0, :, zone_start:zone_end, :]
        sel_local, stats = select_pariskv_indices(
            q_vec, k_zone,
            final_topk=final_topk, n_subspaces=n_subspaces, rho=rho, beta=beta,
            rotation=rotation, rotation_seed=rotation_seed,
        )
        sel_idx = sel_local + zone_start

    parts = []
    if sink_n > 0:
        parts.append(torch.arange(0, sink_n, device=device).view(1, -1).expand(h, -1))
    if sel_idx is not None:
        parts.append(sel_idx)
    if local_n > 0:
        parts.append(torch.arange(n - local_n, n, device=device).view(1, -1).expand(h, -1))
    full_idx = torch.cat(parts, dim=-1) if parts else torch.zeros(h, 0, dtype=torch.long, device=device)
    stats["exact_budget"] = int(full_idx.shape[-1])

    full_idx_b = full_idx.unsqueeze(0)  # [1,H,C]
    gather = full_idx_b.unsqueeze(-1).expand(1, h, full_idx_b.shape[-1], d)
    k_sel = torch.gather(key_states, 2, gather)
    v_sel = torch.gather(value_states, 2, gather)
    logits = torch.matmul(query_states.float(), k_sel.float().transpose(-1, -2)) * scale
    weights = torch.softmax(logits, dim=-1).to(value_states.dtype)
    context = torch.matmul(weights, v_sel)
    return context, stats


# ---- model patch (mirrors baselines_sts.py / eval_task_utility.py's quest) --

def _pariskv_llama_forward(
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
    if len(input_shape) != 2 or input_shape[1] != 1 or past_key_values is None:
        return self._pariskv_original_forward(
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

    context, stats = pariskv_attention(
        query_states, key_states, value_states,
        sink=int(self._pariskv_sink),
        local_window=int(self._pariskv_local),
        final_topk=int(self._pariskv_final_topk),
        n_subspaces=int(self._pariskv_B),
        rho=float(self._pariskv_rho),
        beta=float(self._pariskv_beta),
        rotation=self._pariskv_rotation,
    )
    buf = getattr(self, "_pariskv_stats_buf", None)
    if buf is not None:
        buf.append(stats)

    attn_output = context.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def install_pariskv_patch(
    model: torch.nn.Module,
    *,
    token_budget: int,
    sink: int = 4,
    local_window: int = 128,
    n_subspaces: int = PARISKV_B,
    rho: float = PARISKV_RHO,
    beta: float = PARISKV_BETA,
    min_final_topk: int = PARISKV_MIN_FINAL_TOPK,
    rotation_seed: int = PARISKV_ROTATION_SEED,
) -> tuple[list[torch.nn.Module], int]:
    model_type = getattr(model.config, "model_type", "")
    if model_type not in {"llama", "qwen2", "mistral"}:
        raise ValueError(f"unsupported model_type for pariskv patch: {model_type}")
    if token_budget <= 0:
        raise ValueError(f"pariskv token_budget must be positive, got {token_budget}")

    final_topk = max(int(min_final_topk), int(token_budget) - int(sink) - int(local_window))
    layers = list(model.model.layers)
    head_dim = int(getattr(model.config, "head_dim", 0) or 0) or (
        int(model.config.hidden_size) // int(model.config.num_attention_heads)
    )
    if head_dim % n_subspaces != 0:
        raise ValueError(f"head_dim {head_dim} not divisible by n_subspaces {n_subspaces}")
    model_device = next(model.parameters()).device
    rotation = _get_rotation(head_dim, seed=rotation_seed, device=model_device)

    patched: list[torch.nn.Module] = []
    for layer_idx, decoder_layer in enumerate(layers):
        attn = decoder_layer.self_attn
        if hasattr(attn, "_pariskv_original_forward"):
            raise RuntimeError("pariskv patch already installed")
        attn._pariskv_original_forward = attn.forward
        attn._pariskv_layer_idx = layer_idx
        attn._pariskv_sink = int(sink)
        attn._pariskv_local = int(local_window)
        attn._pariskv_final_topk = int(final_topk)
        attn._pariskv_B = int(n_subspaces)
        attn._pariskv_rho = float(rho)
        attn._pariskv_beta = float(beta)
        attn._pariskv_rotation = rotation
        attn._pariskv_stats_buf = []
        attn.forward = MethodType(_pariskv_llama_forward, attn)
        patched.append(attn)
    return patched, final_topk


def restore_pariskv_patch(patched: list[torch.nn.Module]) -> None:
    for attn in patched:
        original = getattr(attn, "_pariskv_original_forward", None)
        if original is not None:
            attn.forward = original
        for name in _ATTRS:
            if hasattr(attn, name):
                delattr(attn, name)


def collect_pariskv_stats(patched: list[torch.nn.Module]) -> dict[str, Any]:
    n = 0
    exact_budget: list[int] = []
    stage2_skipped = 0
    collision_frac: list[float] = []
    for attn in patched:
        for row in getattr(attn, "_pariskv_stats_buf", None) or []:
            n += 1
            if "exact_budget" in row:
                exact_budget.append(int(row["exact_budget"]))
            if row.get("stage2_skipped"):
                stage2_skipped += 1
            if "mean_collision_score_selected" in row and "collision_score_max_possible" in row:
                denom = row["collision_score_max_possible"] or 1
                collision_frac.append(row["mean_collision_score_selected"] / denom)
        if getattr(attn, "_pariskv_stats_buf", None) is not None:
            attn._pariskv_stats_buf = []
    if n == 0:
        return {}
    out: dict[str, Any] = {"n_scan_records": n, "stage2_skipped_frac": stage2_skipped / n}
    if exact_budget:
        out["mean_exact_budget"] = float(sum(exact_budget) / len(exact_budget))
    if collision_frac:
        out["mean_collision_score_frac"] = float(sum(collision_frac) / len(collision_frac))
    return out


def report_metadata(
    *,
    token_budget: int,
    budget_p: float | None,
    sink: int,
    local_window: int,
    final_topk: int,
    n_subspaces: int = PARISKV_B,
    rho: float = PARISKV_RHO,
    beta: float = PARISKV_BETA,
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "token_budget": int(token_budget),
        "pariskv_sink": int(sink),
        "pariskv_local_window": int(local_window),
        "pariskv_final_topk": int(final_topk),
        "pariskv_n_subspaces": int(n_subspaces),
        "pariskv_rho": float(rho),
        "pariskv_beta": float(beta),
        "pariskv_paper": "arXiv:2602.07721v3, ParisKV: Fast and Drift-Robust KV-Cache Retrieval for Long-Context LLMs",
        "implementation": "harness reproduction, not official (official repo has no Llama adapter / no RULER driver)",
        "pariskv_selector": "pure top-k selector (sink+local+Stage-II candidates), no background term (unlike PQ-HSA)",
        "pariskv_budget_formula": "final_topk = token_budget - sink - local_window; token_budget via the aligned-budget formula",
        "simplifications": [
            "rotation: exact Haar-random orthogonal (QR of Gaussian) instead of SRHT -- same shared/fixed-per-model rotation property, exact rather than approximate isotropy (paper's own Remark treats SRHT as an approximation to this)",
            "centroid codebook never materialized: nearest-centroid assignment to an equal-magnitude sign-pattern set is exactly sign(u), computed directly",
            "RSQ-IP rerank estimator reconstructed from prose (Appendix B.1-B.2); paper's Eq. (7)/(19)-(22) are images in the extracted text, not transcribed equation-for-equation",
            "quantization levels derived by Monte-Carlo Lloyd-Max on samples from random unit vectors in R^m (data-independent, matches paper's claimed Beta-prior target distribution) instead of a closed-form Beta quantile solver",
            "no decode-time incremental buffer / CPU offload / UVA: full-precision KV cache stays resident and normalize+rotate+select is recomputed from scratch each decode step (systems simplification only, does not change which tokens are selected)",
            "final softmax attention always uses full-precision K/V restricted to the selected set -- matches the paper (RSQ-IP only decides *which* tokens are fetched, never computes attention itself)",
            "not implemented: custom CUDA kernels (collision/bucket-topk/fused-rerank/UVA), CPU offload -- out of scope (quality-only reproduction; kernel speed not reproduced)",
        ],
    }
    if budget_p is not None:
        meta["pariskv_budget_p"] = float(budget_p)
        meta["pariskv_budget_mode"] = "grid"
    if stats:
        meta["pariskv_stats"] = stats
    return meta
