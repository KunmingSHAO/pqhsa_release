from __future__ import annotations

import argparse
import copy
import csv
import os
_E21_C2_MASS = os.environ.get("PQ_HSA_E21_C2_MASS", "0") == "1"
# Arm C2b: which per-row constant the PQ-free background anchor uses.
#   "mean"     (default) = mean PQ score of the selected exact set  -> original C2
#   "boundary"           = min EXACT rerank logit of the selected set -> arm C2b
_E21_C2_ANCHOR = os.environ.get("PQ_HSA_E21_C2_ANCHOR", "mean")
# Background-mass diagnostic (opt-in; see _e21_bgmass_report).
_E21_BGMASS_LOG = os.environ.get("PQ_HSA_E21_BGMASS_LOG", "0") == "1"
# Arm T (opt-in; see _e21_trunc_epilogue): fixed-selector truncation --
# arm B's all-PQ top-k selection verbatim, background mass AND value set to 0.
# run_ablation_arm.py --arm T also sets this module attribute directly, so the switch
# does not depend on the env surviving any wrapper. Default False -> unchanged.
_E21_TRUNC_ALLPQ = os.environ.get("PQ_HSA_E21_TRUNC_ALLPQ", "0") == "1"
# Probe (opt-in, diagnosis only): records first-step selection ids and the
# in-call T-vs-hybrid output difference; never changes the returned output.
_E21_PROBE = os.environ.get("PQ_HSA_E21_PROBE", "0") == "1"
_E21_PROBE_STEPS = int(os.environ.get("PQ_HSA_E21_PROBE_STEPS", "1"))
_E21_T_STATS: dict = {"hits": 0, "probe": []}
# Needle diagnosis (opt-in, diagnosis only; set as a module attribute by an
# external diagnosis script): per layer/step, whether the needle value
# tokens are in the all-PQ top-k, their weight under hybrid / truncation / dense
# (same query), and the background share of the denominator. Never changes output.
_E21_T2_DIAG = False
# Arm Bx (opt-in, default OFF): arm B with the background VALUE of list j
# replaced by the mean over its UNSELECTED members,
#   vbar_j^B = (sum_{R_j} v - sum_{T∩R_j} v) / |B∩R_j|   (0 when |B∩R_j| = 0);
# selection, background mass M_j and the shared normalisation are B's, verbatim.
# run_ablation_arm.py --arm Bx sets the module attribute directly.
_E21_BG_UNSELECTED_MEAN = os.environ.get("PQ_HSA_E21_BG_UNSELECTED_MEAN", "0") == "1"
_E21_T2_STATE: dict = {"needle_pos": None, "sample": None, "max_steps": 16, "records": []}

# --- (opt-in, default OFF): cross-step retrieval reuse on the EAGER batched-heads
# path.  When PQ_HSA_REUSE_TOPK=1, a layer whose per-row query drift is small
# (min cosine over [H,G] rows >= PQ_HSA_REUSE_COS) and whose stored retrieval is
# younger than PQ_HSA_REUSE_MAX_STEPS steps skips LUT/scan/list-stats/top-k and
# re-uses the stored exact top-k set + stored (PQ) background quantities; the exact
# rerank on the stored set, sink/local window and the softmax mix are recomputed
# with the CURRENT query.  Unset -> byte-identical default path.
_E42_REUSE = os.environ.get("PQ_HSA_REUSE_TOPK", "0") == "1"
_E42_TAU = float(os.environ.get("PQ_HSA_REUSE_COS", "0.98"))
_E42_RMAX = int(os.environ.get("PQ_HSA_REUSE_MAX_STEPS", "8"))
_E42_TIMING = os.environ.get("PQ_HSA_E42_TIMING", "0") == "1"
# aggregation of the per-row cosine for the all-or-nothing layer decision: min (strict) | mean | q<frac>
_E42_AGG = os.environ.get("PQ_HSA_REUSE_AGG", "min").strip().lower()
_E42_JACCARD = os.environ.get("PQ_HSA_E42_JACCARD", "0") == "1"  # diagnostic: top-k set overlap between consecutive retrievals
_E42_TAU_DIAG = float(os.environ.get("PQ_HSA_E42_TAU_DIAG", "0.98"))  # row-level cosine threshold used only by the diagnostic counters
_E42_STATS: dict = {
    "calls": 0, "reuse_hits": 0, "retrievals": 0, "first_retrievals": 0,
    "forced_drift": 0, "forced_rmax": 0, "forced_invalidate": 0,
    "cos_min_sum": 0.0, "cos_min_n": 0, "cos_rowmean_sum": 0.0,
    "jaccard_sum": 0.0, "jaccard_min_sum": 0.0, "jaccard_n": 0, "jaccard_consec_sum": 0.0, "jaccard_consec_n": 0,
    "rows_cos_ge_tau_sum": 0.0, "rows_jac_ge_09_sum": 0.0, "rows_jac_ge_08_sum": 0.0, "rows_both_sum": 0.0,
    "t_reuse_ms": 0.0, "n_reuse_timed": 0, "t_retr_ms": 0.0, "n_retr_timed": 0,
    "graph_path_calls_unreused": 0,
}
_E42_PRINTED = {"first": False, "graph_note": False}


def _e42_summary_line(tag: str = "REUSE-STATS") -> str:
    s = _E42_STATS
    n = max(1, s["cos_min_n"])
    line = (f"[{tag}] calls={s['calls']} reuse_hits={s['reuse_hits']} retrievals={s['retrievals']} "
            f"first={s['first_retrievals']} forced_drift={s['forced_drift']} forced_rmax={s['forced_rmax']} "
            f"forced_invalidate={s['forced_invalidate']} reuse_rate={s['reuse_hits']/max(1,s['calls']):.4f} "
            f"cos_agg_mean={s['cos_min_sum']/n:.4f} cos_rowmean={s['cos_rowmean_sum']/n:.4f} agg={_E42_AGG} tau={_E42_TAU} rmax={_E42_RMAX} graph_unreused={s['graph_path_calls_unreused']}")
    if s["jaccard_n"]:
        line += (f" | topk-set jaccard between successive retrievals: mean={s['jaccard_sum']/s['jaccard_n']:.4f}"
                 f" rowmin_mean={s['jaccard_min_sum']/s['jaccard_n']:.4f} (n={s['jaccard_n']})"
                 f" consecutive-step mean={s['jaccard_consec_sum']/max(1,s['jaccard_consec_n']):.4f} (n={s['jaccard_consec_n']})")
        line += (f" | row-level: frac_rows cos>={_E42_TAU_DIAG}: {s['rows_cos_ge_tau_sum']/s['jaccard_n']:.4f}"
                 f" jac>=0.9: {s['rows_jac_ge_09_sum']/s['jaccard_n']:.4f} jac>=0.8: {s['rows_jac_ge_08_sum']/s['jaccard_n']:.4f}"
                 f" both: {s['rows_both_sum']/s['jaccard_n']:.4f}")
    if s["n_reuse_timed"] or s["n_retr_timed"]:
        line += (f" | timing us/call: reuse={1000.0*s['t_reuse_ms']/max(1,s['n_reuse_timed']):.1f} (n={s['n_reuse_timed']})"
                 f" retrieval={1000.0*s['t_retr_ms']/max(1,s['n_retr_timed']):.1f} (n={s['n_retr_timed']})")
    return line


def _e42_export_stats() -> None:
    try:
        from benchmarks.vllm_backend.pq_hsa_decode_runtime import STATS as _RST
        for k in ("calls", "reuse_hits", "retrievals", "forced_drift", "forced_rmax", "forced_invalidate"):
            _RST["e42_" + k] = int(_E42_STATS[k])
    except Exception:
        pass


if _E42_REUSE:
    import atexit as _atexit

    def _e42_atexit() -> None:
        print(_e42_summary_line("REUSE-FINAL"), flush=True)

    _atexit.register(_e42_atexit)

import itertools
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb as llama_apply_rotary_pos_emb,
)
from transformers.models.llama.modeling_llama import repeat_kv as llama_repeat_kv
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb as qwen3_apply_rotary_pos_emb,
)
from transformers.models.qwen3.modeling_qwen3 import repeat_kv as qwen3_repeat_kv

from math import sqrt

from pq_hsa import (
    AttentionOutput,
    IVFPQConfig,
    IVFPQDecodeAttentionAdapter,
    IVFPQIndex,
    IVFPQSparseAttention,
    SparseAttentionConfig,
    is_triton_available,
)
from pq_hsa.kernels import (
    batched_topk_merge_triton,
    block_topk_logits_multihead_triton,
    list_exp_sums_sorted_multihead_triton,
    list_stats_sorted_multihead_triton,
    score_packed_4bit_lut_multihead_list_bias,
)
from benchmarks.vllm_backend.paged_kv_fa import (
    gather_flashattn_kv_at_indices,
    gather_flashattn_kv_range,
)


def _paged_kv_enabled() -> bool:
    """Opt-in: read exact-gather / flush-training K/V straight from
    vLLM's paged KV cache instead of the sidecar's own resident copy, for
    the subset of readers not gated by the frozen pq_hsa CUDA kernel.
    Default OFF -- the byte-identical default path is
    unaffected either way.
    """
    return os.environ.get("PQ_HSA_PAGED_KV", "0") == "1"


def _paged_restore_enabled() -> bool:
    """Keep prefix checkpoint/restore alive in ring (PAGED_ATTEND) mode."""
    return os.environ.get("PQ_HSA_PAGED_RESTORE", "0") == "1"


def _prefix_extend_enabled() -> bool:
    """(opt-in PQ_HSA_PREFIX_EXTEND=1): allow a checkpointed prefix to be
    RESTORED AND EXTENDED in place when a new request's prompt strictly extends
    the checkpointed prompt (multi-turn / agentic loads), instead of rebuilding
    the whole sidecar index over the grown context. Default OFF."""
    return os.environ.get("PQ_HSA_PREFIX_EXTEND", "0") == "1"


def _prefix_extend_max_frac() -> float:
    """If delta_tokens / new_length exceeds this fraction, prefer a full
    rebuild (the old codebooks would be extended by a large batch through the
    online-EMA flush path, which is an approximation). Default 0.5."""
    try:
        v = float(os.environ.get("PQ_HSA_PREFIX_EXTEND_MAX_FRAC", "0.5"))
    except ValueError:
        v = 0.5
    return v


def _paged_attend_enabled() -> bool:
    """Opt-in: the PRODUCTION CUDA exact-attend epilogue
    (PQ_HSA_CUDA_ATTEND_ONLY=1, i.e. cuda_attend_only()/pq_exact_attend, the
    frozen-kernel-family path _cg_forward_static captures into a CUDA graph)
    reads its exact-gather K/V rows straight out of vLLM's paged kv_cache via
    block_table (pq_exact_attend_paged), instead of the sidecar's flat
    [H*CAP,D] shared_base_keys/vals duplicate. Default OFF -- with the flag
    off, pq_exact_attend_kernel and cuda_attend_only() are byte-for-byte
    unchanged and still the only code path reached.
    """
    return os.environ.get("PQ_HSA_PAGED_ATTEND", "0") == "1"


def _t12_mark(name: str) -> None:
    """CUDA-event mark. No-op unless PQ_T12_EVENTS=1. Timing only."""
    if os.environ.get("PQ_T12_EVENTS", "0") != "1":
        return
    from benchmarks.cuda_event_log import mark as _mark

    _mark(name)


def _pq_hsa_block_topk() -> bool:
    """Exact block top-k + merge. Default OFF until 128K A/B is green."""
    return os.environ.get("PQ_HSA_BLOCK_TOPK", "0") == "1"


def _retrieval_exact_topk(approx_logits: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    if _pq_hsa_block_topk():
        from pq_hsa.kernels.triton_lut_scan_h20 import exact_block_merge_topk

        return exact_block_merge_topk(approx_logits, topk, sorted=False)
    return torch.topk(approx_logits, k=topk, dim=2, sorted=False)


def _pq_hsa_fp16_lut() -> bool:
    """Fp16 LUT/list GEMM. Default OFF — A/B showed a regression vs fp32 einsum.

    PQ_HSA_FP16_LUT=1 enables the new path; PQ_HSA_FP16_LUT=0 (default) is the fp32 path.
    """
    return os.environ.get("PQ_HSA_FP16_LUT", "0") == "1"


def _lut_and_list_scores(
    q_sub: torch.Tensor,
    codebooks: torch.Tensor,
    q_pq: torch.Tensor,
    coarse: torch.Tensor,
    scale: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build LUT [H,G,M,K] and coarse list scores [H,G,L].

    Default (PQ_HSA_FP16_LUT=1): multiply in the query dtype (fp16 on this stack)
    so we skip the extra fp32 GEMM. Full-region logits stay fp32 (overflow).
    """
    if _pq_hsa_fp16_lut():
        cb = codebooks if codebooks.dtype == q_sub.dtype else codebooks.to(dtype=q_sub.dtype)
        cs = coarse if coarse.dtype == q_pq.dtype else coarse.to(dtype=q_pq.dtype)
        lut = (torch.einsum("hgmd,hmkd->hgmk", q_sub, cb) * scale).to(dtype)
        list_scores = (torch.einsum("hgd,hld->hgl", q_pq, cs) * scale).to(dtype)
        return lut, list_scores
    lut = (torch.einsum("hgmd,hmkd->hgmk", q_sub.float(), codebooks.float()) * scale).to(dtype)
    list_scores = (torch.einsum("hgd,hld->hgl", q_pq.float(), coarse.float()) * scale).to(dtype)
    return lut, list_scores


SWEEP_INDEX_KEYS = {
    "num_lists",
    "nprobe",
    "num_subspaces",
    "num_bits",
    "residual",
    "coarse_max_iter",
    "pq_max_iter",
    "pack_codes",
    "topk_block_size",
    "kernel_backend",
    "rotation",
    "direction_normalize",
    "online_codebook_lr",
    "seed",
}
SWEEP_ATTENTION_KEYS = {
    "sink_tokens",
    "local_window",
    "retrieval_topk",
    "retrieval_top_fraction",
    "retrieval_top_p",
    "retrieval_top_p_scope",
    "nprobe",
    "candidate_budget",
    "exact_rerank",
    "scale",
    "mode",
    "hybrid_value_mode",
    "hybrid_topk_source",
    "hybrid_denominator_source",
    "index_update_interval",
    "index_update_strategy",
    "codebook_refresh_interval",
    "kv_storage",
    "pin_offloaded_kv",
    "prefetch_full_kv",
    "collect_attention_details",
    "profile_attention_components",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end long-context PPL/accuracy sweep for tunable IVF-PQ sparse attention. "
            "Dense prefill is run once, then each PQ config is evaluated by teacher-forced decode."
        )
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--context-tokens", type=int, default=32752)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--prefill-chunk-size", type=int, default=None)
    parser.add_argument("--cpu-cache-snapshot", action="store_true")
    parser.add_argument("--text-lines", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="single", choices=("single", "auto"))
    parser.add_argument("--max-memory-per-gpu", default=None)
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--grid-json",
        default=None,
        help=(
            "Optional JSON file with an array of configs. Each item may use flat keys or "
            "{'index': {...}, 'attention': {...}, 'name': '...'}."
        ),
    )
    parser.add_argument("--max-sweep-configs", type=int, default=None)

    parser.add_argument("--num-lists", default="512")
    parser.add_argument("--nprobe", default="128")
    parser.add_argument("--subspaces", default="8")
    parser.add_argument("--bits", default="4")
    parser.add_argument("--coarse-iter", default="4")
    parser.add_argument("--pq-iter", default="4")
    parser.add_argument("--pack-codes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kernel-backend", default="torch")
    parser.add_argument("--topk-block-size", default="256")
    parser.add_argument("--rotation", default="none")
    parser.add_argument("--direction-normalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--online-codebook-lr", default="0.05")

    parser.add_argument("--sink", default="4")
    parser.add_argument("--local-window", default="128")
    parser.add_argument("--retrieval-topk", default="100")
    parser.add_argument("--retrieval-top-fraction", default="0.10")
    parser.add_argument("--retrieval-top-p", default="none")
    parser.add_argument("--retrieval-top-p-scope", default="retrieval")
    parser.add_argument("--candidate-budget", default="8192")
    parser.add_argument("--exact-rerank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hybrid-value-mode", default="denominator")
    parser.add_argument("--hybrid-topk-source", default="ivf_candidates")
    parser.add_argument("--hybrid-denominator-source", default="all_pq")
    parser.add_argument("--index-update-strategy", default="online")
    parser.add_argument("--index-update-interval", default="64")
    parser.add_argument("--codebook-refresh-interval", default="none")
    parser.add_argument(
        "--apply-pending-index-updates",
        action="store_true",
        help=(
            "After each sparse decode step, flush deferred IVF-PQ migrations "
            "(apply_pending_index_update). Required for a long-generation "
            "index-interval sweep; default off so existing 64/128-step paper "
            "runs stay unchanged."
        ),
    )
    parser.add_argument("--kv-storage", default="device", choices=("device", "cpu"))
    parser.add_argument(
        "--dense-kv-storage",
        default="device",
        choices=("device", "cpu"),
        help=(
            "Where the HuggingFace SDPA cache lives during dense decode. "
            "'cpu' parks each layer on pinned host memory and stages the full "
            "K/V to that layer's GPU around the forward (naive offload baseline)."
        ),
    )
    parser.add_argument("--pin-offloaded-kv", action="store_true")
    parser.add_argument("--prefetch-full-kv", action="store_true")
    parser.add_argument(
        "--collect-attention-details",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep per-stream attention logits/weights/indices in adapter outputs. "
            "Disable for faster E2E benchmarking when only aggregate stats are needed."
        ),
    )
    parser.add_argument(
        "--profile-attention-components",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Synchronize and time major sparse-attention subcomponents. "
            "Use only for diagnosis because it adds CUDA synchronization overhead."
        ),
    )
    parser.add_argument(
        "--share-gqa-kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Build one PQ cache per KV head for grouped-query attention, instead of "
            "repeating KV to every query head. This cuts Qwen3-8B cache/index work by 4x."
        ),
    )
    parser.add_argument(
        "--share-layer-pq-codebook",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Experimental GQA mode: train one IVF/PQ codebook over all KV heads in a layer, "
            "then build per-head list assignments and PQ codes from that shared codebook."
        ),
    )
    parser.add_argument(
        "--prebuild-sparse-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Build sparse PQ adapters from the prefill KV cache before timed sparse decode. "
            "This reports index build time separately and measures steady-state decode speed."
        ),
    )
    parser.add_argument(
        "--sparse-warmup-steps",
        type=int,
        default=0,
        help=(
            "Run this many untimed sparse adapter forwards after prebuild and before timed decode. "
            "This is useful for compiling Triton kernels outside the measured decode window."
        ),
    )

    parser.add_argument("--include-per-layer", action="store_true")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--jsonl-output", default=None)
    parser.add_argument("--csv-output", default=None)

    # Feature 1: multi-window evaluation
    parser.add_argument(
        "--eval-window-offsets",
        default=None,
        help=(
            "Comma-separated list of token offsets for multi-window evaluation "
            "(e.g. '0,140000,280000'). Each offset defines an independent window of "
            "context+decode tokens. All state (KV cache, PQ index, CUDA graph) is "
            "rebuilt from scratch for every window."
        ),
    )

    # Feature 2: PassKey needle-in-a-haystack task
    parser.add_argument(
        "--task",
        default="ppl",
        choices=("ppl", "needle"),
        help="Evaluation task: 'ppl' (default, perplexity) or 'needle' (PassKey retrieval).",
    )
    parser.add_argument(
        "--needle-depths",
        default="0.1,0.5,0.9",
        help="Comma-separated relative depths (0-1) at which to insert the needle.",
    )
    parser.add_argument(
        "--needle-repeats",
        type=int,
        default=3,
        help="Number of repeats per depth (each uses a different random 5-digit key).",
    )

    # Feature 3: sink+local baseline
    parser.add_argument(
        "--sparse-baseline",
        default="none",
        choices=("none", "sink_local"),
        help=(
            "'none' (default) uses the PQ-HSA adapter; "
            "'sink_local' replaces it with a StreamingLLM-style adapter that keeps "
            "only sink + local-window KV for exact SDPA attention."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sparse_warmup_steps < 0:
        raise ValueError("--sparse-warmup-steps must be non-negative")
    torch.manual_seed(args.seed)

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)

    # Parse multi-window offsets (Feature 1).
    eval_window_offsets: list[int] | None = None
    if args.eval_window_offsets is not None:
        eval_window_offsets = [
            int(x.strip()) for x in args.eval_window_offsets.split(",") if x.strip()
        ]

    # Load tokens — for ppl mode we load enough for all windows; for needle mode
    # tokens are synthesised per-trial and full_tokens is not used.
    if args.task == "ppl":
        window_size = args.context_tokens + args.decode_steps + 1
        if eval_window_offsets is not None:
            # Need enough tokens to cover the furthest window.
            token_count_needed = max(off + window_size for off in eval_window_offsets)
        else:
            token_count_needed = window_size
        full_tokens = _load_tokens(
            tokenizer,
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            split=args.split,
            text_lines=args.text_lines,
            token_count=token_count_needed,
        )
    else:
        # needle mode: tokens are synthesised later; full_tokens unused.
        full_tokens = None

    model_kwargs: dict[str, Any] = {
        "local_files_only": args.local_files_only,
        "torch_dtype": dtype,
        "attn_implementation": args.attn_implementation,
    }
    if device.type == "cuda":
        if args.device_map == "auto":
            model_kwargs["device_map"] = "auto"
            if args.max_memory_per_gpu is not None:
                model_kwargs["max_memory"] = {
                    idx: args.max_memory_per_gpu for idx in range(torch.cuda.device_count())
                }
        else:
            model_kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).eval()
    if args.device_map == "single":
        model.to(device)
    input_device = _model_input_device(model, fallback=device)
    if full_tokens is not None:
        full_tokens = full_tokens.to(input_device)

    max_pos = getattr(model.config, "max_position_embeddings", None)
    if args.task == "ppl" and max_pos is not None:
        required = args.context_tokens + args.decode_steps + 1
        if required > int(max_pos):
            raise ValueError(
                f"requested {required} tokens but model max_position_embeddings={max_pos}"
            )

    configs = _sweep_configs(args)
    if args.max_sweep_configs is not None:
        configs = configs[: args.max_sweep_configs]
    if not configs:
        raise ValueError("empty PQ config sweep")

    print(
        json.dumps(
            {
                "event": "start",
                "model": args.model,
                "context_tokens": args.context_tokens,
                "decode_steps": args.decode_steps,
                "sweep_configs": len(configs),
                "task": args.task,
                "sparse_baseline": args.sparse_baseline,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Feature 2: needle task — run per config and return early
    # -------------------------------------------------------------------------
    if args.task == "needle":
        _run_needle_task(model, tokenizer, args, configs, input_device, dtype)
        return

    # -------------------------------------------------------------------------
    # PPL task (original path, extended with multi-window + sink_local baseline)
    # -------------------------------------------------------------------------

    # Determine which windows to evaluate.
    window_size = args.context_tokens + args.decode_steps + 1
    assert full_tokens is not None
    if eval_window_offsets is not None:
        # Validate all offsets before starting any expensive work.
        for off in eval_window_offsets:
            if off + window_size > full_tokens.shape[1]:
                raise ValueError(
                    f"window offset {off} + window_size {window_size} = "
                    f"{off + window_size} exceeds available tokens "
                    f"({full_tokens.shape[1]})"
                )
        windows = eval_window_offsets
    else:
        windows = [0]

    multi_window = len(windows) > 1

    jsonl_handle = None
    if args.jsonl_output is not None:
        jsonl_handle = open(args.jsonl_output, "w", encoding="utf-8")

    all_window_results: list[dict[str, Any]] = []

    try:
        for window_offset in windows:
            # Slice input_ids for this window (from offset, exactly window_size tokens).
            input_ids = full_tokens[:, window_offset : window_offset + window_size]

            # --- Dense prefill + decode for this window -----------------------
            prefill_cache = _run_prefill(
                model,
                input_ids[:, : args.context_tokens],
                chunk_size=args.prefill_chunk_size,
            )
            if prefill_cache is None:
                raise RuntimeError("dense prefill did not return a KV cache")
            if input_device.type == "cuda":
                torch.cuda.empty_cache()

            cache_snapshot = None
            if args.cpu_cache_snapshot:
                cache_snapshot = _cache_to_cpu_snapshot(prefill_cache)
                del prefill_cache
                if input_device.type == "cuda":
                    torch.cuda.empty_cache()
                dense_cache = _cache_from_cpu_snapshot(cache_snapshot)
            else:
                dense_cache = _clone_cache(prefill_cache)

            dense_metrics = _run_decode(
                model,
                input_ids,
                dense_cache,
                context_tokens=args.context_tokens,
                decode_steps=args.decode_steps,
                kv_storage=args.dense_kv_storage,
            )
            del dense_cache
            if input_device.type == "cuda":
                torch.cuda.empty_cache()

            # --- Sparse sweep for this window ---------------------------------
            results = []
            for config_id, config in enumerate(configs):
                if input_device.type == "cuda":
                    torch.cuda.empty_cache()
                    _reset_peak_memory_all()

                # Feature 3: sink_local baseline skips PQ adapter entirely.
                if args.sparse_baseline == "sink_local":
                    result = _run_sink_local_config(
                        model=model,
                        input_ids=input_ids,
                        args=args,
                        config=config,
                        config_id=config_id,
                        dense_metrics=dense_metrics,
                        cache_snapshot=cache_snapshot if args.cpu_cache_snapshot else None,
                        prefill_cache=prefill_cache if not args.cpu_cache_snapshot else None,
                        window_offset=window_offset,
                    )
                else:
                    result = _run_pq_hsa_config(
                        model=model,
                        input_ids=input_ids,
                        args=args,
                        config=config,
                        config_id=config_id,
                        dense_metrics=dense_metrics,
                        cache_snapshot=cache_snapshot,
                        prefill_cache=prefill_cache if not args.cpu_cache_snapshot else None,
                        window_offset=window_offset,
                    )

                results.append(result)
                if jsonl_handle is not None:
                    row = dict(result)
                    if multi_window:
                        row["window_offset"] = window_offset
                    jsonl_handle.write(json.dumps(row, sort_keys=True) + "\n")
                    jsonl_handle.flush()
                print(
                    json.dumps({"event": "config_done", **_summary_row(result)}, sort_keys=True),
                    flush=True,
                )

            # Clean up the dense prefill cache if we held it.
            if not args.cpu_cache_snapshot:
                del prefill_cache
                if input_device.type == "cuda":
                    torch.cuda.empty_cache()

            all_window_results.append(
                {
                    "window_offset": window_offset,
                    "dense": _strip_logits(dense_metrics),
                    "results": results,
                }
            )

    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()

    # -------------------------------------------------------------------------
    # Build final report
    # -------------------------------------------------------------------------
    # For single-window (no --eval-window-offsets), keep the original report shape.
    if not multi_window:
        win = all_window_results[0]
        report = {
            "model": args.model,
            "dense": win["dense"],
            "sweep": win["results"],
            "summary": [_summary_row(item) for item in win["results"]],
        }
    else:
        # Multi-window: aggregate per-config across windows.
        agg_metrics = ["ppl_ratio", "top1_agreement", "dense_to_sparse_kl",
                       "pq_ms_per_token", "dense_ms_per_token"]
        summary_rows = []
        for config_id, config in enumerate(configs):
            per_window = [
                win["results"][config_id]
                for win in all_window_results
            ]
            base_row = _summary_row(per_window[0])
            agg: dict[str, Any] = {}
            for metric in agg_metrics:
                values = [_extract_agg_metric(r, metric) for r in per_window]
                vals_f = [v for v in values if v is not None]
                if vals_f:
                    mean_v = sum(vals_f) / len(vals_f)
                    # sample std (Bessel-corrected): divide by N-1; 0.0 when N=1
                    n = len(vals_f)
                    variance = sum((v - mean_v) ** 2 for v in vals_f) / max(1, n - 1) if n > 1 else 0.0
                    std_v = variance ** 0.5
                else:
                    mean_v = float("nan")
                    std_v = float("nan")
                agg[f"{metric}_mean"] = mean_v
                agg[f"{metric}_std"] = std_v
            summary_rows.append({**base_row, **agg})

        report = {
            "model": args.model,
            "windows": all_window_results,
            "summary": summary_rows,
        }

    if args.json_output is not None:
        with open(args.json_output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    if args.csv_output is not None:
        _write_csv(args.csv_output, report["summary"])

    print(json.dumps(report, indent=2, sort_keys=True))


def _run_prefill(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    chunk_size: int | None,
) -> DynamicCache | None:
    if chunk_size is None or chunk_size <= 0 or chunk_size >= input_ids.shape[1]:
        with torch.no_grad():
            prefill = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        cache = prefill.past_key_values
        del prefill
        if input_ids.device.type == "cuda":
            torch.cuda.empty_cache()
        return cache

    cache = None
    with torch.no_grad():
        for start in range(0, input_ids.shape[1], chunk_size):
            end = min(input_ids.shape[1], start + chunk_size)
            out = model(
                input_ids=input_ids[:, start:end],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = out.past_key_values
            del out
            if input_ids.device.type == "cuda":
                torch.cuda.empty_cache()
    return cache


def _decoder_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    inner = model
    if hasattr(inner, "model") and hasattr(inner.model, "layers"):
        return list(inner.model.layers)
    if hasattr(inner, "layers"):
        return list(inner.layers)
    raise RuntimeError("could not find decoder layers for dense KV offload")


def _park_dense_cache_on_cpu(cache: DynamicCache) -> list[torch.device]:
    homes: list[torch.device] = []
    for layer in cache.layers:
        if layer.keys is None or layer.values is None:
            raise RuntimeError("dense cache layer is missing keys/values")
        homes.append(torch.device(layer.keys.device))
        parked_k = layer.keys.detach().to("cpu", copy=True)
        parked_v = layer.values.detach().to("cpu", copy=True)
        if torch.cuda.is_available():
            parked_k = parked_k.pin_memory()
            parked_v = parked_v.pin_memory()
        layer.keys = parked_k
        layer.values = parked_v
    return homes


def _install_dense_kv_offload_hooks(
    model: torch.nn.Module,
    cache_holder: dict[str, DynamicCache],
    homes: list[torch.device],
) -> list[Any]:
    """Stage full-layer K/V to the layer GPU, then park the updated cache on CPU."""
    layers = _decoder_layers(model)
    if len(layers) != len(homes):
        raise RuntimeError(
            f"decoder depth {len(layers)} != dense cache layers {len(homes)}"
        )
    handles: list[Any] = []

    def _pre(layer_idx: int):
        def hook(_module, _inputs):
            cache = cache_holder["cache"]
            layer = cache.layers[layer_idx]
            device = homes[layer_idx]
            if layer.keys.device != device:
                layer.keys = layer.keys.to(device, non_blocking=True)
            if layer.values.device != device:
                layer.values = layer.values.to(device, non_blocking=True)
        return hook

    def _post(layer_idx: int):
        def hook(_module, _inputs, _output):
            cache = cache_holder["cache"]
            layer = cache.layers[layer_idx]
            parked_k = layer.keys.detach().to("cpu", copy=True)
            parked_v = layer.values.detach().to("cpu", copy=True)
            if torch.cuda.is_available():
                parked_k = parked_k.pin_memory()
                parked_v = parked_v.pin_memory()
            layer.keys = parked_k
            layer.values = parked_v
        return hook

    for idx, module in enumerate(layers):
        handles.append(module.register_forward_pre_hook(_pre(idx)))
        handles.append(module.register_forward_hook(_post(idx)))
    return handles


def _apply_pending_index_updates(patched: list[torch.nn.Module]) -> dict[str, float]:
    """Flush deferred index migrations on every patched layer. Returns counts/ms."""
    start = time.perf_counter()
    applied = 0
    for attn in patched:
        adapter = getattr(attn, "_e2e_pq_adapter", None)
        if adapter is None or not hasattr(adapter, "apply_pending_index_update"):
            continue
        applied += int(adapter.apply_pending_index_update() or 0)
    return {
        "applied": float(applied),
        "ms": (time.perf_counter() - start) * 1000.0,
    }


def _run_decode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    cache: DynamicCache,
    *,
    context_tokens: int,
    decode_steps: int,
    kv_storage: str = "device",
    after_step: Any | None = None,
) -> dict[str, Any]:
    logits_rows = []
    losses = []
    top1 = []
    input_device = _model_input_device(model, fallback=input_ids.device)
    cache_holder = {"cache": cache}
    hook_handles: list[Any] = []
    if kv_storage == "cpu":
        homes = _park_dense_cache_on_cpu(cache)
        hook_handles = _install_dense_kv_offload_hooks(model, cache_holder, homes)
    n_updates = 0
    update_ms = 0.0
    wmg = None
    wmg_steps = 0
    if os.environ.get("PQ_WHOLE_MODEL_GRAPH", "1") == "1":
        from pq_hsa.attention.whole_model_decode import try_build_whole_model_graph

        _cap_token = input_ids[:, context_tokens : context_tokens + 1].to(input_device)
        wmg = try_build_whole_model_graph(model, _cap_token, context_tokens)
    _reset_peak_memory_all()
    start = _time_start(input_device)
    try:
        with torch.no_grad():
            for step in range(decode_steps):
                pos = context_tokens + step
                token = input_ids[:, pos : pos + 1].to(input_device)
                if wmg is None and os.environ.get("PQ_WHOLE_MODEL_GRAPH", "1") == "1":
                    from pq_hsa.attention.whole_model_decode import try_build_whole_model_graph

                    wmg = try_build_whole_model_graph(model, token, pos)
                if wmg is not None and wmg.enabled:
                    logits = wmg.step(token, pos)[:, -1, :]
                    wmg_steps += 1
                else:
                    out = model(
                        input_ids=token,
                        past_key_values=cache_holder["cache"],
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    cache_holder["cache"] = out.past_key_values
                    logits = out.logits[:, -1, :]
                    del out
                target = input_ids[:, pos + 1].to(logits.device)
                logits_rows.append(logits.detach().float().cpu())
                losses.append(F.cross_entropy(logits.float(), target, reduction="none").detach().cpu())
                top1.append((logits.argmax(dim=-1) == target).detach().float().cpu())
                if after_step is not None:
                    stats = after_step()
                    n_updates += int(stats.get("applied", 0))
                    update_ms += float(stats.get("ms", 0.0))
                    if wmg is not None and int(stats.get("applied", 0)) > 0:
                        wmg.invalidate()
                        wmg = None
    finally:
        for handle in hook_handles:
            handle.remove()
    elapsed_ms = _time_stop(input_device, start)
    loss = torch.cat(losses).mean()
    return {
        "nll": float(loss.item()),
        "ppl": float(torch.exp(loss).item()),
        "top1_accuracy": float(torch.cat(top1).mean().item()),
        "eval_ms": elapsed_ms,
        "ms_per_token": elapsed_ms / max(1, decode_steps),
        "max_memory_gb": _max_memory_gb_any(),
        "kv_storage": kv_storage,
        "index_updates_applied": n_updates,
        "index_update_ms": update_ms,
        "logits": torch.cat(logits_rows, dim=0),
        "whole_model_graph": bool(wmg is not None and wmg.enabled) or wmg_steps > 0,
        "whole_model_graph_steps": wmg_steps,
    }


def _extract_agg_metric(result: dict[str, Any], metric: str) -> float | None:
    """Extract a scalar metric from a result dict for aggregation."""
    if metric == "ppl_ratio":
        return result.get("comparison", {}).get("ppl_ratio")
    if metric == "top1_agreement":
        return result.get("comparison", {}).get("top1_agreement")
    if metric == "dense_to_sparse_kl":
        return result.get("comparison", {}).get("dense_to_sparse_kl")
    if metric == "pq_ms_per_token":
        return result.get("pq_hsa", {}).get("ms_per_token")
    if metric == "dense_ms_per_token":
        return result.get("dense", {}).get("ms_per_token")
    return None


def _run_pq_hsa_config(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    args: argparse.Namespace,
    config: dict[str, Any],
    config_id: int,
    dense_metrics: dict[str, Any],
    cache_snapshot: Any | None,
    prefill_cache: Any | None,
    window_offset: int,
) -> dict[str, Any]:
    """Run one PQ-HSA sparse config for the current window. Mirrors the original main() loop body."""
    input_device = _model_input_device(model, fallback=input_ids.device)
    index_config = IVFPQConfig(**config["index"])
    attention_config = SparseAttentionConfig(**config["attention"])
    patched = _install_pq_sparse_patch(
        model,
        index_config,
        attention_config,
        share_gqa_kv_cache=args.share_gqa_kv_cache,
        share_layer_pq_codebook=args.share_layer_pq_codebook,
    )

    if cache_snapshot is not None:
        sparse_cache = _cache_from_cpu_snapshot(cache_snapshot)
    else:
        assert prefill_cache is not None
        sparse_cache = _clone_cache(prefill_cache)

    print(
        json.dumps(
            {
                "event": "config_start",
                "config_id": config_id,
                "name": config["name"],
                "index": config["index"],
                "attention": config["attention"],
                "share_gqa_kv_cache": args.share_gqa_kv_cache,
                "share_layer_pq_codebook": args.share_layer_pq_codebook,
                "window_offset": window_offset,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    sparse_prebuild_ms = 0.0
    sparse_warmup_ms = 0.0
    try:
        if args.prebuild_sparse_cache:
            sparse_prebuild_ms = _prebuild_sparse_adapters(
                patched,
                sparse_cache,
                share_gqa_kv_cache=args.share_gqa_kv_cache,
            )
            _capture_sparse_adapter_prebuild_profiles(patched)
            _reset_sparse_adapter_runtime_stats(patched)
            if args.sparse_warmup_steps > 0:
                sparse_warmup_ms = _warmup_sparse_adapters(
                    patched,
                    steps=args.sparse_warmup_steps,
                )
            if input_device.type == "cuda":
                torch.cuda.empty_cache()
        after_step = None
        if args.apply_pending_index_updates:
            after_step = lambda: _apply_pending_index_updates(patched)
        sparse_metrics = _run_decode(
            model,
            input_ids,
            sparse_cache,
            context_tokens=args.context_tokens,
            decode_steps=args.decode_steps,
            after_step=after_step,
        )
        attention_stats = _collect_patch_metrics(
            patched,
            model,
            include_per_layer=args.include_per_layer,
        )
    finally:
        del sparse_cache
        _restore_pq_sparse_patch(patched)
        if input_device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "config_id": config_id,
        "name": config["name"],
        "model": args.model,
        "window_offset": window_offset,
        "dataset": {
            "name": args.dataset,
            "config": args.dataset_config,
            "split": args.split,
            "context_tokens": args.context_tokens,
            "decode_steps": args.decode_steps,
            "prefill_chunk_size": args.prefill_chunk_size,
            "cpu_cache_snapshot": args.cpu_cache_snapshot,
            "kv_storage": args.kv_storage,
            "dense_kv_storage": args.dense_kv_storage,
            "pin_offloaded_kv": args.pin_offloaded_kv,
            "apply_pending_index_updates": args.apply_pending_index_updates,
            "share_gqa_kv_cache": args.share_gqa_kv_cache,
            "share_layer_pq_codebook": args.share_layer_pq_codebook,
            "prebuild_sparse_cache": args.prebuild_sparse_cache,
        },
        "dense": _strip_logits(dense_metrics),
        "pq_hsa": {
            **_strip_logits(sparse_metrics),
            "prebuild_ms": sparse_prebuild_ms,
            "warmup_ms": sparse_warmup_ms,
            "total_with_prebuild_ms": sparse_metrics["eval_ms"] + sparse_prebuild_ms,
            "ms_per_token_with_prebuild": (
                (sparse_metrics["eval_ms"] + sparse_prebuild_ms)
                / max(1, args.decode_steps)
            ),
            "total_with_prebuild_and_warmup_ms": (
                sparse_metrics["eval_ms"] + sparse_prebuild_ms + sparse_warmup_ms
            ),
            "ms_per_token_with_prebuild_and_warmup": (
                (sparse_metrics["eval_ms"] + sparse_prebuild_ms + sparse_warmup_ms)
                / max(1, args.decode_steps)
            ),
            "max_memory_gb": _max_memory_gb_any(),
            "index_config": asdict(index_config),
            "attention_config": asdict(attention_config),
            "attention_stats": attention_stats,
        },
        "comparison": _compare_decode_metrics(dense_metrics, sparse_metrics),
    }


# ---------------------------------------------------------------------------
# Feature 3: SinkLocal adapter (StreamingLLM-style baseline)
# ---------------------------------------------------------------------------

class SinkLocalDecodeAdapter:
    """Minimal StreamingLLM-style adapter: keeps sink + recent local-window KV.

    Matches the build_cache / append / forward / decode_step interface of
    GQAIVFPQDecodeAttentionAdapter so it can be dropped in at the same call-sites.
    All computation is exact SDPA (scaled dot-product attention via einsum).
    """

    def __init__(
        self,
        sink_tokens: int,
        local_window: int,
        num_query_heads: int,
        num_key_value_heads: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if num_query_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_query_heads must be divisible by num_key_value_heads, "
                f"got {num_query_heads} and {num_key_value_heads}"
            )
        self.sink_tokens = max(0, int(sink_tokens))
        self.local_window = max(1, int(local_window))
        self.num_query_heads = int(num_query_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.num_key_value_groups = self.num_query_heads // self.num_key_value_heads
        self.dtype = dtype
        self.device = device
        # Storage: [H, cap, D] ring buffers for local, [H, sink, D] for sink.
        self._sink_k: torch.Tensor | None = None
        self._sink_v: torch.Tensor | None = None
        self._local_k: torch.Tensor | None = None
        self._local_v: torch.Tensor | None = None
        self._local_write_pos: int = 0   # next write slot (ring)
        self._local_filled: int = 0      # how many valid local slots
        self._H: int = 0
        self._D_k: int = 0
        self._D_v: int = 0

    # Expose a .streams property for compatibility with _collect_adapter_memory
    @property
    def streams(self):
        return ()

    def build_cache(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> "SinkLocalDecodeAdapter":
        """Accept [batch, kv_heads, seq, D] tensors and snapshot sink + tail."""
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError("keys and values must be [batch, kv_heads, seq, dim]")
        # Flatten batch into heads.
        batch, H, seq, D_k = keys.shape
        D_v = values.shape[-1]
        self._H = batch * H
        self._D_k = D_k
        self._D_v = D_v
        flat_k = keys.reshape(self._H, seq, D_k).to(self.device, self.dtype)
        flat_v = values.reshape(self._H, seq, D_v).to(self.device, self.dtype)

        # Capture sink tokens.
        sink_end = min(self.sink_tokens, seq)
        if sink_end > 0:
            self._sink_k = flat_k[:, :sink_end, :].detach().clone()
            self._sink_v = flat_v[:, :sink_end, :].detach().clone()
        else:
            self._sink_k = torch.empty(self._H, 0, D_k, dtype=self.dtype, device=self.device)
            self._sink_v = torch.empty(self._H, 0, D_v, dtype=self.dtype, device=self.device)

        # Allocate ring buffer for local window.
        cap = self.local_window
        self._local_k = torch.empty(self._H, cap, D_k, dtype=self.dtype, device=self.device)
        self._local_v = torch.empty(self._H, cap, D_v, dtype=self.dtype, device=self.device)
        # Fill ring with the tail of the prefill (up to local_window tokens).
        tail_start = max(sink_end, seq - self.local_window)
        tail = flat_k[:, tail_start:, :]
        n_tail = tail.shape[1]
        if n_tail > 0:
            self._local_k[:, :n_tail, :] = tail
            self._local_v[:, :n_tail, :] = flat_v[:, tail_start:, :]
        self._local_filled = min(n_tail, cap)
        self._local_write_pos = self._local_filled % cap
        return self

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Append one new [batch, kv_heads, D] token to the ring buffer."""
        assert self._local_k is not None
        flat_k = keys.reshape(self._H, self._D_k).to(self.device, self.dtype)
        flat_v = values.reshape(self._H, self._D_v).to(self.device, self.dtype)
        pos = self._local_write_pos
        self._local_k[:, pos, :] = flat_k
        self._local_v[:, pos, :] = flat_v
        self._local_write_pos = (pos + 1) % self.local_window
        self._local_filled = min(self._local_filled + 1, self.local_window)

    def forward(self, queries: torch.Tensor) -> "_SinkLocalOutput":
        """Compute scaled dot-product attention over sink + local KV.

        queries: [batch, query_heads, 1, D] or [batch, query_heads, D]
        """
        assert self._sink_k is not None and self._local_k is not None
        squeeze = queries.ndim == 3
        if squeeze:
            queries = queries.unsqueeze(-2)
        q = queries.to(self.device, self.dtype)
        batch, qh, qlen, Dk = q.shape
        assert qlen == 1, "SinkLocalDecodeAdapter only supports single-step decode"

        # Reconstruct ordered local window from ring buffer.
        cap = self.local_window
        filled = self._local_filled
        if filled == cap:
            # Fully filled: tokens in order from write_pos to write_pos-1.
            pos = self._local_write_pos
            local_k = torch.cat(
                [self._local_k[:, pos:, :], self._local_k[:, :pos, :]], dim=1
            )
            local_v = torch.cat(
                [self._local_v[:, pos:, :], self._local_v[:, :pos, :]], dim=1
            )
        else:
            local_k = self._local_k[:, :filled, :]
            local_v = self._local_v[:, :filled, :]

        # Concatenate sink + local: [H, S, D]
        kv_k = torch.cat([self._sink_k, local_k], dim=1)  # [H, S_total, D_k]
        kv_v = torch.cat([self._sink_v, local_v], dim=1)  # [H, S_total, D_v]
        S = kv_k.shape[1]

        # Reshape for GQA: q [H, G, 1, D] where H=kv_heads, G=groups
        H = self._H  # = batch * kv_heads
        G = self.num_key_value_groups
        scale = 1.0 / math.sqrt(Dk)
        q_r = q.reshape(H, G, 1, Dk)
        k_r = kv_k.unsqueeze(1).expand(H, G, S, Dk)
        v_r = kv_v.unsqueeze(1).expand(H, G, S, self._D_v)

        # [H, G, 1, S]
        logits = torch.einsum("hgqd,hgsd->hgqs", q_r.float(), k_r.float()) * scale
        weights = torch.softmax(logits, dim=-1)  # float32
        # [H, G, 1, Dv]
        ctx = torch.einsum("hgqs,hgsd->hgqd", weights, v_r.float()).to(self.dtype)
        # Reshape to [batch, qh, 1, Dv]
        ctx_out = ctx.reshape(batch, qh, 1, self._D_v)
        if squeeze:
            ctx_out = ctx_out.squeeze(-2)
        return _SinkLocalOutput(context=ctx_out)

    def decode_step(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
        *,
        append_kv: bool = True,
    ) -> "_SinkLocalOutput":
        if append_kv:
            if keys is None or values is None:
                raise ValueError("keys and values are required when append_kv=True")
            self.append(keys, values)
        return self.forward(queries)


@dataclass(slots=True)
class _SinkLocalOutput:
    """Minimal output dataclass returned by SinkLocalDecodeAdapter.forward."""
    context: torch.Tensor

    # Satisfy _update_attention_stats which iterates .per_query
    @property
    def per_query(self):
        return ()


def _run_sink_local_config(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    args: argparse.Namespace,
    config: dict[str, Any],
    config_id: int,
    dense_metrics: dict[str, Any],
    cache_snapshot: Any | None,
    prefill_cache: Any | None,
    window_offset: int,
) -> dict[str, Any]:
    """Run one sweep config with SinkLocal baseline (no PQ-HSA)."""
    input_device = _model_input_device(model, fallback=input_ids.device)
    attention_config = config["attention"]
    sink_tokens = int(attention_config.get("sink_tokens", 4))
    local_window = int(attention_config.get("local_window", 128))

    # Restore the prefill cache to feed to the model's standard forward path.
    if cache_snapshot is not None:
        sparse_cache = _cache_from_cpu_snapshot(cache_snapshot)
    else:
        assert prefill_cache is not None
        sparse_cache = _clone_cache(prefill_cache)

    print(
        json.dumps(
            {
                "event": "config_start",
                "config_id": config_id,
                "name": config["name"],
                "sparse_baseline": "sink_local",
                "sink_tokens": sink_tokens,
                "local_window": local_window,
                "window_offset": window_offset,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    try:
        sparse_metrics = _run_sink_local_decode(
            model,
            input_ids,
            sparse_cache,
            context_tokens=args.context_tokens,
            decode_steps=args.decode_steps,
            sink_tokens=sink_tokens,
            local_window=local_window,
        )
    finally:
        del sparse_cache
        if input_device.type == "cuda":
            torch.cuda.empty_cache()

    # Build a result dict with the same shape as _run_pq_hsa_config so
    # _summary_row and comparison helpers work unmodified.
    index_config = IVFPQConfig(**config["index"])
    attention_config_obj = SparseAttentionConfig(**config["attention"])
    dummy_attention_stats = _dummy_attention_stats_for_sink_local()
    return {
        "config_id": config_id,
        "name": config["name"],
        "model": args.model,
        "window_offset": window_offset,
        "sparse_baseline": "sink_local",
        "dataset": {
            "name": args.dataset,
            "config": args.dataset_config,
            "split": args.split,
            "context_tokens": args.context_tokens,
            "decode_steps": args.decode_steps,
            "prefill_chunk_size": args.prefill_chunk_size,
            "cpu_cache_snapshot": args.cpu_cache_snapshot,
            "kv_storage": args.kv_storage,
            "dense_kv_storage": args.dense_kv_storage,
            "pin_offloaded_kv": args.pin_offloaded_kv,
            "apply_pending_index_updates": args.apply_pending_index_updates,
            "share_gqa_kv_cache": args.share_gqa_kv_cache,
            "share_layer_pq_codebook": args.share_layer_pq_codebook,
            "prebuild_sparse_cache": False,
        },
        "dense": _strip_logits(dense_metrics),
        "pq_hsa": {
            **_strip_logits(sparse_metrics),
            "prebuild_ms": 0.0,
            "warmup_ms": 0.0,
            "total_with_prebuild_ms": sparse_metrics["eval_ms"],
            "ms_per_token_with_prebuild": sparse_metrics["ms_per_token"],
            "total_with_prebuild_and_warmup_ms": sparse_metrics["eval_ms"],
            "ms_per_token_with_prebuild_and_warmup": sparse_metrics["ms_per_token"],
            "max_memory_gb": _max_memory_gb_any(),
            "index_config": asdict(index_config),
            "attention_config": asdict(attention_config_obj),
            "attention_stats": dummy_attention_stats,
        },
        "comparison": _compare_decode_metrics(dense_metrics, sparse_metrics),
    }


def _dummy_attention_stats_for_sink_local() -> dict[str, Any]:
    """Return zeroed attention stats for sink_local (no PQ retrieval)."""
    zero_stats = _finalize_attention_stats(_empty_attention_stats())
    zero_mem = _finalize_memory_stats(_empty_memory_stats())
    return {
        "summary": zero_stats,
        "memory": zero_mem,
        "num_layers": 0,
    }


def _run_sink_local_decode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    cache: DynamicCache,
    *,
    context_tokens: int,
    decode_steps: int,
    sink_tokens: int,
    local_window: int,
) -> dict[str, Any]:
    """Teacher-forced decode using a SinkLocal attention adapter patched into the model.

    The model runs normally (standard HF forward), but attention layers that
    are aware of the SinkLocal adapter do not exist — instead we do the full
    HF SDPA decode and then recompute attention using SinkLocalDecodeAdapter
    for comparison purposes.

    Actually: for correctness we run the model's own standard forward (which
    uses DynamicCache and SDPA internally), and then separately gather the
    sink_local logits by replacing attention with our adapter per-step.

    For the baseline we *don't* patch the model; we simply run standard HF
    decode on the FULL cache (same as dense) then call it the sparse baseline.
    This would make it identical to dense. Instead we do the right thing:
    install a lightweight patch that at each decode step replaces the full
    past KV with sink+local KV via our SinkLocalDecodeAdapter.
    """
    input_device = _model_input_device(model, fallback=input_ids.device)

    # Patch model to use SinkLocalDecodeAdapter at every attention layer.
    patched_layers = _install_sink_local_patch(
        model,
        sink_tokens=sink_tokens,
        local_window=local_window,
    )
    logits_rows = []
    losses = []
    top1 = []
    _reset_peak_memory_all()
    start = _time_start(input_device)
    try:
        with torch.no_grad():
            for step in range(decode_steps):
                pos = context_tokens + step
                out = model(
                    input_ids=input_ids[:, pos : pos + 1].to(input_device),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                cache = out.past_key_values
                logits = out.logits[:, -1, :]
                target = input_ids[:, pos + 1].to(logits.device)
                logits_rows.append(logits.detach().float().cpu())
                losses.append(F.cross_entropy(logits.float(), target, reduction="none").detach().cpu())
                top1.append((logits.argmax(dim=-1) == target).detach().float().cpu())
                del out
    finally:
        _restore_sink_local_patch(patched_layers)

    elapsed_ms = _time_stop(input_device, start)
    loss = torch.cat(losses).mean()
    return {
        "nll": float(loss.item()),
        "ppl": float(torch.exp(loss).item()),
        "top1_accuracy": float(torch.cat(top1).mean().item()),
        "eval_ms": elapsed_ms,
        "ms_per_token": elapsed_ms / max(1, decode_steps),
        "max_memory_gb": _max_memory_gb_any(),
        "logits": torch.cat(logits_rows, dim=0),
    }


def _install_sink_local_patch(
    model: torch.nn.Module,
    *,
    sink_tokens: int,
    local_window: int,
) -> list[torch.nn.Module]:
    """Install a forward patch that routes single-token decode through SinkLocal attention."""
    model_type = getattr(model.config, "model_type", "")
    if model_type in {"qwen3", "qwen3_moe"}:
        forward = _qwen3_sink_local_forward
    elif model_type in {"llama", "qwen2", "mistral"}:
        # See _install_pq_sparse_patch: Qwen2/Mistral attention matches Llama.
        forward = _llama_sink_local_forward
    else:
        raise ValueError(f"unsupported model_type for sink_local patch: {model_type}")

    patched = []
    for layer_idx, decoder_layer in enumerate(model.model.layers):
        attn = decoder_layer.self_attn
        if hasattr(attn, "_sl_original_forward"):
            raise RuntimeError("sink_local patch already installed")
        attn._sl_original_forward = attn.forward
        attn._sl_sink_tokens = sink_tokens
        attn._sl_local_window = local_window
        attn._sl_adapter = None  # built lazily on first decode step
        attn._sl_layer_idx = layer_idx
        attn.forward = MethodType(forward, attn)
        patched.append(attn)
    return patched


def _restore_sink_local_patch(patched: list[torch.nn.Module]) -> None:
    for attn in patched:
        original = getattr(attn, "_sl_original_forward", None)
        if original is not None:
            attn.forward = original
        for name in (
            "_sl_original_forward",
            "_sl_sink_tokens",
            "_sl_local_window",
            "_sl_adapter",
            "_sl_layer_idx",
        ):
            if hasattr(attn, name):
                delattr(attn, name)


def _qwen3_sink_local_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_values: DynamicCache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    del attention_mask, kwargs
    input_shape = hidden_states.shape[:-1]
    if len(input_shape) != 2 or input_shape[1] != 1 or past_key_values is None:
        return self._sl_original_forward(
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
    query_states, key_states = qwen3_apply_rotary_pos_emb(query_states, key_states, cos, sin)
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = past_key_values.update(
        key_states, value_states, self.layer_idx, cache_kwargs,
    )
    return _sink_local_project_output(self, input_shape, query_states, key_states, value_states)


def _llama_sink_local_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_values: DynamicCache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    del attention_mask, kwargs
    input_shape = hidden_states.shape[:-1]
    if len(input_shape) != 2 or input_shape[1] != 1 or past_key_values is None:
        return self._sl_original_forward(
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
    query_states, key_states = llama_apply_rotary_pos_emb(query_states, key_states, cos, sin)
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = past_key_values.update(
        key_states, value_states, self.layer_idx, cache_kwargs,
    )
    return _sink_local_project_output(self, input_shape, query_states, key_states, value_states)


def _sink_local_project_output(
    attn: torch.nn.Module,
    input_shape: torch.Size,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> tuple[torch.Tensor, None]:
    """Compute output using SinkLocalDecodeAdapter."""
    if attn._sl_adapter is None:
        # Build adapter from the full KV now in the cache (post-prefill state).
        batch, kv_heads, seq, D_k = key_states.shape
        dtype = key_states.dtype
        device = key_states.device
        num_query_heads = query_states.shape[1]
        attn._sl_adapter = SinkLocalDecodeAdapter(
            sink_tokens=attn._sl_sink_tokens,
            local_window=attn._sl_local_window,
            num_query_heads=num_query_heads,
            num_key_value_heads=kv_heads,
            dtype=dtype,
            device=device,
        )
        # Feed the full KV history (sink + tail from prefill).
        attn._sl_adapter.build_cache(key_states[:, :, :-1, :], value_states[:, :, :-1, :])

    # Append the just-generated token's KV.
    new_k = key_states[:, :, -1:, :]
    new_v = value_states[:, :, -1:, :]
    adapter_output = attn._sl_adapter.decode_step(
        query_states,
        new_k.squeeze(-2),
        new_v.squeeze(-2),
        append_kv=True,
    )

    context = adapter_output.context
    if context.ndim == 3:
        context = context.unsqueeze(-2)
    attn_output = context.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = attn.o_proj(attn_output)
    return attn_output, None


# ---------------------------------------------------------------------------
# Feature 2: PassKey needle task
# ---------------------------------------------------------------------------

def _build_needle_input_ids(
    tokenizer: Any,
    *,
    context_tokens: int,
    depth: float,
    passkey: str,
    rng: "torch.Generator",
    device: torch.device,
) -> torch.Tensor:
    """Build a synthetic input_ids tensor of exactly context_tokens length.

    Structure (Landmark-attention passkey format):
      - instruction preamble
      - filler text ("The grass is green. ..." repeated)
      - needle sentence with the passkey stated twice, newline-delimited,
        inserted at the given relative depth aligned to a filler-sentence
        boundary (so the needle never lands mid-sentence)
      - prompt suffix at the end ("\nWhat is the pass key? The pass key is")

    The prompt strength matters: the single-statement mid-sentence variant
    put the base model right at the edge of retrieval (correctness flipped
    with a ±2% depth perturbation), which makes accuracy pure noise.
    """
    preamble = (
        "There is an important piece of information hidden inside a lot of "
        "irrelevant text. Find it and memorize it. I will quiz you about the "
        "important information.\n"
    )
    filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    needle = f"\nThe pass key is {passkey}. Remember it. {passkey} is the pass key.\n"
    suffix = "\nWhat is the pass key? The pass key is"

    # Encode components.
    def enc(text: str) -> torch.Tensor:
        return tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]

    preamble_ids = enc(preamble)
    suffix_ids = enc(suffix)
    needle_ids = enc(needle)
    n_preamble = preamble_ids.shape[0]
    n_suffix = suffix_ids.shape[0]
    n_needle = needle_ids.shape[0]

    # Budget for filler tokens.
    filler_budget = context_tokens - n_preamble - n_suffix - n_needle
    if filler_budget <= 0:
        raise ValueError(
            f"context_tokens={context_tokens} too small to fit preamble "
            f"({n_preamble}) + needle ({n_needle}) + suffix ({n_suffix})"
        )

    # Build enough filler.
    filler_ids = enc(filler)
    n_filler_sent = filler_ids.shape[0]
    repeats = (filler_budget // n_filler_sent) + 2
    full_filler = filler_ids.repeat(repeats)[:filler_budget]

    # Insert needle at depth * filler_budget, snapped to a filler-sentence
    # boundary so the needle never lands mid-sentence.
    insert_pos = int(depth * filler_budget)
    insert_pos = (insert_pos // n_filler_sent) * n_filler_sent
    insert_pos = max(0, min(insert_pos, filler_budget))
    before = full_filler[:insert_pos]
    after = full_filler[insert_pos:]
    # Total: preamble + before + needle + after + suffix = context_tokens
    combined = torch.cat(
        [preamble_ids, before, needle_ids, after[: filler_budget - insert_pos], suffix_ids]
    )
    assert combined.shape[0] == context_tokens, (
        f"needle construction bug: got {combined.shape[0]}, expected {context_tokens}"
    )
    return combined.unsqueeze(0).to(device)  # [1, context_tokens]


def _greedy_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    cache: DynamicCache,
    *,
    context_tokens: int,
    gen_steps: int,
) -> str:
    """Generate gen_steps tokens autoregressively from context (true generation, not teacher-forced)."""
    input_device = _model_input_device(model, fallback=input_ids.device)
    current_ids = input_ids[:, context_tokens - 1 : context_tokens].to(input_device)
    generated = []
    with torch.no_grad():
        for _ in range(gen_steps):
            out = model(
                input_ids=current_ids,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(int(next_tok[0, 0]))
            current_ids = next_tok
            del out
    return "".join([str(t) for t in generated])  # raw token IDs as string for debug


def _greedy_generate_text(
    model: torch.nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    cache: DynamicCache,
    *,
    context_tokens: int,
    gen_steps: int,
) -> str:
    """Generate gen_steps tokens and decode to text."""
    input_device = _model_input_device(model, fallback=input_ids.device)
    current_ids = input_ids[:, context_tokens - 1 : context_tokens].to(input_device)
    gen_ids = []
    with torch.no_grad():
        for _ in range(gen_steps):
            out = model(
                input_ids=current_ids,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_ids.append(int(next_tok[0, 0]))
            current_ids = next_tok
            del out
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def _run_needle_task(
    model: torch.nn.Module,
    tokenizer: Any,
    args: argparse.Namespace,
    configs: list[dict[str, Any]],
    input_device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Run needle/PassKey retrieval evaluation and print + save results."""
    depths = [float(x.strip()) for x in args.needle_depths.split(",") if x.strip()]
    repeats = args.needle_repeats
    gen_steps = 8  # fixed: generate 8 tokens to capture the 5-digit passkey

    rng = torch.Generator()
    rng.manual_seed(args.seed)

    all_trials: list[dict[str, Any]] = []

    for config_id, config in enumerate(configs):
        index_config = IVFPQConfig(**config["index"])
        attention_config = SparseAttentionConfig(**config["attention"])

        dense_results_per_depth: dict[float, list[bool]] = {d: [] for d in depths}
        sparse_results_per_depth: dict[float, list[bool]] = {d: [] for d in depths}

        for depth in depths:
            for rep in range(repeats):
                # Random 5-digit passkey.
                passkey = str(int(torch.randint(10000, 99999, (1,), generator=rng).item()))
                # Small random depth perturbation ±2 %.
                perturb = (torch.rand(1, generator=rng).item() - 0.5) * 0.04
                actual_depth = float(max(0.0, min(1.0, depth + perturb)))

                input_ids = _build_needle_input_ids(
                    tokenizer,
                    context_tokens=args.context_tokens,
                    depth=actual_depth,
                    passkey=passkey,
                    rng=rng,
                    device=input_device,
                )

                # --- Dense prefill + greedy generate ---
                dense_prefill_cache = _run_prefill(
                    model,
                    input_ids,
                    chunk_size=args.prefill_chunk_size,
                )
                assert dense_prefill_cache is not None
                if input_device.type == "cuda":
                    torch.cuda.empty_cache()

                dense_gen = _greedy_generate_text(
                    model,
                    tokenizer,
                    input_ids,
                    dense_prefill_cache,
                    context_tokens=args.context_tokens,
                    gen_steps=gen_steps,
                )
                del dense_prefill_cache
                if input_device.type == "cuda":
                    torch.cuda.empty_cache()

                dense_correct = passkey in dense_gen

                # --- Sparse prefill + greedy generate ---
                if args.sparse_baseline == "sink_local":
                    sparse_prefill_cache = _run_prefill(
                        model,
                        input_ids,
                        chunk_size=args.prefill_chunk_size,
                    )
                    assert sparse_prefill_cache is not None
                    if input_device.type == "cuda":
                        torch.cuda.empty_cache()
                    patched_sl = _install_sink_local_patch(
                        model,
                        sink_tokens=int(attention_config.sink_tokens),
                        local_window=int(attention_config.local_window),
                    )
                    try:
                        sparse_gen = _greedy_generate_text(
                            model,
                            tokenizer,
                            input_ids,
                            sparse_prefill_cache,
                            context_tokens=args.context_tokens,
                            gen_steps=gen_steps,
                        )
                    finally:
                        _restore_sink_local_patch(patched_sl)
                    del sparse_prefill_cache
                    if input_device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    # PQ-HSA sparse path.
                    sparse_prefill_cache = _run_prefill(
                        model,
                        input_ids,
                        chunk_size=args.prefill_chunk_size,
                    )
                    assert sparse_prefill_cache is not None
                    if input_device.type == "cuda":
                        torch.cuda.empty_cache()

                    patched_pq = _install_pq_sparse_patch(
                        model,
                        index_config,
                        attention_config,
                        share_gqa_kv_cache=args.share_gqa_kv_cache,
                        share_layer_pq_codebook=args.share_layer_pq_codebook,
                    )
                    try:
                        if args.prebuild_sparse_cache:
                            _prebuild_sparse_adapters(
                                patched_pq,
                                sparse_prefill_cache,
                                share_gqa_kv_cache=args.share_gqa_kv_cache,
                            )
                        sparse_gen = _greedy_generate_text(
                            model,
                            tokenizer,
                            input_ids,
                            sparse_prefill_cache,
                            context_tokens=args.context_tokens,
                            gen_steps=gen_steps,
                        )
                    finally:
                        del sparse_prefill_cache
                        _restore_pq_sparse_patch(patched_pq)
                        if input_device.type == "cuda":
                            torch.cuda.empty_cache()

                sparse_correct = passkey in sparse_gen

                dense_results_per_depth[depth].append(dense_correct)
                sparse_results_per_depth[depth].append(sparse_correct)

                trial = {
                    "config_id": config_id,
                    "name": config["name"],
                    "depth": depth,
                    "actual_depth": actual_depth,
                    "repeat": rep,
                    "passkey": passkey,
                    "dense_gen": dense_gen,
                    "sparse_gen": sparse_gen,
                    "dense_correct": dense_correct,
                    "sparse_correct": sparse_correct,
                }
                all_trials.append(trial)
                print(json.dumps({"event": "needle_trial", **trial}, sort_keys=True), flush=True)

        # Per-depth accuracy summary.
        for depth in depths:
            d_acc = sum(dense_results_per_depth[depth]) / max(1, len(dense_results_per_depth[depth]))
            s_acc = sum(sparse_results_per_depth[depth]) / max(1, len(sparse_results_per_depth[depth]))
            print(
                json.dumps(
                    {
                        "event": "needle_depth_summary",
                        "config_id": config_id,
                        "depth": depth,
                        "dense_accuracy": d_acc,
                        "sparse_accuracy": s_acc,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    report = {"task": "needle", "trials": all_trials}
    if args.json_output is not None:
        with open(args.json_output, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
    if args.jsonl_output is not None:
        with open(args.jsonl_output, "w", encoding="utf-8") as fh:
            for trial in all_trials:
                fh.write(json.dumps(trial, sort_keys=True) + "\n")
    if args.csv_output is not None:
        _write_csv(args.csv_output, all_trials)
    print(json.dumps(report, indent=2, sort_keys=True))


def _install_pq_sparse_patch(
    model: torch.nn.Module,
    index_config: IVFPQConfig,
    attention_config: SparseAttentionConfig,
    *,
    share_gqa_kv_cache: bool,
    share_layer_pq_codebook: bool,
) -> list[torch.nn.Module]:
    model_type = getattr(model.config, "model_type", "")
    if model_type in {"qwen3", "qwen3_moe"}:
        forward = _qwen3_pq_sparse_forward
    elif model_type in {"llama", "qwen2", "mistral"}:
        # Qwen2(.5) and Mistral attention are architecturally identical to Llama
        # for our purposes: same forward signature (transformers>=4.48 unified
        # attention API), same rotary embedding math (llama_apply_rotary_pos_emb),
        # GQA via num_key_value_groups, and no q/k norm (unlike Qwen3).
        forward = _llama_pq_sparse_forward
    else:
        raise ValueError(f"unsupported model_type for PQ sparse patch: {model_type}")

    patched = []
    for layer_idx, decoder_layer in enumerate(model.model.layers):
        attn = decoder_layer.self_attn
        if hasattr(attn, "_e2e_pq_original_forward"):
            raise RuntimeError("model is already patched; restore before installing a new PQ config")
        attn._e2e_pq_original_forward = attn.forward
        attn._e2e_pq_index_config = copy.deepcopy(index_config)
        attn._e2e_pq_attention_config = copy.deepcopy(attention_config)
        attn._e2e_pq_share_gqa_kv_cache = bool(share_gqa_kv_cache)
        attn._e2e_pq_share_layer_pq_codebook = bool(share_layer_pq_codebook)
        attn._e2e_pq_adapter = None
        attn._e2e_pq_layer_idx = layer_idx
        attn._e2e_pq_stats = _empty_attention_stats()
        attn._e2e_pq_prebuild_profile = {}
        attn.forward = MethodType(forward, attn)
        patched.append(attn)
    return patched


def _prebuild_sparse_adapters(
    patched: list[torch.nn.Module],
    cache: DynamicCache,
    *,
    share_gqa_kv_cache: bool,
) -> float:
    legacy_cache = _legacy_kv_pairs(cache)
    if len(legacy_cache) != len(patched):
        raise ValueError(
            f"cache has {len(legacy_cache)} layers but model patch has {len(patched)} layers"
        )
    device = _first_cache_device(legacy_cache)
    start = _time_start(device)
    for attn, (key_states, value_states) in zip(patched, legacy_cache, strict=True):
        if getattr(attn, "_e2e_pq_adapter", None) is not None:
            continue
        use_shared_gqa = (
            bool(share_gqa_kv_cache)
            and int(getattr(attn, "num_key_value_groups", 1)) > 1
        )
        if not use_shared_gqa and int(getattr(attn, "num_key_value_groups", 1)) > 1:
            key_states = _repeat_kv_cache_tensor(
                key_states,
                int(getattr(attn, "num_key_value_groups")),
            )
            value_states = _repeat_kv_cache_tensor(
                value_states,
                int(getattr(attn, "num_key_value_groups")),
            )
        attn._e2e_pq_adapter = _new_sparse_adapter_for_kv(
            attn,
            key_states,
            share_gqa_kv_cache=use_shared_gqa,
        )
        attn._e2e_pq_adapter.build_cache(
            key_states.detach().contiguous(),
            value_states.detach().contiguous(),
        )
    return _time_stop(device, start)


def _warmup_sparse_adapters(
    patched: list[torch.nn.Module],
    *,
    steps: int,
) -> float:
    if steps <= 0:
        return 0.0

    # Warm every adapter through its REAL decode path (adapter.forward), not a
    # single stream per device.  This compiles Triton kernels, builds the
    # batched-heads snapshot, and — when PQ_USE_CUDA_GRAPH=1 — performs the
    # one-time CUDA-graph capture for all layers OUTSIDE the timed decode.
    warm_adapters = []
    for attn in patched:
        adapter = getattr(attn, "_e2e_pq_adapter", None)
        if adapter is None:
            continue
        streams = getattr(adapter, "streams", ())
        if not streams:
            continue
        cache = getattr(streams[0], "cache", None)
        if cache is None or cache.index is None or cache.regions.retrieval.numel() == 0:
            continue
        warm_adapters.append(adapter)

    if not warm_adapters:
        return 0.0

    first_cache = warm_adapters[0].streams[0].cache
    first_device = torch.device(first_cache.index_device)
    start = _time_start(first_device)
    with torch.no_grad():
        for _ in range(steps):
            for adapter in warm_adapters:
                stream = adapter.streams[0]
                cache = stream.cache
                assert cache is not None
                num_kv_heads = len(adapter.streams)
                query_count = int(getattr(adapter, "num_key_value_groups", 1))
                query = torch.zeros(
                    1,
                    num_kv_heads * query_count,
                    1,
                    cache._key_shape[0],
                    device=cache.index_device,
                    dtype=cache.keys.dtype,
                )
                adapter.forward(query)
    elapsed_ms = _time_stop(first_device, start)
    _reset_sparse_adapter_runtime_stats(patched)
    return elapsed_ms


def _reset_sparse_adapter_runtime_stats(patched: list[torch.nn.Module]) -> None:
    for attn in patched:
        if hasattr(attn, "_e2e_pq_stats"):
            attn._e2e_pq_stats = _empty_attention_stats()
        adapter = getattr(attn, "_e2e_pq_adapter", None)
        if adapter is None:
            continue
        if hasattr(adapter, "reset_profile_stats"):
            adapter.reset_profile_stats()
        for stream in getattr(adapter, "streams", ()):
            stream.profile_stats.clear()
            if stream.cache is not None:
                stream.cache.reset_fetch_stats()
                if hasattr(stream.cache, "reset_profile_stats"):
                    stream.cache.reset_profile_stats()


def _capture_sparse_adapter_prebuild_profiles(patched: list[torch.nn.Module]) -> None:
    for attn in patched:
        attn._e2e_pq_prebuild_profile = _collect_adapter_profile(
            getattr(attn, "_e2e_pq_adapter", None)
        )


def _first_cache_device(
    legacy_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
) -> torch.device:
    if len(legacy_cache) == 0:
        return torch.device("cpu")
    return legacy_cache[0][0].device


def _repeat_kv_cache_tensor(tensor: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return tensor
    batch, kv_heads, seq_len, dim = tensor.shape
    return (
        tensor[:, :, None, :, :]
        .expand(batch, kv_heads, groups, seq_len, dim)
        .reshape(batch, kv_heads * groups, seq_len, dim)
        .contiguous()
    )


def _restore_pq_sparse_patch(patched: list[torch.nn.Module]) -> None:
    for attn in patched:
        original = getattr(attn, "_e2e_pq_original_forward", None)
        if original is not None:
            attn.forward = original
        for name in (
            "_e2e_pq_original_forward",
            "_e2e_pq_index_config",
            "_e2e_pq_attention_config",
            "_e2e_pq_share_gqa_kv_cache",
            "_e2e_pq_share_layer_pq_codebook",
            "_e2e_pq_adapter",
            "_e2e_pq_layer_idx",
            "_e2e_pq_stats",
            "_e2e_pq_prebuild_profile",
        ):
            if hasattr(attn, name):
                delattr(attn, name)


def _qwen3_pq_sparse_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_values: DynamicCache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    del attention_mask, kwargs
    input_shape = hidden_states.shape[:-1]
    if len(input_shape) != 2 or input_shape[1] != 1 or past_key_values is None:
        return self._e2e_pq_original_forward(
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
    query_states, key_states = qwen3_apply_rotary_pos_emb(query_states, key_states, cos, sin)
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = past_key_values.update(
        key_states,
        value_states,
        self.layer_idx,
        cache_kwargs,
    )
    if not _should_share_gqa_kv_cache(self):
        key_states = qwen3_repeat_kv(key_states, self.num_key_value_groups).contiguous()
        value_states = qwen3_repeat_kv(value_states, self.num_key_value_groups).contiguous()
    return _pq_sparse_project_output(self, input_shape, query_states, key_states, value_states)


def _llama_pq_sparse_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_values: DynamicCache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    del attention_mask, kwargs
    input_shape = hidden_states.shape[:-1]
    if len(input_shape) != 2 or input_shape[1] != 1 or past_key_values is None:
        return self._e2e_pq_original_forward(
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
    query_states, key_states = llama_apply_rotary_pos_emb(query_states, key_states, cos, sin)
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = past_key_values.update(
        key_states,
        value_states,
        self.layer_idx,
        cache_kwargs,
    )
    if not _should_share_gqa_kv_cache(self):
        key_states = llama_repeat_kv(key_states, self.num_key_value_groups).contiguous()
        value_states = llama_repeat_kv(value_states, self.num_key_value_groups).contiguous()
    return _pq_sparse_project_output(self, input_shape, query_states, key_states, value_states)


def _pq_sparse_project_output(
    attn: torch.nn.Module,
    input_shape: torch.Size,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> tuple[torch.Tensor, None]:
    query_for_index = query_states.detach().contiguous()
    key_for_cache = key_states.detach().contiguous()
    value_for_cache = value_states.detach().contiguous()
    start = _wall_time_start()

    if attn._e2e_pq_adapter is None:
        attn._e2e_pq_adapter = _new_sparse_adapter_for_kv(
            attn,
            key_for_cache,
            query_states=query_states,
            share_gqa_kv_cache=_should_share_gqa_kv_cache(attn),
        )
        attn._e2e_pq_adapter.build_cache(key_for_cache, value_for_cache)
        adapter_output = attn._e2e_pq_adapter.forward(query_for_index)
    else:
        adapter_output = attn._e2e_pq_adapter.decode_step(
            query_for_index,
            key_for_cache[:, :, -1, :],
            value_for_cache[:, :, -1, :],
            append_kv=True,
        )

    elapsed_ms = _wall_time_stop(start)
    _update_attention_stats(attn._e2e_pq_stats, adapter_output, elapsed_ms)
    if os.environ.get("PQ_COLLECT_RECALL", "0") == "1":
        _record_recall_from_adapter(attn, query_for_index, key_for_cache, adapter_output)
    attn_output = adapter_output.context.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = attn.o_proj(attn_output)
    return attn_output, None


def _record_recall_from_adapter(
    attn: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    adapter_output: Any,
) -> None:
    """Oracle top-k recall of IVF candidates / exact rerank vs dense qk.

    Candidate indices from the index are local to the retrieval region.
    Dense top-k is computed on the same retrieval keys and decode query.
    """
    sink = int(getattr(attn._e2e_pq_attention_config, "sink_tokens", 4))
    local = int(getattr(attn._e2e_pq_attention_config, "local_window", 128))
    frac = getattr(attn._e2e_pq_attention_config, "retrieval_top_fraction", 0.01)
    seq_len = int(key_states.shape[-2])
    retrieval = _retrieval_region_indices(
        seq_len, sink_tokens=sink, local_window=local, device=key_states.device
    )
    if retrieval.numel() == 0:
        return
    k_oracle = max(1, int(math.ceil(float(frac) * retrieval.numel())))
    q_heads = int(query_states.shape[1])
    kv_heads = int(key_states.shape[1])
    groups = max(1, q_heads // kv_heads)
    per_stream = []
    for query_output in adapter_output.per_query:
        per_stream.extend(list(query_output.per_stream))
    records = getattr(attn, "_e2e_pq_recall", None)
    if records is None:
        records = []
        attn._e2e_pq_recall = records
    stream_idx = 0
    for kv_h in range(kv_heads):
        retr_keys = key_states[0, kv_h].index_select(0, retrieval)
        for g in range(groups):
            q_h = kv_h * groups + g
            if q_h >= q_heads or stream_idx >= len(per_stream):
                continue
            query = query_states[0, q_h, 0].to(dtype=torch.float32)
            scores = torch.matmul(retr_keys.float(), query)
            oracle = torch.topk(scores, k=min(k_oracle, scores.numel()), largest=True).indices
            oracle_set = set(int(x) for x in oracle.detach().cpu().tolist())
            stream_output = per_stream[stream_idx]
            stream_idx += 1
            sr = getattr(stream_output, "search_result", None)
            cand_local: set[int] = set()
            budget = int(getattr(attn._e2e_pq_attention_config, "candidate_budget", 4096) or 4096)
            if sr is not None and sr.candidate_indices is not None:
                cand = sr.candidate_indices.detach().reshape(-1)
                scores = getattr(sr, "candidate_scores", None)
                if scores is not None and cand.numel() > budget:
                    top = scores.detach().reshape(-1).topk(min(budget, cand.numel()), largest=True).indices
                    cand = cand.index_select(0, top)
                elif cand.numel() > budget:
                    cand = cand[:budget]
                cand_local = set(
                    int(x) for x in cand.detach().cpu().tolist() if 0 <= int(x) < retrieval.numel()
                )
            rerank_local: set[int] = set()
            exact_idx = getattr(stream_output, "indices", None)
            if exact_idx is not None:
                exact = exact_idx.detach().reshape(-1).cpu()
                retr_cpu = retrieval.detach().cpu()
                global_to_local = {int(gpos): i for i, gpos in enumerate(retr_cpu.tolist())}
                for raw in exact.tolist():
                    val = int(raw)
                    if val in global_to_local:
                        rerank_local.add(global_to_local[val])
                    elif 0 <= val < retrieval.numel():
                        rerank_local.add(val)
            hit_c = len(oracle_set & cand_local)
            hit_r = len(oracle_set & rerank_local)
            needle_pos = getattr(attn, "_e2e_pq_needle_pos", None)
            needle_local = None
            if needle_pos is not None:
                retr_list = retrieval.detach().cpu().tolist()
                if int(needle_pos) in retr_list:
                    needle_local = retr_list.index(int(needle_pos))
            records.append(
                {
                    "layer": int(getattr(attn, "_e2e_pq_layer_idx", -1)),
                    "kv_head": kv_h,
                    "group": g,
                    "oracle_k": len(oracle_set),
                    "n_candidates": len(cand_local),
                    "n_rerank": len(rerank_local),
                    "recall_candidates": hit_c / max(1, len(oracle_set)),
                    "recall_rerank": hit_r / max(1, len(oracle_set)),
                    "needle_in_candidates": (
                        None if needle_local is None else needle_local in cand_local
                    ),
                    "needle_in_rerank": (
                        None if needle_local is None else needle_local in rerank_local
                    ),
                }
            )


def _new_sparse_adapter_for_kv(
    attn: torch.nn.Module,
    key_states: torch.Tensor,
    *,
    query_states: torch.Tensor | None = None,
    share_gqa_kv_cache: bool,
) -> IVFPQDecodeAttentionAdapter | "GQAIVFPQDecodeAttentionAdapter":
    if share_gqa_kv_cache:
        num_key_value_heads = key_states.shape[1]
        if query_states is not None:
            num_query_heads = query_states.shape[1]
        else:
            num_query_heads = num_key_value_heads * int(getattr(attn, "num_key_value_groups", 1))
        return GQAIVFPQDecodeAttentionAdapter(
            attn._e2e_pq_index_config,
            attn._e2e_pq_attention_config,
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            share_layer_pq_codebook=bool(
                getattr(attn, "_e2e_pq_share_layer_pq_codebook", False)
            ),
        )
    return IVFPQDecodeAttentionAdapter(
        attn._e2e_pq_index_config,
        attn._e2e_pq_attention_config,
    )


def _should_share_gqa_kv_cache(attn: torch.nn.Module) -> bool:
    return (
        bool(getattr(attn, "_e2e_pq_share_gqa_kv_cache", True))
        and int(getattr(attn, "num_key_value_groups", 1)) > 1
    )


@dataclass(slots=True)
class _GroupedQueryAttentionOutput:
    output: torch.Tensor
    per_stream: tuple[Any, ...]


@dataclass(slots=True)
class _GroupedQueryDecodeOutput:
    context: torch.Tensor
    per_query: tuple[_GroupedQueryAttentionOutput, ...]


def _retrieval_region_indices(
    seq_len: int,
    *,
    sink_tokens: int,
    local_window: int,
    device: torch.device,
) -> torch.Tensor:
    sink_end = min(max(0, sink_tokens), seq_len)
    if local_window > 0:
        retrieval_end = max(sink_end, seq_len - local_window)
    else:
        retrieval_end = max(sink_end, seq_len)
    return torch.arange(sink_end, retrieval_end, device=device, dtype=torch.long)


def _list_sort_perm(index, *, device: torch.device) -> torch.Tensor:
    """Token permutation that groups retrieval keys by IVF list (CSR order).

    Derived from ``list_ids`` when the index has released its CSR tensors.
    ``torch.argsort(..., stable=True)`` is exactly how ``IVFPQIndex`` builds
    ``inverted_list_indices``, so the snapshot stays bit-identical.
    """

    if index.inverted_list_indices is not None:
        return index.inverted_list_indices.to(device=device, dtype=torch.long)
    return torch.argsort(index.list_ids.to(torch.int64), stable=True).to(device=device, dtype=torch.long)


def _list_sort_offsets(index, *, device: torch.device, num_lists: int) -> torch.Tensor:
    if index.inverted_list_offsets is not None:
        return index.inverted_list_offsets.to(device=device, dtype=torch.long)
    counts = torch.bincount(index.list_ids.to(device=device, dtype=torch.long), minlength=num_lists)
    offsets = torch.empty(num_lists + 1, device=device, dtype=torch.long)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0)
    return offsets


def _update_codebooks_ema_nosync(self, vectors: torch.Tensor, *, learning_rate: float) -> torch.Tensor:
    """Bit-identical, sync-free replacement for ``ProductQuantizer.update_codebooks_ema``.

    The library version indexes with the boolean ``non_empty`` mask three times
    per subspace; each mask index forces a device->host sync (nonzero), which
    measured ~3 ms per stream per deferred flush at 128K. Computing the
    blended codebook densely and selecting rows with ``torch.where`` performs
    the same elementwise arithmetic on the kept rows, so results are
    bit-identical while never leaving the device. Bound onto the sidecar's own
    ``ProductQuantizer`` instances only (see ``_install_fast_ema``).
    """
    self._check_trained()
    assert self.codebooks is not None
    assert self.subdim is not None

    if not (0.0 < learning_rate <= 1.0):
        raise ValueError("learning_rate must be in (0, 1]")
    if vectors.ndim != 2 or vectors.shape[1] != self.dim:
        raise ValueError(f"vectors must be [N, {self.dim}], got {tuple(vectors.shape)}")
    if vectors.shape[0] == 0:
        return torch.empty(
            0,
            self.config.num_subspaces,
            device=self.codebooks.device,
            dtype=torch.long,
        )

    num_vectors = vectors.shape[0]
    chunks = vectors.reshape(num_vectors, self.config.num_subspaces, self.subdim)
    assignments = []
    updated_codebooks = self.codebooks.clone()
    for subspace in range(self.config.num_subspaces):
        distances = torch.cdist(
            chunks[:, subspace, :].float(),
            self.codebooks[subspace].float(),
            p=2,
        )
        sub_assignments = distances.argmin(dim=1)
        assignments.append(sub_assignments)

        sums = torch.zeros_like(self.codebooks[subspace])
        counts = torch.bincount(
            sub_assignments,
            minlength=self.config.num_codes,
        ).to(vectors.dtype)
        sums.index_add_(0, sub_assignments, chunks[:, subspace, :])
        non_empty = counts > 0
        # counts >= 1 wherever non_empty, so clamp_min(eps) is a no-op on the
        # rows that survive the where(); empty rows divide by eps and are
        # discarded, exactly matching the masked-index version row-for-row.
        means = sums / counts.unsqueeze(1).clamp_min(self.config.eps)
        blended = (
            (1.0 - learning_rate) * self.codebooks[subspace] + learning_rate * means
        )
        updated_codebooks[subspace] = torch.where(
            non_empty.unsqueeze(1), blended, self.codebooks[subspace]
        )

    self.codebooks = updated_codebooks
    return torch.stack(assignments, dim=1)


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _t83e_append_scatter_kernel(
        sk_ptr, sv_ptr, fk_ptr, fv_ptr, mask_ptr, nk_ptr, nv_ptr,
        pos, spos,
        stride_sk_h, stride_sk_n,
        stride_sv_h, stride_sv_n,
        stride_fk_h, stride_fk_n,
        stride_fv_h, stride_fv_n,
        stride_m_h, stride_m_g,
        D, G,
        BLOCK_D: tl.constexpr,
    ):
        h = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        mask_d = offs < D
        nk = tl.load(nk_ptr + h * D + offs, mask=mask_d)
        nv = tl.load(nv_ptr + h * D + offs, mask=mask_d)
        tl.store(sk_ptr + h * stride_sk_h + pos * stride_sk_n + offs, nk, mask=mask_d)
        tl.store(sv_ptr + h * stride_sv_h + pos * stride_sv_n + offs, nv, mask=mask_d)
        tl.store(fk_ptr + h * stride_fk_h + spos * stride_fk_n + offs, nk, mask=mask_d)
        tl.store(fv_ptr + h * stride_fv_h + spos * stride_fv_n + offs, nv, mask=mask_d)
        g = tl.arange(0, 8)
        tl.store(
            mask_ptr + h * stride_m_h + g * stride_m_g + spos,
            0.0,
            mask=g < G,
        )
except Exception:  # pragma: no cover - triton optional at import
    triton = None  # type: ignore[assignment]
    _t83e_append_scatter_kernel = None


def _t83e_append_scatter(
    shared_k: torch.Tensor,
    shared_v: torch.Tensor,
    full_k: torch.Tensor,
    full_v: torch.Tensor,
    mask: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    pos: int,
    static_pos: int,
) -> bool:
    """One-launch copy of the new token into shared-base + graph full buffers.

    Bit-identical with five ``copy_`` / mask stores. Returns False to fall back.
    """
    if _t83e_append_scatter_kernel is None or triton is None:
        return False
    new_k = new_k.contiguous()
    new_v = new_v.contiguous()
    H, D = int(new_k.shape[0]), int(new_k.shape[1])
    G = int(mask.shape[1])
    block_d = 128 if D <= 128 else int(triton.next_power_of_2(D))
    try:
        _t83e_append_scatter_kernel[(H,)](
            shared_k, shared_v, full_k, full_v, mask, new_k, new_v,
            pos, static_pos,
            shared_k.stride(0), shared_k.stride(1),
            shared_v.stride(0), shared_v.stride(1),
            full_k.stride(0), full_k.stride(1),
            full_v.stride(0), full_v.stride(1),
            mask.stride(0), mask.stride(1),
            D, G,
            BLOCK_D=block_d,
        )
        return True
    except Exception:
        return False


# --- Background-mass diagnostic (opt-in: PQ_HSA_E21_BGMASS_LOG=1) ------
# Prints, for the first few decode steps of every layer, how the softmax
# denominator splits between the PQ-centroid background term, the exact
# reranked set, and the sink+local window. This is what distinguishes
# "the uncalibrated background mass swamps the selected tokens" from
# "the selection went wrong": under C2 the background share should approach
# 1.0 on the aggregation tasks (cwe/fwe) and stay small on niah.
#
# Layer ordinals are assigned on first use, in call order, so ordinal 0 is the
# first adapter that runs in a forward pass and coming back to 0 means a new
# decode step. Strictly opt-in: one dict lookup per layer when unset.
_E21_BGMASS_STATE = {"next_ord": 0, "step": -1}


def _e21_bgmass_report(adapter, background_mass, exact_exp_sum, full_exp_sum, denom):
    st = _E21_BGMASS_STATE
    ordinal = getattr(adapter, "_e21_bgmass_ord", None)
    if ordinal is None:
        ordinal = st["next_ord"]
        adapter._e21_bgmass_ord = ordinal
        st["next_ord"] = ordinal + 1
    if ordinal == 0:
        st["step"] += 1
    step = st["step"]
    if step >= int(os.environ.get("PQ_HSA_E21_BGMASS_STEPS", "4")):
        return
    bg = background_mass.float().sum(dim=2)          # [H,G] net of the exact-approx subtraction
    dn = denom.float()                               # [H,G]
    bg_mean = bg.mean().item()
    if os.environ.get("PQ_HSA_E21_C2_MASS", "0") == "1":
        anchor = (os.environ.get("PQ_HSA_E21_C2_ANCHOR", _E21_C2_ANCHOR) or "mean").strip().lower()
    else:
        anchor = "off"
    print(
        f"[bgmass] layer={ordinal} step={step} c2_anchor={anchor} "
        f"bg_over_denom={(bg / dn).mean().item():.6f} "
        f"exact_over_denom={(exact_exp_sum.float() / dn).mean().item():.6f} "
        f"full_over_denom={(full_exp_sum.float() / dn).mean().item():.6f} "
        f"log10_bg_mass={math.log10(max(bg_mean, 1e-300)):.4f}",
        flush=True,
    )


def _e21_trunc_exact_only(full_logits, full_values, exact_logits, exact_values):
    """Exact-only softmax over E = sink+local (full region) + selected top-k.

    Same formula as the exact-only truncation epilogue used for the LongBench
    same-selector comparison: fp32, max taken over E only, output = N_E / Z_E. Shapes (eager path):
    full_logits [H,G,F], full_values [H,F,Dv], exact_logits [H,G,K],
    exact_values [H,G,K,Dv] -> [H,G,Dv] fp32.
    """
    f = full_logits.float()
    e = exact_logits.float()
    m = f.amax(dim=-1)
    if e.shape[-1]:
        m = torch.maximum(m, e.amax(dim=-1))
    pf = torch.exp(f - m.unsqueeze(-1))
    pe = torch.exp(e - m.unsqueeze(-1))
    z = pf.sum(dim=-1) + pe.sum(dim=-1)
    n = torch.einsum("hgf,hfd->hgd", pf, full_values.float())
    if e.shape[-1]:
        n = n + torch.einsum("hgk,hgkd->hgd", pe, exact_values.float())
    return n / z.unsqueeze(-1)


def _e21_trunc_epilogue(adapter, hybrid_output, full_logits, full_values,
                        exact_logits, exact_values, exact_global,
                        background_mass, denom):
    """Arm T epilogue + opt-in probe. Called only when _E21_TRUNC_ALLPQ
    or _E21_PROBE is set; everything upstream (LUT, scan, all-PQ top-k, exact
    rerank, hybrid softmax) is arm B's code, untouched."""
    out = hybrid_output
    t_out = None
    if _E21_TRUNC_ALLPQ:
        t_out = _e21_trunc_exact_only(full_logits, full_values, exact_logits, exact_values)
        _E21_T_STATS["hits"] += 1
        if _E21_T_STATS["hits"] == 1:
            print("[variant-T] fixed-selector truncation active on EAGER batched-heads path "
                  "(all-PQ top-k selection unchanged; background mass=0, value=0; out=N_E/Z_E)",
                  flush=True)
        out = t_out.to(hybrid_output.dtype)
    if _E21_PROBE:
        step = getattr(adapter, "_e21_probe_step", 0)
        adapter._e21_probe_step = step + 1
        if step < _E21_PROBE_STEPS:
            if t_out is None:
                t_out = _e21_trunc_exact_only(full_logits, full_values, exact_logits, exact_values)
            hyb = hybrid_output.float()
            bg = background_mass.float().sum(dim=2)
            rec = {
                "sample": getattr(adapter, "_e21_sample", -1),
                "layer": getattr(adapter, "_e21_layer_idx", -1),
                "step": step,
                "trunc_active": bool(_E21_TRUNC_ALLPQ),
                "rel_l2_T_vs_hybrid": float((t_out - hyb).norm() / hyb.norm().clamp_min(1e-30)),
                "bg_over_denom_mean": float((bg / denom.float()).mean()),
                "returned_is_T": bool(out is not hybrid_output),
            }
            if exact_global is not None and step == 0:
                rec["exact_global"] = exact_global.to(torch.int32).cpu()
            _E21_T_STATS["probe"].append(rec)
    return out


def _e21_bx_list_value_sums(adapter, bh, N, Dv):
    """Per-(kv head, IVF list) fp32 SUM of the resident fp16 values of every
    retrieval-region member, plus member counts; once per batched-heads snapshot."""
    cached = bh.get("_e21_bx")
    if cached is not None and cached[2] == N:
        return cached[0], cached[1]
    list_ids = bh["list_ids"][:, :N].long()
    rg = bh["retrieval_global"][:, :N].long()
    H = list_ids.shape[0]
    L = int(bh["num_lists"])
    sbv = bh.get("shared_base_vals")
    dev = list_ids.device
    sums = torch.zeros(H, L, Dv, device=dev, dtype=torch.float32)
    cnt = torch.zeros(H, L, device=dev, dtype=torch.float32)
    for h in range(H):
        if sbv is not None:
            vh = sbv[h].index_select(0, rg[h])
        else:
            cv = adapter._streams[h].cache.values
            vh = cv[rg[h].to(cv.device)].to(dev)
        sums[h].index_add_(0, list_ids[h], vh.float())
        cnt[h].index_add_(0, list_ids[h], torch.ones(N, device=dev, dtype=torch.float32))
    bh["_e21_bx"] = (sums, cnt, N)
    return sums, cnt


def _e21_bx_unselected_means(sums, cnt, list_ids_n, exact_local, exact_values):
    """vbar^B [H,G,L,Dv] (fp32): per query row, list sums minus the row's selected
    members' values, divided by the unselected count; 0 where no member is unselected."""
    H, L, Dv = sums.shape
    G = exact_local.shape[1]
    N = list_ids_n.shape[1]
    s = sums.unsqueeze(1).expand(H, G, L, Dv).clone()
    c = cnt.unsqueeze(1).expand(H, G, L).clone()
    if exact_local.shape[-1]:
        ex_lists = torch.gather(list_ids_n.long().unsqueeze(1).expand(H, G, N), 2, exact_local)   # [H,G,K]
        s.scatter_add_(2, ex_lists.unsqueeze(-1).expand(-1, -1, -1, Dv), -exact_values.float())
        c.scatter_add_(2, ex_lists, torch.full(ex_lists.shape, -1.0, device=c.device))
    ok = c > 0.5
    return torch.where(ok.unsqueeze(-1), s / c.clamp_min(1.0).unsqueeze(-1), torch.zeros_like(s)), c


def _e21_bx_epilogue(adapter, hybrid_output, full_output, exact_output, bh, exact_local,
                     exact_values, background_weight):
    N = int(bh["N"])
    Dv = exact_values.shape[-1]
    sums, cnt = _e21_bx_list_value_sums(adapter, bh, N, Dv)
    vbar_b, _ = _e21_bx_unselected_means(sums, cnt, bh["list_ids"][:, :N], exact_local, exact_values)
    bg_out = torch.einsum("hgl,hgld->hgd", background_weight.float(), vbar_b)
    out = full_output.float() + exact_output.float() + bg_out
    _E21_T_STATS["bx_hits"] = _E21_T_STATS.get("bx_hits", 0) + 1
    if _E21_T_STATS["bx_hits"] == 1:
        print("[variant-Bx] unselected-member list-mean background value active on EAGER batched-heads path "
              "(selection, background mass and normalisation = arm B)", flush=True)
    if _E21_PROBE:
        step = getattr(adapter, "_e21_probe_step", 0)
        adapter._e21_probe_step = step + 1
        if step < _E21_PROBE_STEPS:
            hyb = hybrid_output.float()
            vc = bh["value_centroids"].float().unsqueeze(1)                               # [H,1,L,Dv]
            _E21_T_STATS["probe"].append({
                "sample": getattr(adapter, "_e21_sample", -1),
                "layer": getattr(adapter, "_e21_layer_idx", -1),
                "step": step, "arm": "Bx",
                "rel_l2_Bx_vs_hybrid": float((out - hyb).norm() / hyb.norm().clamp_min(1e-30)),
                "rel_l2_vbarB_vs_vbar": float((vbar_b - vc).norm() / vc.expand_as(vbar_b).norm().clamp_min(1e-30)),
                "returned_is_Bx": True,
                "exact_global": (torch.gather(bh["retrieval_global"][:, :N].unsqueeze(1).expand(
                    exact_local.shape[0], exact_local.shape[1], N), 2, exact_local).to(torch.int32).cpu()
                    if (step == 0 and exact_local.shape[-1]) else None),
            })
    return out.to(hybrid_output.dtype)


def _e21_t2_diag(adapter, bh, q, scale, approx_logits, row_max, exact_local,
                 exact_logits, exact_approx_logits, full_logits, denom,
                 background_mass, first_cache, topk):
    st = _E21_T2_STATE
    npos = st.get("needle_pos")
    if npos is None or approx_logits is None or topk <= 0:
        return
    step = getattr(adapter, "_e21_t2_step", 0)
    adapter._e21_t2_step = step + 1
    if step >= int(st.get("max_steps", 16)):
        return
    with torch.no_grad():
        H, G, _ = q.shape
        dev = q.device
        L = int(first_cache._length)
        npos_d = npos.to(dev)
        sbk = bh.get("shared_base_keys")
        if sbk is not None:
            keys = sbk[:, :L, :]
        else:
            keys = torch.stack([adapter._streams[h].cache.keys[:L].to(dev) for h in range(H)], 0)
        dl = torch.einsum("hgd,hld->hgl", q.float(), keys.float()) * scale          # [H,G,L] dense logits, same q
        dlse = torch.logsumexp(dl, dim=-1)                                          # [H,G]
        dn = dl[:, :, npos_d]                                                       # [H,G,n]
        w_dense = torch.exp(dn - dlse.unsqueeze(-1)).sum(-1)                        # [H,G]
        needle_exact_max = dn.max(-1).values
        dense_rank = (dl > needle_exact_max.unsqueeze(-1)).sum(-1)                  # tokens above best needle token
        ret_global = bh["retrieval_global"]                                         # [H,N]
        N = ret_global.shape[1]
        is_needle = torch.isin(ret_global, npos_d)                                  # [H,N]
        isn = is_needle.unsqueeze(1).expand(H, G, N)
        sel = torch.zeros(H, G, N, dtype=torch.bool, device=dev)
        sel.scatter_(2, exact_local, True)
        n_needle_ret = is_needle.sum(-1)                                            # [H]
        n_needle_sel = (sel & isn).sum(-1)                                          # [H,G]
        rm = row_max.float().unsqueeze(-1)
        dnm = denom.float()
        ex_isn = torch.gather(isn, 2, exact_local)                                  # [H,G,K]
        w_hyb_exact = (torch.exp(exact_logits.float() - rm) * ex_isn).sum(-1) / dnm
        ap = approx_logits.float()
        w_hyb_bg = (torch.exp(ap - rm) * (isn & ~sel)).sum(-1) / dnm
        fl = full_logits.float()
        el = exact_logits.float()
        m = torch.maximum(fl.amax(-1), el.amax(-1)).unsqueeze(-1)
        z_e = torch.exp(fl - m).sum(-1) + torch.exp(el - m).sum(-1)
        w_trunc = (torch.exp(el - m) * ex_isn).sum(-1) / z_e
        bg_share = background_mass.float().sum(-1) / dnm
        needle_approx_max = torch.where(isn, ap, torch.full_like(ap, float("-inf"))).amax(-1)
        pq_rank = (ap > needle_approx_max.unsqueeze(-1)).sum(-1)                    # retrieval tokens above best needle (PQ)
        topk_thr = exact_approx_logits.float().amin(-1)                             # k-th PQ score (selection threshold)
        # exact logit of the best needle token vs PQ approx of that same token
        rec = {
            "sample": st.get("sample"), "layer": getattr(adapter, "_e21_layer_idx", -1),
            "step": step, "topk": int(topk), "N": int(N), "L": L,
            "n_needle_tok": int(npos.numel()),
        }
        for k, v in (("w_dense", w_dense), ("w_hyb_exact", w_hyb_exact), ("w_hyb_bg_needle", w_hyb_bg),
                     ("w_trunc", w_trunc), ("bg_share", bg_share), ("n_needle_sel", n_needle_sel),
                     ("n_needle_ret", n_needle_ret), ("needle_exact_max", needle_exact_max),
                     ("needle_approx_max", needle_approx_max), ("pq_rank", pq_rank),
                     ("dense_rank", dense_rank), ("topk_thr", topk_thr)):
            rec[k] = v.detach().to("cpu")
        st["records"].append(rec)


class GQAIVFPQDecodeAttentionAdapter:
    """Decode adapter that shares one sparse KV/PQ cache across each GQA query group."""

    def __init__(
        self,
        index_config: IVFPQConfig,
        attention_config: SparseAttentionConfig,
        *,
        num_query_heads: int,
        num_key_value_heads: int,
        share_layer_pq_codebook: bool = False,
    ):
        if num_query_heads % num_key_value_heads != 0:
            raise ValueError(
                "num_query_heads must be divisible by num_key_value_heads, got "
                f"{num_query_heads} and {num_key_value_heads}"
            )
        self.index_config = index_config
        self.attention_config = attention_config
        self.num_query_heads = int(num_query_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.num_key_value_groups = self.num_query_heads // self.num_key_value_heads
        self.share_layer_pq_codebook = bool(share_layer_pq_codebook)
        self._shared_codebook_profile: dict[str, float] = {}
        self._streams: tuple[IVFPQSparseAttention, ...] = ()
        self._leading_shape: tuple[int, int] | None = None
        self._key_dim: int | None = None
        self._value_dim: int | None = None
        # Phase-2 approx top-k flag. When True, replaces aten::topk with a
        # Triton block-topk + batched-merge pass (approximate; recall >= 0.85).
        self._approx_topk: bool = False
        # CUDA-graph decode: default True; set False to force eager path.
        # CUDA-graph decode replay. Wins in the standalone attention benchmark
        # (compare_dense_sparse_attention.py: 0.545 vs dense 0.681 ms/layer at
        # 128K), but inside the multi-GPU HF pipeline the per-step maintenance
        # + per-layer replay sentinel currently cost more than eager, and one
        # layer showed a rare non-finite replay (self-healed by the sentinel).
        # Default OFF for e2e until the in-model overhead is resolved.
        # Opt-in via env for e2e experiments (PQ_USE_CUDA_GRAPH=1).
        # Event marks cannot live inside a captured graph; disable graph
        # when PQ_T12_EVENTS=1 so phase events stay on the eager path.
        self._use_cuda_graph: bool = (
            os.environ.get("PQ_USE_CUDA_GRAPH", "0") == "1"
            and os.environ.get("PQ_T12_EVENTS", "0") != "1"
        )
        self._fast_append: bool = os.environ.get("PQ_HSA_FAST_APPEND", "1") == "1"
        self._cg_sentinel_interval: int = int(os.environ.get("PQ_CG_SENTINEL_INTERVAL", "16"))
        # Cache the events flag at construction (getenv every replay was
        # cheap but still on the decode hot path). Re-read when HOST_SYNC_FIX=0.
        self._steady_events: bool = os.environ.get("PQ_HSA_STEADY_EVENTS", "0") == "1"
        self._bk_caches = None
        self._bk_regions = None
        # CUDA-graph state.
        self._cg: dict | None = None  # captured graph + static buffers + metadata
        # Prefix-cache persistence state. This is a lightweight transactional
        # checkpoint: full K/V stays in the shared backing buffers while the
        # request-local lengths, compressed-index references, and graph snapshot
        # are restored between identical-prefix generate() calls.
        self._prefix_checkpoint: dict | None = None
        # (PQ_HSA_PAGED_KV=1, opt-in): vLLM paged-KV context for this
        # decode step, set once per step from pq_hsa_decode_runtime.run_pq_decode
        # (which already has kv_cache/attn_metadata in scope). None on the
        # default path -- every read site below falls back to
        # self._shared_base_keys/_vals whenever this is unset.
        self._paged_kv_cache: torch.Tensor | None = None
        self._paged_attn_metadata = None
        self._paged_seq_idx: int = 0

    @property
    def streams(self) -> tuple[IVFPQSparseAttention, ...]:
        return self._streams

    def set_paged_kv_context(self, kv_cache, attn_metadata, seq_idx: int = 0) -> None:
        """Stash vLLM's paged KV cache + metadata for this decode step."""
        self._paged_kv_cache = kv_cache
        self._paged_attn_metadata = attn_metadata
        self._paged_seq_idx = seq_idx

    # ------------------------------------------------------------------
    # Shrink the sidecar's own raw-K/V duplicate (_shared_base_keys/
    # _vals) to a small fixed-size ring buffer under PQ_HSA_PAGED_ATTEND=1.
    #
    # Why this is safe: once PAGED_ATTEND=1, nothing reads _shared_base_keys/
    # _vals back by ABSOLUTE position for correctness --
    #   * the exact-gather CUDA kernel reads vLLM's own paged kv_cache
    #     directly via block_table (cuda_attend_only_paged),
    #   * full_k/full_v (sink+local) are refilled from vLLM's paged kv_cache
    #     too (_cg_refill_buffers), not sliced from this buffer,
    #   * the periodic IVF-PQ flush/training range is read via a temporary
    #     gather_flashattn_kv_range() call (_apply_pending_batched), not
    #     sliced from this buffer.
    # What still needs it: pq_hsa/attention/kv_cache.py's frozen
    # SparseKVCache.__init__ requires `_key_buf.shape[0] >= initial_len`
    # (untouched, not editable) -- satisfied at construction time by the
    # transient full-length buffer build_cache() already builds; this method
    # runs AFTER that constructor returns and replaces it with a small ring,
    # remapping each stream's _key_buf/_val_buf/keys/values to point at it.
    # Steady-state writes then wrap via _paged_ring_pos() instead of growing.
    #
    # NOT ring-safe: restore_prefix()/checkpoint_prefix() read/write a real
    # historical row by absolute position (must not be modified) --
    # this adapter therefore only enables ring mode when build_cache() itself
    # is paged-attend-gated, and none of this round's benchmark scripts
    # invoke checkpoint/restore while PAGED_ATTEND=1 is set. If a caller does
    # combine them, restore_prefix's assertions/(silently wrong) row reads
    # would surface immediately.
    # ------------------------------------------------------------------
    def _paged_ring_cap(self) -> int:
        override = os.environ.get("PQ_HSA_PAGED_RING_CAP")
        if override:
            try:
                v = int(override)
                if v > 0:
                    return v
            except ValueError:
                pass
        ac = self.attention_config
        return (
            int(getattr(ac, "sink_tokens", 0) or 0)
            + int(getattr(ac, "local_window", 0) or 0)
            + int(getattr(ac, "index_update_interval", 0) or 0)
            + 4096
        )

    def _paged_ring_pos(self, pos: int) -> int:
        if not getattr(self, "_shared_base_ring", False):
            return int(pos)
        cap = int(self._shared_base_cap)
        return int(pos) % cap if cap > 0 else 0

    def _paged_ring_shrink(self, H: int, key_dim: int, val_dim: int, dtype, device) -> None:
        """Replace the just-built full-length shared base with a small ring.

        Called once at the end of build_cache(), only when
        _paged_attend_enabled() and the shared-base path was actually used.
        """
        ring_cap = max(64, self._paged_ring_cap())
        ring_k = torch.empty(H, ring_cap, key_dim, dtype=dtype, device=device)
        ring_v = torch.empty(H, ring_cap, val_dim, dtype=dtype, device=device)
        self._shared_base_keys = ring_k
        self._shared_base_vals = ring_v
        self._shared_base_cap = ring_cap
        self._shared_base_ring = True
        for h, stream in enumerate(self._streams):
            cache = stream.cache
            if cache is None:
                continue
            cache._key_buf = ring_k[h]
            cache._val_buf = ring_v[h]
            cache._buf_capacity = ring_cap
            n = min(int(cache._length), ring_cap)
            cache.keys = cache._key_buf[:n]
            cache.values = cache._val_buf[:n]

    def _install_fast_ema(self) -> None:
        """Bind the sync-free EMA codebook update onto this adapter's PQ objects."""
        for stream in self._streams:
            cache = stream.cache
            if cache is None or cache.index is None or cache.index.pq is None:
                continue
            pq = cache.index.pq
            if getattr(pq, "_t83d_fast_ema", False):
                continue
            pq.update_codebooks_ema = MethodType(_update_codebooks_ema_nosync, pq)
            pq._t83d_fast_ema = True

    def apply_pending_index_update(self) -> int:
        """Flush deferred IVF-PQ migrations and rebuild the decode snapshot."""
        self._lean_resync_views()
        applied = -1
        if os.environ.get("PQ_HSA_BATCHED_FLUSH", "1") == "1":
            applied = self._apply_pending_batched()
        if applied < 0:
            if os.environ.get("PQ_HSA_FAST_EMA", "1") == "1":
                self._install_fast_ema()
            applied = 0
            for stream in self._streams:
                if stream.apply_pending_index_update():
                    applied += 1
        if applied:
            # Try in-place snapshot + keep graph when N is unchanged.
            # _maybe_build_batched_heads invalidates only when reuse fails.
            self._bk_regions = None
            self._bk_caches = None
            self._maybe_build_batched_heads()
        return applied

    def _apply_pending_batched(self) -> int:
        """Head-batched deferred flush.

        The per-stream ``SparseKVCache.apply_pending_index_update`` runs the
        full online refresh once per KV head: at 128K this measured ~7-9 ms per
        head per layer (dozens of small kernels plus boolean-mask host syncs),
        i.e. ~60-70 ms per layer flush and 2.0-2.4 s per 512-token request.
        This method performs the same update equations for all H heads in
        batched tensor ops and writes the results back into each stream's
        index, cutting the flush to a handful of large kernels.

        Numerics: elementwise steps (normalization, EMA blends, residuals,
        packing, stable argsort CSR) match the per-stream path bit-for-bit.
        The coarse/PQ assignment distances go through batched ``torch.cdist``
        (baddbmm) instead of per-head 2D cdist (addmm), and index_add_
        accumulations run over flattened head-major indices, so ULP-level
        differences can flip near-tie assignments; measured agreement is
        reported in the report and both passkey gates must stay green.

        Returns the number of streams applied, or -1 when the batched path's
        preconditions do not hold (caller falls back to the per-stream loop).
        """
        caches = [s.cache for s in self._streams]
        if not caches or any(c is None for c in caches):
            return -1
        c0 = caches[0]
        if c0.index_update_strategy != "deferred":
            return -1
        if (
            getattr(self, "_shared_base_keys", None) is None
            or getattr(self, "_shared_base_vals", None) is None
            or getattr(self, "_shared_base_fallback", True)
        ):
            return -1
        pendings = [bool(c._deferred_index_update_pending) for c in caches]
        if not any(pendings):
            return 0
        if not all(pendings):
            return -1
        for c in caches:
            if (
                c.index is None
                or c.index.pq is None
                or c.index.pq.codebooks is None
                or c.index.coarse_centroids is None
                or c.codebook_refresh_interval is not None
                or c.kv_storage != "device"
                or c._length != c0._length
                or c._indexed_length != c0._indexed_length
                or c.sink_tokens != c0.sink_tokens
                or c.local_window != c0.local_window
            ):
                return -1
        idx0 = c0.index
        cfg = idx0.config
        if (
            cfg.rotation != "none"
            or not cfg.residual
            or not idx0.stores_packed_codes
            or cfg.num_subspaces % 2 != 0
            or cfg.num_bits != 4
        ):
            return -1

        H = len(caches)
        dev = self._shared_base_keys.device
        length = int(c0._length)
        old_retrieval_len = int(c0.regions.retrieval.numel())

        # --- bookkeeping (identical to SparseKVCache._refresh_index_now) ----
        sink_end = min(c0.sink_tokens, length)
        indexed_length = length
        if c0.local_window > 0:
            retrieval_end = max(sink_end, indexed_length - c0.local_window)
        else:
            retrieval_end = max(sink_end, indexed_length)
        regions = self._shared_regions(sink_end, retrieval_end, length, dev)
        for c in caches:
            c._indexed_length = indexed_length
            c._pending_update_tokens = 0
            c.regions = regions
            c._cached_sink_end = sink_end
            c._cached_retrieval_end = retrieval_end
            c._cached_length = length
            c._cached_regions = regions
            c.keys = c._key_buf[:length]
            c.values = c._val_buf[:length]

        N = retrieval_end - sink_end
        new_len = N - old_retrieval_len
        if new_len <= 0:
            for c in caches:
                c._deferred_index_update_pending = False
            return H

        lr = float(cfg.online_codebook_lr)
        eps = float(cfg.eps)
        num_lists = int(idx0.coarse_centroids.shape[0])
        M = int(cfg.num_subspaces)
        subdim = int(idx0.pq.subdim)
        K = int(idx0.pq.config.num_codes)
        next_refresh_count = int(c0._tokens_since_codebook_refresh) + new_len

        if (
            (_paged_kv_enabled() or getattr(self, "_shared_base_ring", False))
            and self._paged_kv_cache is not None
        ):
            # Temporary contiguous gather straight from vLLM's
            # paged KV for the flush-time training range -- used once here,
            # then dropped (no permanent second copy). Same bytes as the
            # _shared_base_keys slice below since both are populated from the
            # same per-step key/value tensors (reshape_and_cache_flash runs
            # before append() on every decode step). Mandatory (not just
            # same-bytes) once ring mode is on -- _shared_base_keys no
            # longer holds real historical content at absolute positions.
            keys_all, vals_all = gather_flashattn_kv_range(
                self._paged_kv_cache, self._paged_attn_metadata, self._paged_seq_idx,
                sink_end, retrieval_end,
            )  # [H, N, D] / [H, N, vD], freshly materialized (not a view)
        else:
            keys_all = self._shared_base_keys[:, sink_end:retrieval_end, :]  # [H, N, D] view
            vals_all = self._shared_base_vals[:, sink_end:retrieval_end, :]  # [H, N, vD] view
        D = int(keys_all.shape[-1])
        vD = int(vals_all.shape[-1])

        # --- _project_keys (rotation none) -----------------------------------
        if cfg.direction_normalize:
            norms = torch.linalg.vector_norm(keys_all.float(), dim=-1).to(keys_all.dtype)
            norms = norms.clamp_min(eps)  # [H, N]
            tkeys = keys_all / norms.unsqueeze(-1)
        else:
            norms = None
            tkeys = keys_all

        coarse = torch.stack(
            [c.index.coarse_centroids for c in caches], dim=0
        )  # [H, L, D] value copy of the pre-update centroids
        head_off_l = (
            torch.arange(H, device=dev, dtype=torch.long).unsqueeze(1) * num_lists
        )  # [H, 1]
        t_upd = tkeys[:, -new_len:, :].contiguous()  # [H, U, D]

        # --- coarse assignment of the update batch, EMA, re-assignment -------
        upd_ids = torch.cdist(t_upd.float(), coarse.float(), p=2).argmin(dim=-1)  # [H, U]
        flat_upd = (upd_ids + head_off_l).reshape(-1)
        sums = torch.zeros(H * num_lists, D, device=dev, dtype=tkeys.dtype)
        sums.index_add_(0, flat_upd, t_upd.reshape(-1, D))
        counts = torch.bincount(flat_upd, minlength=H * num_lists).to(tkeys.dtype)
        non_empty = counts > 0
        means = sums / counts.unsqueeze(1).clamp_min(eps)
        coarse_flat = coarse.reshape(H * num_lists, D)
        blended = (1.0 - lr) * coarse_flat + lr * means
        new_coarse = torch.where(non_empty.unsqueeze(1), blended, coarse_flat).reshape(
            H, num_lists, D
        )
        for h, c in enumerate(caches):
            c.index.coarse_centroids.copy_(new_coarse[h])
        coarse = new_coarse
        upd_ids = torch.cdist(t_upd.float(), coarse.float(), p=2).argmin(dim=-1)
        upd_res = t_upd - torch.gather(
            coarse, 1, upd_ids.unsqueeze(-1).expand(H, new_len, D)
        )  # [H, U, D]

        # --- PQ codebook EMA on the update residuals --------------------------
        cbs = torch.stack(
            [c.index.pq.codebooks for c in caches], dim=0
        )  # [H, M, K, subdim]
        chunks_u = (
            upd_res.reshape(H, new_len, M, subdim)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(H * M, new_len, subdim)
        )
        cbs_flat = cbs.reshape(H * M, K, subdim)
        a_u = torch.cdist(chunks_u.float(), cbs_flat.float(), p=2).argmin(dim=-1)  # [H*M, U]
        row_off = torch.arange(H * M, device=dev, dtype=torch.long).unsqueeze(1) * K
        flat_a = (a_u + row_off).reshape(-1)
        sums2 = torch.zeros(H * M * K, subdim, device=dev, dtype=cbs.dtype)
        sums2.index_add_(0, flat_a, chunks_u.reshape(-1, subdim))
        counts2 = torch.bincount(flat_a, minlength=H * M * K).to(upd_res.dtype)
        non_empty2 = counts2 > 0
        means2 = sums2 / counts2.unsqueeze(1).clamp_min(eps)
        cbs_2d = cbs_flat.reshape(H * M * K, subdim)
        blended2 = (1.0 - lr) * cbs_2d + lr * means2
        new_cbs = torch.where(non_empty2.unsqueeze(1), blended2, cbs_2d).reshape(
            H, M, K, subdim
        )

        # --- assign + encode --------------------------------------------------
        # Default incremental flush keeps old list_ids / packed codes
        # and only assigns+encodes the new Δ tokens. Full-N cdist is the 0.8s
        # head. PQ_HSA_INCREMENTAL_FLUSH=0 restores the full re-encode.
        incremental = os.environ.get("PQ_HSA_INCREMENTAL_FLUSH", "1") == "1"
        old_ok = incremental and old_retrieval_len > 0 and all(
            c.index.list_ids is not None
            and c.index.packed_codes is not None
            and int(c.index.num_vectors) == old_retrieval_len
            and c.index.inverted_list_offsets is not None
            and c.retrieval_value_centroids is not None
            for c in caches
        )
        if old_ok:
            # Re-encode only the update batch against the *new* codebooks.
            # upd_res already uses the post-EMA coarse assignment.
            chunks_u2 = (
                upd_res.reshape(H, new_len, M, subdim)
                .permute(0, 2, 1, 3)
                .contiguous()
                .reshape(H * M, new_len, subdim)
            )
            codes_u = (
                torch.cdist(
                    chunks_u2.float(),
                    new_cbs.reshape(H * M, K, subdim).float(),
                    p=2,
                )
                .argmin(dim=-1)
                .reshape(H, M, new_len)
                .permute(0, 2, 1)
            )  # [H, U, M]
            codes_u8 = codes_u.to(torch.uint8)
            new_packed = (
                (codes_u8[..., 0::2] & 0x0F) | (codes_u8[..., 1::2] << 4)
            ).contiguous()
            old_ids = torch.stack(
                [c.index.list_ids.to(dtype=torch.int64) for c in caches], dim=0
            )  # [H, old_N]
            old_packed = torch.stack([c.index.packed_codes for c in caches], dim=0)
            all_ids = torch.cat([old_ids, upd_ids.to(dtype=torch.int64)], dim=1)
            packed = torch.cat([old_packed, new_packed], dim=1)

            counts_l = torch.bincount(
                (all_ids + head_off_l).reshape(-1), minlength=H * num_lists
            ).reshape(H, num_lists)
            offsets = torch.zeros(H, num_lists + 1, device=dev, dtype=torch.int64)
            offsets[:, 1:] = counts_l.cumsum(dim=1)
            inv_indices = torch.argsort(all_ids, dim=1, stable=True)

            old_cent = torch.stack(
                [c.retrieval_value_centroids for c in caches], dim=0
            )  # [H, L, vD]
            old_off = torch.stack(
                [c.index.inverted_list_offsets for c in caches], dim=0
            )  # [H, L+1]
            old_counts = (old_off[:, 1:] - old_off[:, :-1]).to(vals_all.dtype)
            vals_upd = vals_all[:, -new_len:, :].reshape(-1, vD)
            vsums_new = torch.zeros(H * num_lists, vD, device=dev, dtype=vals_all.dtype)
            vsums_new.index_add_(0, (upd_ids + head_off_l).reshape(-1), vals_upd)
            counts_new = torch.bincount(
                (upd_ids + head_off_l).reshape(-1), minlength=H * num_lists
            ).to(vals_all.dtype).reshape(H, num_lists)
            new_counts = old_counts + counts_new
            value_centroids = torch.where(
                (new_counts > 0).unsqueeze(-1),
                (
                    old_cent * old_counts.unsqueeze(-1)
                    + vsums_new.reshape(H, num_lists, vD)
                )
                / new_counts.unsqueeze(-1).clamp_min(eps),
                torch.zeros((), device=dev, dtype=vals_all.dtype),
            )
        else:
            all_ids = torch.cdist(tkeys.float(), coarse.float(), p=2).argmin(dim=-1)  # [H, N]
            res_all = tkeys - torch.gather(coarse, 1, all_ids.unsqueeze(-1).expand(H, N, D))
            chunks = (
                res_all.reshape(H, N, M, subdim)
                .permute(0, 2, 1, 3)
                .contiguous()
                .reshape(H * M, N, subdim)
            )
            codes = (
                torch.cdist(chunks.float(), new_cbs.reshape(H * M, K, subdim).float(), p=2)
                .argmin(dim=-1)
                .reshape(H, M, N)
                .permute(0, 2, 1)
            )  # [H, N, M]
            # pack_4bit_codes without its .min()/.max() host syncs (argmin of K=16
            # codewords is always in [0, 15]); nibble layout matches packing.py.
            codes_u8 = codes.to(torch.uint8)
            packed = (codes_u8[..., 0::2] & 0x0F) | (codes_u8[..., 1::2] << 4)  # [H, N, M//2]
            packed = packed.contiguous()

            # --- inverted lists (CSR), stable order identical to _rebuild ---------
            counts_l = torch.bincount(
                (all_ids + head_off_l).reshape(-1), minlength=H * num_lists
            ).reshape(H, num_lists)
            offsets = torch.zeros(H, num_lists + 1, device=dev, dtype=torch.int64)
            offsets[:, 1:] = counts_l.cumsum(dim=1)
            inv_indices = torch.argsort(all_ids, dim=1, stable=True)  # [H, N]

            # --- retrieval value centroids ----------------------------------------
            vsums = torch.zeros(H * num_lists, vD, device=dev, dtype=vals_all.dtype)
            vsums.index_add_(
                0, (all_ids + head_off_l).reshape(-1), vals_all.reshape(-1, vD)
            )
            vcounts = counts_l.to(vals_all.dtype)  # [H, L]
            value_centroids = torch.where(
                (vcounts > 0).unsqueeze(-1),
                vsums.reshape(H, num_lists, vD) / vcounts.unsqueeze(-1),
                torch.zeros((), device=dev, dtype=vals_all.dtype),
            )

        # --- write back per stream --------------------------------------------
        for h, c in enumerate(caches):
            idx = c.index
            idx.list_ids = idx._pack_list_ids(all_ids[h])
            idx.key_norms = None if norms is None else norms[h].contiguous()
            idx.packed_codes = packed[h]
            idx._codes = None
            idx.num_vectors = N
            idx.inverted_list_offsets = offsets[h]
            idx.inverted_list_indices = inv_indices[h]
            idx.pq.codebooks = new_cbs[h].clone()
            c.retrieval_value_centroids = value_centroids[h]
            c._retrieval_value_sums = None
            c._retrieval_value_counts = None
            c._tokens_since_codebook_refresh = next_refresh_count
            c.online_codebook_updates += 1
            c.key_fetches += 1
            c.value_fetches += 1
            c.fetched_key_rows += N
            c.fetched_value_rows += N
            c._deferred_index_update_pending = False
        return H

    @property
    def shared_codebook_profile(self) -> dict[str, float]:
        return self._shared_codebook_profile

    def reset_profile_stats(self) -> None:
        self._shared_codebook_profile = {}

    @property
    def prefix_checkpoint_length(self) -> int | None:
        checkpoint = self._prefix_checkpoint
        return None if checkpoint is None else int(checkpoint["length"])

    def checkpoint_prefix(self) -> None:
        """Checkpoint the built prefix without cloning its full K/V storage.

        A no-op when the ring buffer is active. restore_prefix()
        (must not be modified) reads/writes _shared_base_keys/_vals by
        ABSOLUTE historical position, which the small ring buffer this round
        introduces does not preserve (see _paged_ring_shrink). Skipping the
        checkpoint here means prefix_checkpoint_length stays None, so the
        caller's can_restore check (pq_hsa_decode_runtime.run_pq_decode)
        naturally always misses and takes the existing "rebuild via
        build_cache()" path instead of restore_prefix() -- correct (if
        slower for same-prompt-reuse callers) rather than corrupted.
        """
        if getattr(self, "_shared_base_ring", False) and not (
            _paged_restore_enabled() or _prefix_extend_enabled()
        ):
            self._prefix_checkpoint = None
            return
        self._lean_resync_views()
        self._check_built()
        stream_states = []
        for stream in self._streams:
            cache = stream.cache
            assert cache is not None and cache.index is not None
            index = cache.index
            assert index.pq is not None and index.coarse_centroids is not None
            stream_states.append({
                "cache": cache,
                "length": int(cache._length),
                "indexed_length": int(cache._indexed_length),
                "pending_update_tokens": int(cache._pending_update_tokens),
                "deferred_index_update_pending": bool(cache._deferred_index_update_pending),
                "tokens_since_codebook_refresh": int(cache._tokens_since_codebook_refresh),
                "index_rebuilds": int(cache.index_rebuilds),
                "index_adds": int(cache.index_adds),
                "online_codebook_updates": int(cache.online_codebook_updates),
                "codebook_refreshes": int(cache.codebook_refreshes),
                "deferred_index_updates": int(cache.deferred_index_updates),
                "regions": cache.regions,
                "cached_sink_end": int(cache._cached_sink_end),
                "cached_retrieval_end": int(cache._cached_retrieval_end),
                "cached_length": int(cache._cached_length),
                "cached_regions": cache._cached_regions,
                "retrieval_value_centroids": cache.retrieval_value_centroids,
                "retrieval_value_sums": cache._retrieval_value_sums,
                "retrieval_value_counts": cache._retrieval_value_counts,
                "index": index,
                # online_refresh mutates coarse centroids in place, while the
                # remaining index tensors are replaced. Keep one small value
                # copy for coarse and references for everything else.
                "coarse_centroids": index.coarse_centroids,
                "coarse_centroids_value": index.coarse_centroids.detach().clone(),
                "list_ids": index.list_ids,
                "inverted_list_offsets": index.inverted_list_offsets,
                "inverted_list_indices": index.inverted_list_indices,
                "codes": index._codes,
                "packed_codes": index.packed_codes,
                "pq": index.pq,
                "pq_codebooks": index.pq.codebooks,
                "rotation_matrix": index.rotation_matrix,
                "key_norms": index.key_norms,
                "dim": index.dim,
                "num_vectors": int(index.num_vectors),
            })

        first_cache = self._streams[0].cache
        assert first_cache is not None
        self._prefix_checkpoint = {
            "length": int(first_cache._length),
            "stream_states": tuple(stream_states),
            "batched_heads": self._batched_heads,
            "cg": self._cg,
            "cg_capture_failed": bool(getattr(self, "_cg_capture_failed", False)),
            "cg_capture_tries": int(getattr(self, "_cg_capture_tries", 0)),
        }

    def restore_prefix_ring(
        self,
        keys: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
        *,
        kv_cache: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        seq_idx: int = 0,
    ) -> bool:
        """(opt-in, PQ_HSA_PAGED_RESTORE=1): ring-buffer-aware prefix restore.

        In PQ_HSA_PAGED_ATTEND=1 mode the sidecar's K/V copy is a small ring
 that is never read: the exact gather reads vLLM's paged KV and
        the sink+local static buffers are refilled from paged KV by
        _cg_refill_buffers.  The frozen restore_prefix() writes that
        copy by ABSOLUTE position, which overflows the ring, so disabled
        checkpointing in ring mode -- every same-prompt generate() then rebuilt
        the whole index (~6.8 s at 128K, i.e. +100 ms/tok on a 64-step call).
        This wrapper restores the index / graph state through the unmodified
        restore_prefix(keys=None, values=None) (which skips every absolute
        K/V write), then applies the ring-mode equivalents of the two things
        that call skipped: the current decode token's row goes to the ring
        slot pos % cap, and the graph's static full_k/full_v last-local row is
        patched straight from the supplied keys/values.
        """
        if not getattr(self, "_shared_base_ring", False):
            return self.restore_prefix(
                keys, values, kv_cache=kv_cache, attn_metadata=attn_metadata, seq_idx=seq_idx
            )
        checkpoint = self._prefix_checkpoint
        if checkpoint is None:
            return False
        ok = self.restore_prefix(None, None)
        if not ok:
            return False
        base_length = int(checkpoint["length"])
        # The checkpoint shares the live graph dict; a flush during the previous
        # generate() rewrites the static sink+local rows.  In ring mode those rows
        # are re-gathered straight from vLLM's paged KV (cheap: ~sink+local tokens),
        # which makes the restore correct regardless of what the last call did.
        cg0 = self._cg
        if (
            cg0 is not None
            and getattr(self, "_paged_kv_cache", None) is not None
            and os.environ.get("PQ_HSA_PAGED_RESTORE_REFILL", "1") == "1"
        ):
            try:
                self._cg_refill_buffers(cg0["full_k"], cg0["full_v"], cg0["mask"], int(cg0["sink_end"]))
            except Exception as _rf_exc:
                self._paged_restore_refill_error = f"{type(_rf_exc).__name__}: {_rf_exc}"
        if keys is not None and values is not None:
            assert self._key_dim is not None and self._value_dim is not None
            k_row = keys.reshape(-1, self._key_dim)
            v_row = values.reshape(-1, self._value_dim)
            if self._shared_base_keys is not None and self._shared_base_vals is not None:
                wpos = self._paged_ring_pos(base_length - 1)
                self._shared_base_keys[:, wpos, :].copy_(k_row.to(self._shared_base_keys.dtype))
                self._shared_base_vals[:, wpos, :].copy_(v_row.to(self._shared_base_vals.dtype))
            cg = self._cg
            if cg is not None:
                first_cache = self._streams[0].cache
                assert first_cache is not None
                local_count = int(first_cache.regions.local.numel())
                if local_count > 0:
                    static_pos = int(cg["valid_full"]) - 1
                    cg["full_k"][:, static_pos, :].copy_(k_row.to(cg["full_k"].dtype))
                    cg["full_v"][:, static_pos, :].copy_(v_row.to(cg["full_v"].dtype))
        return True

    def append_many(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        start: int | None = None,
        flush_every: int = 0,
    ) -> int:
        """Vectorized sibling of append() for a contiguous run of tokens.

        ``keys``/``values`` are ``[H, n, D]`` (or ``[1, H, n, D]``) rows for
        positions ``[start, start + n)``; ``start`` defaults to the current
        length. Unlike n calls of append() this does ONE buffer write per
        stream (shared-base fast path when available, per-stream buffers
        otherwise, ring slots in PAGED_ATTEND mode) and updates the region /
        pending bookkeeping once, then marks a deferred index update. With
        ``flush_every > 0`` the run is ingested in chunks of that many tokens
        and ``apply_pending_index_update()`` runs after every chunk (same
        cadence a generated-token stream would see at ``flush_interval``);
        with 0 the caller flushes once afterwards. Returns the number of
        flushes performed. Bookkeeping is the same as append()'s tail
        (_notify_external_append) applied n times, without touching the
        CUDA graph's incremental slots (callers refresh the static buffers
        through the flush path or _cg_refill_buffers).
        """
        self._lean_resync_views()
        self._check_built()
        if keys.ndim == 4:
            keys = keys.reshape(-1, keys.shape[-2], keys.shape[-1])
            values = values.reshape(-1, values.shape[-2], values.shape[-1])
        if keys.ndim != 3 or values.ndim != 3:
            raise ValueError("append_many expects [H, n, D] keys/values")
        caches = [st.cache for st in self._streams]
        if not caches or any(c is None for c in caches):
            raise RuntimeError("append_many: streams not built")
        H = len(caches)
        if int(keys.shape[0]) != H:
            raise ValueError(f"append_many: expected {H} kv heads, got {int(keys.shape[0])}")
        c0 = caches[0]
        n = int(keys.shape[1])
        if n <= 0:
            return 0
        pos0 = int(c0._length) if start is None else int(start)
        if pos0 > int(c0._length):
            raise ValueError("append_many: start beyond current length")
        end = pos0 + n
        _ring = bool(getattr(self, "_shared_base_ring", False))
        use_shared = (
            self._shared_base_keys is not None
            and self._shared_base_vals is not None
            and not getattr(self, "_shared_base_fallback", True)
        )
        # --- buffer writes: one copy per stream / one strided write overall --
        if use_shared and not _ring:
            if end > int(self._shared_base_cap):
                raise ValueError(
                    f"append_many: end {end} exceeds shared base cap {int(self._shared_base_cap)}"
                )
            self._shared_base_keys[:, pos0:end, :].copy_(keys.to(self._shared_base_keys.dtype))
            self._shared_base_vals[:, pos0:end, :].copy_(values.to(self._shared_base_vals.dtype))
        elif use_shared and _ring:
            cap = int(self._shared_base_cap)
            take = min(n, cap)  # only the last `cap` rows can live in the ring
            slots = torch.arange(end - take, end, device=self._shared_base_keys.device) % cap
            self._shared_base_keys.index_copy_(1, slots, keys[:, n - take:, :].to(self._shared_base_keys.dtype))
            self._shared_base_vals.index_copy_(1, slots, values[:, n - take:, :].to(self._shared_base_vals.dtype))
        else:
            for h, c in enumerate(caches):
                while int(c._buf_capacity) < end:
                    c._grow_buf()
                c._key_buf[pos0:end].copy_(c._prepare_full_tensor(keys[h]))
                c._val_buf[pos0:end].copy_(c._prepare_full_tensor(values[h]))
        # --- bookkeeping (chunked or single) ---------------------------------
        flushes = 0
        step = int(flush_every) if int(flush_every) > 0 else n
        cur = pos0
        while cur < end:
            nxt = min(end, cur + step)
            for c in caches:
                c._length = nxt
                c._pending_update_tokens = int(c._pending_update_tokens) + (nxt - cur)
                n_view = min(nxt, int(c._buf_capacity)) if _ring else nxt
                c.keys = c._key_buf[:n_view]
                c.values = c._val_buf[:n_view]
                if not c._deferred_index_update_pending:
                    c.deferred_index_updates += 1
                    c._deferred_index_update_pending = True
                c.regions = c._make_regions(nxt, c.index_device, c._indexed_length)
            self._bk_regions = None
            self._bk_caches = None
            self._lean_views_dirty = False
            if int(flush_every) > 0 and nxt < end:
                self.apply_pending_index_update()
                flushes += 1
            cur = nxt
        return flushes

    def extend_prefix(
        self,
        keys: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
        *,
        kv_cache: torch.Tensor,
        attn_metadata: object,
        seq_idx: int = 0,
        new_length: int,
    ) -> dict | None:
        """(opt-in PQ_HSA_PREFIX_EXTEND=1): restore the checkpointed prefix
        and extend it IN PLACE to ``new_length`` tokens.

        Scenario: multi-turn / agentic serving where request r+1's prompt is
        request r's full prompt followed by new tokens (tool returns, the
        previous turn's generation). The exact-match checkpoint/restore path
        (restore_prefix / restore_prefix_ring) misses because the prompt digest
        and the length changed, and the runtime rebuilds the whole index
        (1.3-6 s at 128K). This method instead:

          1. restores the checkpointed index/graph state through the UNMODIFIED
             restore_prefix(None, None) (skips every absolute-position K/V
             write, so it is ring-safe);
          2. materializes the delta rows [base_length-1, new_length) -- the row
             of the old request's first decode token is overwritten by the new
             prompt's token at that position -- where the sidecar's own copy is
             still authoritative (non-ring shared base / per-stream buffers).
             In ring mode (PAGED_ATTEND) nothing is copied: the exact gather,
             sink+local refill and the flush training range all read vLLM's
             paged KV, which already holds the delta (prefill) and the current
             decode token (reshape_and_cache_flash ran before this call);
          3. advances every stream's length / pending counters and marks a
             deferred index update, i.e. the delta is treated exactly like a
             run of generated tokens that reached a flush boundary;
          4. runs apply_pending_index_update() -- the same batched flush path
             a generated-token flush uses (codebook EMA + incremental encode
             of the delta + CSR/value-centroid update + graph refill or
             recapture via _maybe_build_batched_heads).

        Codebooks are NOT retrained: the delta is encoded with the checkpointed
        codebooks after one EMA step (the standard deferred-flush semantics);
        this is an approximation relative to a full rebuild, bounded by the
        caller through PQ_HSA_PREFIX_EXTEND_MAX_FRAC.

        Returns a small dict of timings/counters on success, None when the
        caller must fall back to a full rebuild (no checkpoint, delta does not
        fit the non-ring shared base, paged context missing in ring mode).
        Default path (flag off) never reaches this method.
        """
        checkpoint = self._prefix_checkpoint
        if checkpoint is None:
            return None
        base_length = int(checkpoint["length"])
        new_length = int(new_length)
        if new_length <= base_length:
            return None
        _ring = bool(getattr(self, "_shared_base_ring", False))
        if _ring and getattr(self, "_paged_kv_cache", None) is None:
            return None
        info: dict = {"base_length": base_length, "new_length": new_length,
                      "delta_tokens": new_length - base_length, "ring": _ring}
        t0 = time.perf_counter()
        if not self.restore_prefix(None, None):
            return None
        info["restore_s"] = time.perf_counter() - t0

        self._lean_resync_views()
        caches = [s.cache for s in self._streams]
        if not caches or any(c is None for c in caches):
            return None
        c0 = caches[0]
        delta_start = base_length - 1
        n_delta = new_length - delta_start

        # --- 1) delta rows: one gather from vLLM's paged KV, one append_many ----
        # Non-ring: the sidecar's own K/V copy is still read by the exact
        # gather / sink+local refill, so the rows must land there. Ring
        # (PAGED_ATTEND): nothing reads the copy by absolute position; only
        # the tail rows are dropped into ring slots (parity with append()).
        t1 = time.perf_counter()
        use_shared = (
            self._shared_base_keys is not None
            and self._shared_base_vals is not None
            and not getattr(self, "_shared_base_fallback", True)
        )
        if use_shared and not _ring and new_length > int(self._shared_base_cap):
            # Delta does not fit the shared base's growth margin. The index
            # state is a consistent restore of the checkpoint; caller rebuilds.
            info["reject"] = "shared_base_cap"
            return None
        dk, dv = gather_flashattn_kv_range(
            kv_cache, attn_metadata, seq_idx, delta_start, new_length
        )  # [H, n_delta, D] / [H, n_delta, vD]; includes the current decode token row
        info["gather_s"] = time.perf_counter() - t1

        # --- 2) bookkeeping: the delta behaves like appended tokens ----------
        # Rewind the per-stream length to delta_start so the old request's
        # first-decode row is overwritten, then ingest [delta_start, new_length).
        for c in caches:
            c._length = delta_start
        chunk = 0
        try:
            chunk = int(os.environ.get("PQ_HSA_PREFIX_EXTEND_CHUNK", "0"))
        except ValueError:
            chunk = 0
        t_ing = time.perf_counter()
        info["chunk_flushes"] = self.append_many(dk, dv, start=delta_start, flush_every=chunk)
        info["ingest_s"] = time.perf_counter() - t_ing

        # --- 3) flush the delta into the index (generated-token flush path) --
        t2 = time.perf_counter()
        applied = int(self.apply_pending_index_update() or 0)
        info["flush_s"] = time.perf_counter() - t2
        info["flush_applied_streams"] = applied

        # --- 4) graph static buffers for the new local window ---------------
        # apply_pending_index_update -> _maybe_build_batched_heads already
        # refilled (graph reuse) or invalidated (recapture on next forward)
        # when something was flushed. If nothing needed flushing (delta
        # entirely inside the local window), refill sink+local explicitly.
        if applied <= 0:
            cg = self._cg
            if cg is not None:
                sink_end = min(int(c0.sink_tokens), new_length)
                self._cg_refill_buffers(cg["full_k"], cg["full_v"], cg["mask"], sink_end)
                local_count = int(c0.regions.local.numel())
                cg["valid_full"] = min(int(cg["sink_end"]) + local_count, int(cg["FULL_CAP"]))
        if getattr(self, "_wmg_ready", False):
            self._wmg_invalidate()
        self._cg_step_counter = 0
        info["total_s"] = time.perf_counter() - t0
        info["indexed_length"] = int(c0._indexed_length)
        info["retrieval_len"] = int(c0.regions.retrieval.numel())
        return info

    def restore_prefix(
        self,
        keys: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
        *,
        kv_cache: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        seq_idx: int = 0,
    ) -> bool:
        """Restore a checkpointed prefix after a completed decode request.

        The current request's first decode K/V may differ under non-greedy
        sampling, so the final checkpoint row is overwritten when supplied.
        It is in the exact local region and therefore does not alter the frozen
        retrieval snapshot.

        The checkpointed ``_shared_base_keys/_vals`` snapshot was taken
        during a *different* generate() call. When the current (restoring)
        call's prompt is the same token sequence, vLLM's own automatic prefix
        caching can serve most of it as a cache hit but still has to run a
        real forward pass over (at least) the last KV-cache block of the
        prompt to produce valid decode state -- that recompute is
        numerically deterministic but not necessarily bit-identical to the
        original checkpointing call's computation of that same block
        (reproduced with two same-prompt ``generate()`` calls on one
        engine, ctx=16384, ``PQ_HSA_FLUSH_INTERVAL`` irrelevant, stale range
        always exactly the block-size-token span immediately before the
        single row this function already patches; a single generate() call
        never shows it). The frozen snapshot
        restored above only fixes the single final row, leaving the rest of
        that last block stale relative to vLLM's *live* KV for the new
        request. When ``kv_cache``/``attn_metadata`` are supplied, this also
        refreshes that block-aligned tail range (a bounded, <=block_size-1
        token read) straight from vLLM's paged KV via the same
        ``gather_flashattn_kv_range`` helper already used elsewhere in this
        file -- a pure copy-boundary widening, no change to region/
        index semantics. Callers that do not pass them (e.g. the batch
        decode path today, or direct-call test probes) keep the prior
        single-row-only behavior unchanged.
        """
        checkpoint = self._prefix_checkpoint
        if checkpoint is None:
            return False

        for state in checkpoint["stream_states"]:
            cache = state["cache"]
            index = state["index"]
            coarse = state["coarse_centroids"]
            coarse.copy_(state["coarse_centroids_value"])
            index.coarse_centroids = coarse
            index.list_ids = state["list_ids"]
            index.inverted_list_offsets = state["inverted_list_offsets"]
            index.inverted_list_indices = state["inverted_list_indices"]
            index._codes = state["codes"]
            index.packed_codes = state["packed_codes"]
            index.pq = state["pq"]
            index.pq.codebooks = state["pq_codebooks"]
            index.rotation_matrix = state["rotation_matrix"]
            index.key_norms = state["key_norms"]
            index.dim = state["dim"]
            index.num_vectors = state["num_vectors"]

            cache.index = index
            cache._length = state["length"]
            cache._indexed_length = state["indexed_length"]
            cache._pending_update_tokens = state["pending_update_tokens"]
            cache._deferred_index_update_pending = state["deferred_index_update_pending"]
            cache._tokens_since_codebook_refresh = state["tokens_since_codebook_refresh"]
            cache.index_rebuilds = state["index_rebuilds"]
            cache.index_adds = state["index_adds"]
            cache.online_codebook_updates = state["online_codebook_updates"]
            cache.codebook_refreshes = state["codebook_refreshes"]
            cache.deferred_index_updates = state["deferred_index_updates"]
            cache.keys = cache._key_buf[: cache._length]
            cache.values = cache._val_buf[: cache._length]
            cache.regions = state["regions"]
            cache._cached_sink_end = state["cached_sink_end"]
            cache._cached_retrieval_end = state["cached_retrieval_end"]
            cache._cached_length = state["cached_length"]
            cache._cached_regions = state["cached_regions"]
            cache.retrieval_value_centroids = state["retrieval_value_centroids"]
            cache._retrieval_value_sums = state["retrieval_value_sums"]
            cache._retrieval_value_counts = state["retrieval_value_counts"]

        self._batched_heads = checkpoint["batched_heads"]
        self._cg = checkpoint["cg"]
        self._cg_capture_failed = checkpoint["cg_capture_failed"]
        self._cg_capture_tries = checkpoint["cg_capture_tries"]
        self._cg_step_counter = 0
        self._bk_regions = None
        self._bk_caches = None

        base_length = int(checkpoint["length"])
        if keys is not None or values is not None:
            if keys is None or values is None:
                raise ValueError("keys and values must be supplied together")
            assert self._leading_shape is not None
            assert self._key_dim is not None and self._value_dim is not None
            expected_key_shape = (*self._leading_shape, self._key_dim)
            expected_value_shape = (*self._leading_shape, self._value_dim)
            if tuple(keys.shape) != expected_key_shape:
                raise ValueError(f"keys must have shape {expected_key_shape}, got {tuple(keys.shape)}")
            if tuple(values.shape) != expected_value_shape:
                raise ValueError(
                    f"values must have shape {expected_value_shape}, got {tuple(values.shape)}"
                )
            assert self._shared_base_keys is not None and self._shared_base_vals is not None
            self._shared_base_keys[:, base_length - 1, :].copy_(
                keys.reshape(-1, self._key_dim)
            )
            self._shared_base_vals[:, base_length - 1, :].copy_(
                values.reshape(-1, self._value_dim)
            )

            # Refresh the rest of the tail block (see docstring) --
            # [tail_start, tail_end), tail_end = base_length - 1 (exclusive:
            # that row was just patched above with the current call's own
            # freshly-computed K/V, which is already live/authoritative).
            # tail_start is the start of the KV-cache block that CONTAINS
            # position tail_end - 1 -- i.e. the last full block of the
            # ORIGINAL PROMPT (base_length == prompt_len + 1 at checkpoint
            # time, so prompt_len == tail_end), not the block containing
            # base_length - 1 itself (locate: block-aligned prompt
            # lengths put position base_length-1 at the START of a fresh
            # block with nothing stale before it in THAT block -- the actual
            # stale range sits in the block before it, confirmed empirically).
            if kv_cache is not None and attn_metadata is not None and base_length > 1:
                # Layout-agnostic block size (vLLM 0.8.5 5-D vs >=0.10 4-D page).
                from benchmarks.vllm_backend.paged_kv_fa import kv_cache_dims as _kv_dims

                block_size = int(_kv_dims(kv_cache)[0])
                tail_end = base_length - 1
                tail_start = max(0, ((tail_end - 1) // block_size) * block_size) if tail_end > 0 else 0
                if tail_start < tail_end:
                    fresh_k, fresh_v = gather_flashattn_kv_range(
                        kv_cache, attn_metadata, seq_idx, tail_start, tail_end
                    )
                    self._shared_base_keys[:, tail_start:tail_end, :].copy_(
                        fresh_k.to(self._shared_base_keys.dtype)
                    )
                    self._shared_base_vals[:, tail_start:tail_end, :].copy_(
                        fresh_v.to(self._shared_base_vals.dtype)
                    )

        cg = self._cg
        if cg is not None:
            first_cache = self._streams[0].cache
            assert first_cache is not None
            local_count = int(first_cache.regions.local.numel())
            valid_full = int(cg["sink_end"]) + local_count
            if keys is not None and values is not None and local_count > 0:
                static_pos = valid_full - 1
                cg["full_k"][:, static_pos, :].copy_(
                    self._shared_base_keys[:, base_length - 1, :]
                )
                cg["full_v"][:, static_pos, :].copy_(
                    self._shared_base_vals[:, base_length - 1, :]
                )
            cg["mask"][:, :, :valid_full] = 0.0
            if valid_full < int(cg["FULL_CAP"]):
                cg["mask"][:, :, valid_full:] = float("-inf")
            cg["valid_full"] = valid_full
        return True

    def build_cache(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> "GQAIVFPQDecodeAttentionAdapter":
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError("keys and values must be [batch, kv_heads, seq, dim]")
        if keys.shape[:-2] != values.shape[:-2] or keys.shape[-2] != values.shape[-2]:
            raise ValueError("keys and values must have matching batch/head/sequence dimensions")
        if keys.shape[1] != self.num_key_value_heads:
            raise ValueError(
                f"expected {self.num_key_value_heads} KV heads, got {keys.shape[1]}"
            )

        self._leading_shape = (keys.shape[0], keys.shape[1])
        self._key_dim = keys.shape[-1]
        self._value_dim = values.shape[-1]
        flat_keys = keys.reshape(-1, keys.shape[-2], keys.shape[-1])
        flat_values = values.reshape(-1, values.shape[-2], values.shape[-1])
        H = flat_keys.shape[0]
        initial_len = flat_keys.shape[1]
        key_dim = flat_keys.shape[2]
        val_dim = flat_values.shape[2]
        shared_coarse_centroids = None
        shared_pq_codebooks = None
        if self.share_layer_pq_codebook and H > 1:
            retrieval_indices = _retrieval_region_indices(
                flat_keys.shape[1],
                sink_tokens=self.attention_config.sink_tokens,
                local_window=self.attention_config.local_window,
                device=flat_keys.device,
            )
            if retrieval_indices.numel() > 0:
                shared_training_keys = flat_keys[:, retrieval_indices, :].reshape(
                    -1,
                    flat_keys.shape[-1],
                ).contiguous()
                shared_index = IVFPQIndex(
                    self.index_config,
                    profile=self.attention_config.profile_attention_components,
                ).build(shared_training_keys)
                shared_coarse_centroids = shared_index.coarse_centroids
                assert shared_index.pq is not None
                shared_pq_codebooks = shared_index.pq.codebooks
                self._shared_codebook_profile = {
                    f"shared_codebook_{key}": float(value)
                    for key, value in shared_index.profile_stats.items()
                }

        # --- Batched per-head codebook training (opt-in) -----------
        # PQ_HSA_BATCHED_BUILD=1: train every head's coarse centroids + PQ
        # codebooks for this layer in two batched Flash-KMeans calls, then let
        # each stream build with its OWN (per-head) codebooks via the existing
        # shared-codebook constructor path.  Rebuild semantics are restored
        # below (see _e28_per_head_codebooks) so later deferred flushes retrain
        # exactly as the default path does.
        _e28_per_head_codebooks = None
        if (
            shared_coarse_centroids is None
            and H > 1
            and flat_keys.is_cuda
            and os.environ.get("PQ_HSA_BATCHED_BUILD", "0") == "1"
            and getattr(self.index_config, "rotation", "none") == "none"
        ):
            from pq_hsa.index.batched_build import train_codebooks_batched

            _ri = _retrieval_region_indices(
                flat_keys.shape[1],
                sink_tokens=self.attention_config.sink_tokens,
                local_window=self.attention_config.local_window,
                device=flat_keys.device,
            )
            if _ri.numel() > 0:
                from pq_hsa.index.batched_build import take_prebuilt

                _e28_per_head_codebooks = take_prebuilt(H, int(_ri.numel()))
                if _e28_per_head_codebooks is None:
                    _e28_per_head_codebooks = train_codebooks_batched(
                        flat_keys[:, _ri, :].contiguous(), self.index_config,
                        return_assignments=True,
                    )

        # --- Shared base buffer (device storage only) -------------------------
        # Allocate ONE [H, cap, D] buffer so all H streams share contiguous
        # storage.  This enables a single flat gather in the forward pass instead
        # of H separate Python-level index gathers.  Falls back to per-stream
        # allocation when kv_storage != "device".
        _growth_margin = 4096
        _cap = initial_len + _growth_margin
        # (opt-in PQ_HSA_SHARED_BASE_H1=1): TP=4 on a 4-KV-head model
        # (Qwen3-235B) leaves H=1 per rank; the H>1 gate below then skips the shared
        # base buffer, and restore_prefix()/the CUDA-graph tail (both assume it) fail
        # (`assert self._shared_base_keys is not None`).  With the flag
        # on, H=1 takes exactly the H>1 storage path; unset -> byte-identical to before.
        _h1_shared = os.environ.get("PQ_HSA_SHARED_BASE_H1", "0") == "1"
        _use_shared_base = (
            self.attention_config.kv_storage == "device"
            and (H > 1 or (H == 1 and _h1_shared))
            and flat_keys.is_cuda
        )
        if _use_shared_base:
            _base_keys = torch.empty(H, _cap, key_dim, dtype=flat_keys.dtype, device=flat_keys.device)
            _base_vals = torch.empty(H, _cap, val_dim, dtype=flat_values.dtype, device=flat_values.device)
            self._shared_base_keys = _base_keys  # [H, cap, D]
            self._shared_base_vals = _base_vals
            self._shared_base_cap = _cap
            self._shared_base_fallback = False  # set True if any stream overflows cap
        else:
            self._shared_base_keys = None
            self._shared_base_vals = None
            self._shared_base_cap = 0
            self._shared_base_fallback = False

        streams = []
        for h, (stream_keys, stream_values) in enumerate(
            zip(flat_keys, flat_values, strict=True)
        ):
            if _use_shared_base:
                key_buf_h = _base_keys[h]   # [cap, D]
                val_buf_h = _base_vals[h]   # [cap, D]
                streams.append(
                    IVFPQSparseAttention(self.index_config, self.attention_config).build_cache(
                        stream_keys,
                        stream_values,
                        shared_coarse_centroids=(
                            _e28_per_head_codebooks[0][h]
                            if _e28_per_head_codebooks is not None
                            else shared_coarse_centroids
                        ),
                        shared_pq_codebooks=(
                            _e28_per_head_codebooks[1][h]
                            if _e28_per_head_codebooks is not None
                            else shared_pq_codebooks
                        ),
                        _precomputed_list_ids=(
                            _e28_per_head_codebooks[2][h]
                            if _e28_per_head_codebooks is not None
                            else None
                        ),
                        _precomputed_codes=(
                            _e28_per_head_codebooks[3][h]
                            if _e28_per_head_codebooks is not None
                            else None
                        ),
                        _key_buf=key_buf_h,
                        _val_buf=val_buf_h,
                    )
                )
            else:
                streams.append(
                    IVFPQSparseAttention(self.index_config, self.attention_config).build_cache(
                        stream_keys,
                        stream_values,
                        shared_coarse_centroids=(
                            _e28_per_head_codebooks[0][h]
                            if _e28_per_head_codebooks is not None
                            else shared_coarse_centroids
                        ),
                        shared_pq_codebooks=(
                            _e28_per_head_codebooks[1][h]
                            if _e28_per_head_codebooks is not None
                            else shared_pq_codebooks
                        ),
                        _precomputed_list_ids=(
                            _e28_per_head_codebooks[2][h]
                            if _e28_per_head_codebooks is not None
                            else None
                        ),
                        _precomputed_codes=(
                            _e28_per_head_codebooks[3][h]
                            if _e28_per_head_codebooks is not None
                            else None
                        ),
                    )
                )
        if (
            _e28_per_head_codebooks is not None
            and os.environ.get("PQ_HSA_BATCHED_BUILD_KEEP_SHARED", "0") != "1"
        ):
            # Restore default rebuild semantics: later deferred flushes retrain
            # per head (as the default path does) instead of reusing the
            # prefill-time codebooks.
            for _s in streams:
                _s.cache.shared_coarse_centroids = None
                _s.cache.shared_pq_codebooks = None
        self._streams = tuple(streams)
        self._batched_heads = None
        self._shared_base_ring = False
        if (
            _use_shared_base
            and self._shared_base_keys is not None
            and _paged_attend_enabled()
        ):
            # The [H,cap,D] buffer just built above (cap ~= prompt
            # length + margin) was needed transiently to satisfy
            # SparseKVCache.__init__'s frozen invariant; replace it with a
            # small ring now that construction (and any initial index
            # training, which read the constructor's `keys`/`values`
            # arguments, not this buffer) is done. See _paged_ring_shrink.
            self._paged_ring_shrink(H, key_dim, val_dim, flat_keys.dtype, flat_keys.device)
        self._maybe_build_batched_heads()
        return self

    # --- Head-batched decode fast path -------------------------------------
    # Decode is launch-bound: the per-head Python loop in _forward_many issues
    # ~15-20 tiny CUDA kernels per (head, layer). Stacking all KV heads along a
    # head dim lets us score/topk/fetch/combine in single batched ops per layer,
    # cutting launch count ~num_kv_heads-fold. Gated to the steady-state config
    # this benchmark runs (single batch, hybrid + all_pq + centroid, no detail
    # collection, exact_rerank, Triton, 4-bit packed codes, uniform per-head
    # shapes). Falls back to the verified per-head loop otherwise.

    def _batched_heads_supported(self) -> bool:
        ac = self.attention_config
        if self._leading_shape is None or self._leading_shape[0] != 1:
            return False
        if ac.mode != "hybrid" or ac.hybrid_topk_source != "all_pq":
            return False
        if ac.hybrid_denominator_source != "all_pq" or ac.hybrid_value_mode != "centroid":
            return False
        if ac.collect_attention_details or not ac.exact_rerank:
            return False
        if ac.retrieval_top_p is not None or ac.retrieval_top_fraction is None:
            return False
        if not self._streams:
            return False
        first = self._streams[0].cache
        if first is None or first.index is None or first.retrieval_value_centroids is None:
            return False
        idx = first.index
        if not (idx.stores_packed_codes and idx.packed_codes is not None):
            return False
        if not (self.index_config.kernel_backend in {"auto", "triton", "h20"} and is_triton_available()):
            return False
        if not idx.packed_codes.is_cuda:
            return False
        # When direction_normalize, require key_norms available on CUDA for every stream.
        if self.index_config.direction_normalize:
            for s in self._streams:
                if s.cache is None or s.cache.index is None:
                    return False
                kn = s.cache.index.key_norms
                if kn is None or not kn.is_cuda:
                    return False
        # With rotation, codebooks/centroids live in rotated space and the
        # batched forward rotates queries once before LUT/list scoring. All
        # streams share one seeded rotation matrix; verify that holds.
        if getattr(self.index_config, "rotation", "none") not in (None, "none"):
            mats = [s.cache.index.rotation_matrix for s in self._streams]
            if any(m is None for m in mats):
                return False
            if any(not torch.equal(m, mats[0]) for m in mats[1:]):
                return False
        if not self.index_config.residual:
            return False
        # Uniform shapes across heads (required to stack into [H, ...]).
        ret_lens = {int(s.cache.regions.retrieval.numel()) for s in self._streams}
        if len(ret_lens) != 1 or next(iter(ret_lens)) == 0:
            return False
        list_counts = {int(s.cache.index.coarse_centroids.shape[0]) for s in self._streams}
        if len(list_counts) != 1:
            return False
        return True

    def _maybe_build_batched_heads(self) -> None:
        self._lean_resync_views()
        if not self._batched_heads_supported():
            self._batched_heads = None
            self._cg_invalidate()
            return
        prev_bh = self._batched_heads
        prev_cg = self._cg
        streams = self._streams
        H = len(streams)
        dev = streams[0].cache.index.packed_codes.device
        caches = [s.cache for s in streams]
        idxs = [c.index for c in caches]
        retrieval_idx = caches[0].regions.retrieval  # frozen during decode
        N = int(retrieval_idx.numel())
        # Per-head permutation that groups retrieval tokens by IVF list
        # (the CSR inverted-list order). Storing the snapshot in this order
        # makes the per-list background mass a contiguous segment sum
        # (cumsum + offset diff) instead of a large atomic scatter_add over N.
        perm = torch.stack(
            [_list_sort_perm(i, device=dev) for i in idxs], dim=0
        ).contiguous()  # [H, N]
        packed = torch.stack([i.packed_codes for i in idxs], dim=0)  # [H, N, W]
        packed = torch.gather(
            packed, 1, perm.unsqueeze(-1).expand(-1, -1, packed.shape[-1])
        ).contiguous()
        list_ids = torch.stack(
            [i.list_ids.to(dev, torch.long) for i in idxs], dim=0
        )
        list_ids = torch.gather(list_ids, 1, perm).contiguous()  # sorted by list
        # Snapshot per-head key norms for direction_normalize (list-sorted to match packed).
        # idx.key_norms is already retrieval-local ([N] over retrieval tokens, insertion order).
        if self.index_config.direction_normalize:
            knorm = torch.stack([i.key_norms.to(dev) for i in idxs], dim=0)  # [H, N]
            knorm = torch.gather(knorm, 1, perm).contiguous()  # same list-sort permutation
            assert knorm.shape == (H, N), f"knorm shape {knorm.shape} != ({H}, {N})"
        else:
            knorm = None
        # Snapshot the FROZEN retrieval-zone state. The retrieval region does not
        # change during decode (deferred index updates), so packed codes, list
        # ids, centroids and retrieval KV can be stacked once. The full region
        # (sink + local window) grows every step and is read live in the forward.
        self._batched_heads = {
            "H": H,
            "N": N,
            "packed": packed,
            "list_ids": list_ids,
            # Per-token key norms [H, N] (list-sorted), used when direction_normalize=True.
            # None when direction_normalize=False.
            "knorm": knorm,
            # Shared seeded rotation matrix (None when rotation="none"). The
            # forward rotates queries once before LUT / list-bias scoring;
            # full-region and exact-rerank logits keep the raw query.
            "rotation": (
                idxs[0].rotation_matrix.to(dev)
                if idxs[0].rotation_matrix is not None
                else None
            ),
            "coarse": torch.stack([i.coarse_centroids for i in idxs], dim=0).contiguous(),
            "codebooks": torch.stack([i.pq.codebooks for i in idxs], dim=0).contiguous(),
            "value_centroids": torch.stack(
                [c.retrieval_value_centroids.to(dev) for c in caches], dim=0
            ).contiguous(),
            # Exact-rerank keys/values are gathered live from the per-stream
            # caches in the forward (only the top-k ~fraction*N rows), so we do
            # NOT snapshot a full [H, N, D] copy of the retrieval KV here — that
            # duplicate (~32 GiB at 256K) OOMs on top of the sparse cache. We
            # keep the frozen retrieval-global indices to map local->global.
            # Because the snapshot is list-sorted per head, the local->global
            # map is per-head: [H, N].
            "retrieval_global": retrieval_idx.to(dev, torch.long)[perm].contiguous(),
            # Inverted lists per head (for candidate-pruned scoring at 256K).
            # Offsets [H, num_lists+1]; token order inside the snapshot already
            # IS the inverted-list order, so indices are the identity.
            "inv_offsets": torch.stack(
                [
                    _list_sort_offsets(i, device=dev, num_lists=int(idxs[0].coarse_centroids.shape[0]))
                    for i in idxs
                ],
                dim=0,
            ).contiguous(),
            "inv_indices": torch.arange(N, device=dev)
            .unsqueeze(0)
            .expand(H, N)
            .contiguous(),
            "retrieval_start": min(caches[0].sink_tokens, caches[0]._length),
            "num_lists": int(idxs[0].coarse_centroids.shape[0]),
            "subdim": idxs[0].pq.subdim,
            "M": self.index_config.num_subspaces,
            "indexed_retrieval_len": int(retrieval_idx.numel()),
            "nprobe": self.attention_config.nprobe,
            "candidate_budget": self.attention_config.candidate_budget,
            "block_size": (
                self.index_config.topk_block_size
                if self.index_config.topk_block_size is not None
                else 256
            ),
            # Shared-base flat-gather support.
            # When all H streams share a single [H, cap, D] backing buffer, the
            # per-head exact-rerank gather loop is replaced by one flat gather:
            #   flat_keys = base_keys.reshape(H*cap, D)
            #   idx = (exact_global + head_offsets.view(H,1,1)).reshape(-1)
            #   exact_keys = flat_keys[idx].view(H, G, topk, D)
            # "shared_base_keys/vals": references to the [H, cap, D] tensors.
            # "head_offsets": [H, 1, 1] long tensor of h * cap for each head.
            # "shared_base_cap": int capacity (may grow; set to -1 when fallback).
            "shared_base_keys": (
                self._shared_base_keys
                if (
                    getattr(self, "_shared_base_keys", None) is not None
                    and not getattr(self, "_shared_base_fallback", True)
                )
                else None
            ),
            "shared_base_vals": (
                self._shared_base_vals
                if (
                    getattr(self, "_shared_base_vals", None) is not None
                    and not getattr(self, "_shared_base_fallback", True)
                )
                else None
            ),
            "shared_base_cap": (
                getattr(self, "_shared_base_cap", 0)
                if not getattr(self, "_shared_base_fallback", True)
                else 0
            ),
            "head_offsets": (
                torch.arange(H, device=dev, dtype=torch.long).unsqueeze(-1).unsqueeze(-1)
                * (
                    getattr(self, "_shared_base_cap", 0)
                    if not getattr(self, "_shared_base_fallback", True)
                    else 0
                )
                if (
                    getattr(self, "_shared_base_keys", None) is not None
                    and not getattr(self, "_shared_base_fallback", True)
                )
                else None
            ),
            # Static approx_topk slot size (for Task 3c: avoid per-step .item() sync)
            "_static_slots": None,  # filled below if approx_topk supported
        }
        # Task 3c: precompute static slot size for _approx_topk path.
        # This is the max per-list size across all heads, frozen at snapshot time.
        num_lists_val = int(idxs[0].coarse_centroids.shape[0])
        inv_offsets_snap = self._batched_heads["inv_offsets"]  # [H, L+1]
        list_sizes = inv_offsets_snap[:, 1:] - inv_offsets_snap[:, :-1]  # [H, L]
        static_slots = int(list_sizes.max().item()) if list_sizes.numel() > 0 else 1
        self._batched_heads["_static_slots"] = max(1, static_slots)

        # B: refill static buffers and keep the graph when
        # the live retrieval length still fits in the captured N_CAP.
        nbh = self._batched_heads
        n_live = int(nbh["N"])
        slack = 0
        if os.environ.get("PQ_HSA_FLUSH_NO_RECAPTURE", "0") == "1":
            slack = int(os.environ.get("PQ_HSA_GRAPH_N_SLACK", "512"))
            self._pad_batched_heads_slack(nbh, slack)
        reused = False
        if (
            os.environ.get("PQ_HSA_REUSE_GRAPH", "1") == "1"
            and prev_cg is not None
            and prev_bh is not None
        ):
            cap = int(prev_bh.get("N_CAP") or prev_bh["N"])
            same_exact = (
                slack == 0
                and int(prev_bh["N"]) == n_live
                and tuple(prev_bh["packed"].shape) == tuple(nbh["packed"].shape)
                and tuple(prev_bh["list_ids"].shape) == tuple(nbh["list_ids"].shape)
            )
            slack_ok = slack > 0 and n_live <= cap and int(prev_bh["H"]) == int(nbh["H"])
            if same_exact or slack_ok:
                self._inplace_refresh_snapshot(prev_bh, nbh, n_live)
                self._batched_heads = prev_bh
                self._cg = prev_cg
                sink_end = min(int(caches[0].sink_tokens), int(caches[0]._length))
                self._cg_refill_buffers(
                    prev_cg["full_k"], prev_cg["full_v"], prev_cg["mask"], sink_end
                )
                local_count = int(caches[0].regions.local.numel())
                prev_cg["valid_full"] = min(
                    int(prev_cg["sink_end"]) + local_count, int(prev_cg["FULL_CAP"])
                )
                reused = True
                try:
                    from benchmarks.vllm_backend.pq_hsa_decode_runtime import STATS as _ST

                    _ST["pq_norecapture_flush"] = int(_ST.get("pq_norecapture_flush") or 0) + 1
                except Exception:
                    pass
        if reused:
            return
        try:
            from benchmarks.vllm_backend.pq_hsa_decode_runtime import STATS as _ST

            _ST["pq_recapture_flush"] = int(_ST.get("pq_recapture_flush") or 0) + 1
        except Exception:
            pass
        self._cg_invalidate()

    def _pad_batched_heads_slack(self, bh: dict, slack: int) -> None:
        """Pad snapshot tensors to N+slack so a later flush can copy_ in-place."""
        if slack <= 0 or bh.get("N_CAP") is not None:
            return
        N = int(bh["N"])
        N_CAP = N + slack
        H = int(bh["H"])
        dev = bh["packed"].device
        W = int(bh["packed"].shape[-1])
        packed = bh["packed"].new_zeros(H, N_CAP, W)
        packed[:, :N].copy_(bh["packed"])
        bh["packed"] = packed
        lids = bh["list_ids"].new_zeros(H, N_CAP)
        lids[:, :N].copy_(bh["list_ids"])
        bh["list_ids"] = lids
        if bh.get("knorm") is not None:
            kn = torch.ones(H, N_CAP, device=dev, dtype=bh["knorm"].dtype)
            kn[:, :N].copy_(bh["knorm"])
            bh["knorm"] = kn
        rg = bh["retrieval_global"].new_zeros(H, N_CAP)
        rg[:, :N].copy_(bh["retrieval_global"])
        bh["retrieval_global"] = rg
        bh["inv_indices"] = (
            torch.arange(N_CAP, device=dev, dtype=torch.long).unsqueeze(0).expand(H, N_CAP).contiguous()
        )
        rm = torch.zeros(H, 1, N_CAP, device=dev, dtype=torch.float16)
        rm[:, :, N:] = float("-inf")
        bh["ret_mask"] = rm
        bh["N_live"] = N
        bh["N_CAP"] = N_CAP
        bh["N"] = N_CAP

    def _inplace_refresh_snapshot(self, dst: dict, src: dict, n_live: int) -> None:
        """Copy a freshly built (possibly unpadded) snapshot into padded dst buffers."""
        def _row(dst_t, src_t):
            if dst_t is None or src_t is None:
                return
            n = min(n_live, int(src_t.shape[1]), int(dst_t.shape[1]))
            dst_t[:, :n].copy_(src_t[:, :n])

        _row(dst.get("packed"), src.get("packed"))
        _row(dst.get("list_ids"), src.get("list_ids"))
        _row(dst.get("retrieval_global"), src.get("retrieval_global"))
        if dst.get("knorm") is not None and src.get("knorm") is not None:
            _row(dst["knorm"], src["knorm"])
        for key in ("coarse", "codebooks", "value_centroids", "inv_offsets"):
            d, s = dst.get(key), src.get(key)
            if d is not None and s is not None and tuple(d.shape) == tuple(s.shape):
                d.copy_(s)
        rm = dst.get("ret_mask")
        if rm is not None:
            rm.zero_()
            rm[:, :, n_live:] = float("-inf")
        dst["N_live"] = n_live
        dst["indexed_retrieval_len"] = n_live
        dst["_static_slots"] = src.get("_static_slots")

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self._lean_resync_views()
        self._check_built()
        assert self._leading_shape is not None
        assert self._key_dim is not None
        assert self._value_dim is not None
        expected_key_shape = (*self._leading_shape, self._key_dim)
        expected_value_shape = (*self._leading_shape, self._value_dim)
        if tuple(keys.shape) != expected_key_shape:
            raise ValueError(f"keys must have shape {expected_key_shape}, got {tuple(keys.shape)}")
        if tuple(values.shape) != expected_value_shape:
            raise ValueError(
                f"values must have shape {expected_value_shape}, got {tuple(values.shape)}"
            )

        flat_keys = keys.reshape(-1, self._key_dim)
        flat_values = values.reshape(-1, self._value_dim)
        H = len(self._streams)

        # Fast vectorized path: when all H streams share a [H, cap, D] base buffer
        # and none will overflow, write all H rows with ONE tensor assign then
        # call _notify_external_append per stream (bookkeeping only, no buf writes).
        use_shared = (
            (H > 1 or (H == 1 and os.environ.get("PQ_HSA_SHARED_BASE_H1", "0") == "1"))  # Opt-in
            and not getattr(self, "_shared_base_fallback", True)
            and self._shared_base_keys is not None
            and self._shared_base_vals is not None
        )
        _ring = getattr(self, "_shared_base_ring", False)
        if use_shared and not _ring:
            # Check cap overflow before writing — if any stream would exceed the
            # shared buffer capacity, fall back to the per-stream path.
            # Skipped entirely in ring mode -- the ring buffer is
            # small BY DESIGN and never "grows"; positions wrap instead
            # (see _paged_ring_pos / build_cache's _paged_ring_shrink).
            first_cache = self._streams[0].cache
            if first_cache is not None and first_cache._length >= self._shared_base_cap:
                self._shared_base_fallback = True
                use_shared = False
                if self._batched_heads is not None:
                    self._batched_heads["shared_base_keys"] = None
                    self._batched_heads["shared_base_vals"] = None
                    self._batched_heads["head_offsets"] = None

        if use_shared:
            pos = self._streams[0].cache._length  # all streams share the same pos
            wpos = self._paged_ring_pos(pos)
            if not getattr(self, "_fast_append", True) or not self._fast_bookkeep_append():
                self._shared_base_keys[:, wpos, :] = flat_keys   # type: ignore[index]
                self._shared_base_vals[:, wpos, :] = flat_values  # type: ignore[index]
                for stream in self._streams:
                    stream.cache._notify_external_append()
                self._cg_update_after_append()
            else:
                # One fused scatter into shared-base + graph buffers
                # (was 5 launches: 2 shared writes + 2 full copies + mask).
                cg = self._cg
                # D: defer scatter until replay so append+attend share one
                # launch burst. Bookkeeping still happens here.
                if (
                    os.environ.get("PQ_HSA_GRAPH_APPEND", "0") == "1"
                    and cg is not None
                    and int(cg["valid_full"]) < int(cg["FULL_CAP"])
                ):
                    self._ga_k = flat_keys
                    self._ga_v = flat_values
                    self._ga_pos = wpos
                    self._ga_spos = int(cg["valid_full"])
                    cg["valid_full"] = int(cg["valid_full"]) + 1
                elif (
                    cg is not None
                    and int(cg["valid_full"]) < int(cg["FULL_CAP"])
                    and _t83e_append_scatter(
                        self._shared_base_keys,
                        self._shared_base_vals,
                        cg["full_k"],
                        cg["full_v"],
                        cg["mask"],
                        flat_keys,
                        flat_values,
                        wpos,
                        int(cg["valid_full"]),
                    )
                ):
                    cg["valid_full"] = int(cg["valid_full"]) + 1
                else:
                    self._shared_base_keys[:, wpos, :] = flat_keys   # type: ignore[index]
                    self._shared_base_vals[:, wpos, :] = flat_values  # type: ignore[index]
                    self._cg_update_after_append()
        else:
            # Normal path: per-stream append with full buffer write + bookkeeping.
            # Check before appending whether any stream will overflow the shared cap.
            if (
                not getattr(self, "_shared_base_fallback", True)
                and self._shared_base_keys is not None
                and not _ring
            ):
                for stream in self._streams:
                    if stream.cache is not None and stream.cache._length >= self._shared_base_cap:
                        self._shared_base_fallback = True
                        if self._batched_heads is not None:
                            self._batched_heads["shared_base_keys"] = None
                            self._batched_heads["shared_base_vals"] = None
                            self._batched_heads["head_offsets"] = None
                        break
            for stream, key, value in zip(self._streams, flat_keys, flat_values, strict=True):
                stream.append(key, value)

    # ------------------------------------------------------------------
    # Lean steady-decode append (opt-in, PQ_HSA_LEAN_APPEND=1).
    #
    # The default append() costs ~84 us of HOST time per layer per token at
    # 128K: ~32 us shape validation / reshape,
    # ~34 us for the Triton scatter launch and ~17 us of bookkeeping whose bulk
    # is 16 `buf[:n]` reslices.  The compute-stream Event span around it is
    # therefore mostly GPU idle waiting for the host.  This path
    #   * replaces the Triton scatter AND the eager `cg["q"].copy_` with ONE
    #     custom-CUDA launch (pq_append_prep, 4.2 us host / 1.1 us device),
    #   * drops the shape validation (shapes are re-checked once, on entry),
    #   * defers the public keys/values reslices to the next rare event
    #     (Delta flush / snapshot rebuild / checkpoint), which resyncs them.
    # Data movement is bit-identical with the path it replaces.
    # ------------------------------------------------------------------
    def _lean_append_fn(self):
        fn = getattr(self, "_lean_ap_fn", "?")
        if fn != "?":
            return fn
        fn = None
        if os.environ.get("PQ_HSA_LEAN_APPEND", "0") == "1":
            try:
                from pq_hsa.kernels.cuda.pq_fused_h20 import (
                    append_prep, append_prep_available,
                )

                if append_prep_available():
                    fn = append_prep
            except Exception:
                fn = None
        self._lean_ap_fn = fn
        return fn

    def _lean_resync_views(self) -> None:
        """Re-expose the public keys/values views after a lean append run."""
        if not getattr(self, "_lean_views_dirty", False):
            return
        self._lean_views_dirty = False
        for stream in self._streams:
            c = stream.cache
            if c is None:
                continue
            c.keys = c._key_buf[: c._length]
            c.values = c._val_buf[: c._length]

    def lean_append(self, keys, values, q_src=None) -> bool:
        """One-launch steady append. Returns False -> caller runs append()."""
        ap = self._lean_append_fn()
        if ap is None:
            return False
        cg = self._cg
        if cg is None or getattr(self, "_shared_base_fallback", True):
            return False
        if self._shared_base_keys is None or self._shared_base_vals is None:
            return False
        caches = self._bk_caches
        if caches is None:
            caches = tuple(s.cache for s in self._streams)
            self._bk_caches = caches
        c0 = caches[0]
        pos = c0._length
        _ring = getattr(self, "_shared_base_ring", False)
        if not _ring and pos >= self._shared_base_cap:
            return False
        wpos = self._paged_ring_pos(pos)
        pending = c0._pending_update_tokens + 1
        if pending >= c0.index_update_interval:
            # Delta boundary: keep the original per-stream path byte-for-byte.
            return False
        spos = cg["valid_full"]
        if spos >= cg["FULL_CAP"]:
            return False
        if not (keys.is_contiguous() and values.is_contiguous()):
            return False
        q_dst = None
        if q_src is not None and q_src.is_contiguous():
            q_dst = cg["q"]
        else:
            q_src = None
        try:
            ap(keys, values, self._shared_base_keys, self._shared_base_vals,
               cg["full_k"], cg["full_v"], cg["mask"], wpos, spos, q_src, q_dst)
        except Exception:
            return False
        cg["valid_full"] = spos + 1
        new_len = pos + 1
        sink_end = c0._cached_sink_end
        retrieval_end = c0._cached_retrieval_end
        if sink_end < 0 or retrieval_end < 0:
            sink_end = min(c0.sink_tokens, new_len)
            indexed_length = min(c0._indexed_length, new_len)
            if c0.local_window > 0:
                retrieval_end = max(sink_end, indexed_length - c0.local_window)
            else:
                retrieval_end = max(sink_end, indexed_length)
        ar = getattr(self, "_regions_arange", None)
        regions = self._bk_regions
        if (
            ar is not None
            and ar.numel() >= new_len
            and regions is not None
            and regions.sink.data_ptr() == ar.data_ptr()
        ):
            regions.local = ar[retrieval_end:new_len]
        else:
            regions = self._shared_regions(sink_end, retrieval_end, new_len,
                                           c0.index_device)
            self._bk_regions = regions
        for c in caches:
            c._length = new_len
            c._pending_update_tokens = pending
            c.regions = regions
        self._lean_views_dirty = True
        if q_dst is not None:
            self._q_staged_cg = cg
        return True

    def _shared_regions(self, sink_end: int, retrieval_end: int, length: int, device):
        """Region tensors as slices of one cached arange (zero kernel launches).

        Values are identical to ``SparseKVCache._make_regions`` output.
        """
        from pq_hsa.attention.kv_cache import KVCacheRegions

        ar = getattr(self, "_regions_arange", None)
        if ar is None or ar.numel() < length or ar.device != device:
            cap = max(length + 8192, int(getattr(self, "_shared_base_cap", 0)))
            ar = torch.arange(cap, device=device, dtype=torch.long)
            self._regions_arange = ar
        return KVCacheRegions(
            sink=ar[:sink_end],
            retrieval=ar[sink_end:retrieval_end],
            local=ar[retrieval_end:length],
        )

    def _fast_bookkeep_append(self) -> bool:
        """Steady-decode bookkeeping for a shared-base append, batched over heads.

        Equivalent to calling ``cache._notify_external_append()`` on every
        stream, but builds ONE shared ``KVCacheRegions`` (slices of a cached
        arange, so zero kernel launches) and assigns it to all H caches.
        Returns False when the slow per-stream path must run instead (interval
        boundary reached -> deferred-flush marking, or non-uniform stream
        state).
        """
        caches = self._bk_caches
        if caches is None:
            caches = tuple(s.cache for s in self._streams)
            self._bk_caches = caches
        c0 = caches[0]
        new_len = c0._length + 1
        pending = c0._pending_update_tokens + 1
        if pending >= c0.index_update_interval:
            # Rare (once per Delta): keep the original per-stream path so the
            # deferred-update marking and counters stay byte-for-byte the same.
            return False

        # Shared-base path keeps heads uniform; skip the per-token recheck.
        sink_end = c0._cached_sink_end
        retrieval_end = c0._cached_retrieval_end
        if sink_end < 0 or retrieval_end < 0:
            sink_end = min(c0.sink_tokens, new_len)
            indexed_length = min(c0._indexed_length, new_len)
            if c0.local_window > 0:
                retrieval_end = max(sink_end, indexed_length - c0.local_window)
            else:
                retrieval_end = max(sink_end, indexed_length)

        ar = getattr(self, "_regions_arange", None)
        regions = self._bk_regions
        if (
            ar is not None
            and ar.numel() >= new_len
            and regions is not None
            and regions.sink.data_ptr() == ar.data_ptr()
        ):
            regions.local = ar[retrieval_end:new_len]
        else:
            regions = self._shared_regions(sink_end, retrieval_end, new_len, c0.index_device)
            self._bk_regions = regions
        for c in caches:
            c._length = new_len
            c.keys = c._key_buf[:new_len]
            c.values = c._val_buf[:new_len]
            c._pending_update_tokens = pending
            c.regions = regions
            c._cached_length = new_len
            c._cached_regions = regions
        return True

    def forward(self, queries: torch.Tensor) -> _GroupedQueryDecodeOutput:
        query_states, had_query_len = _normalize_gqa_queries(queries)
        result = self._forward_many(query_states)
        context = result.context
        if not had_query_len:
            context = context.squeeze(-2)
        return _GroupedQueryDecodeOutput(context=context, per_query=result.per_query)

    def decode_step(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
        *,
        append_kv: bool = True,
    ) -> _GroupedQueryDecodeOutput:
        if append_kv:
            if keys is None or values is None:
                raise ValueError("keys and values are required when append_kv=True")
            _t12_mark("index.append")
            self.append(_squeeze_gqa_decode_token(keys, "keys"), _squeeze_gqa_decode_token(values, "values"))
        return self.forward(queries)

    def _forward_many(self, queries: torch.Tensor) -> _GroupedQueryDecodeOutput:
        self._check_built()
        assert self._leading_shape is not None
        assert self._value_dim is not None
        if queries.ndim != 4:
            raise ValueError("queries must be [batch, query_heads, query_len, dim]")
        batch, query_heads, query_len, key_dim = queries.shape
        if batch != self._leading_shape[0] or query_heads != self.num_query_heads:
            raise ValueError(
                f"queries must be [batch={self._leading_shape[0]}, "
                f"query_heads={self.num_query_heads}, query_len, dim], got {tuple(queries.shape)}"
            )
        if key_dim != self._key_dim:
            raise ValueError(f"query dim must be {self._key_dim}, got {key_dim}")

        if (
            self._batched_heads is not None
            and batch == 1
            and query_len == 1
        ):
            if (_E21_TRUNC_ALLPQ or _E21_BG_UNSELECTED_MEAN) and (
                getattr(self, "_candidate_pruned", False) or self._use_cuda_graph
            ):
                raise RuntimeError(
                    "[variant-T] PQ_HSA_E21_TRUNC_ALLPQ is implemented only on the eager "
                    "batched-heads path; candidate-pruned / CUDA-graph decode refused"
                )
            if getattr(self, "_candidate_pruned", False):
                batched = self._forward_many_batched_heads_candidate(queries)
                if batched is not None:
                    return batched
            # Try CUDA-graph replay first, then fall back to eager batched path.
            if self._use_cuda_graph and not getattr(self, "_candidate_pruned", False):
                graph_result = self._forward_many_batched_heads_graph(queries)
                if graph_result is not None:
                    if _E42_REUSE:
                        _E42_STATS["graph_path_calls_unreused"] += 1
                        if not _E42_PRINTED["graph_note"]:
                            _E42_PRINTED["graph_note"] = True
                            print("[REUSE] NOTE: CUDA-graph replay path active; eager reuse NOT applied on this path", flush=True)
                    return graph_result
            batched = self._forward_many_batched_heads(queries)
            if batched is not None:
                return batched

        if (_E21_TRUNC_ALLPQ or _E21_BG_UNSELECTED_MEAN) and query_len == 1:
            # Same policy as the truncation variants: a per-head fallback step would be
            # silently hybrid, so it is a hard failure, never a quiet mix.
            raise RuntimeError(
                "[variant-T] per-head fallback reached under PQ_HSA_E21_TRUNC_ALLPQ=1 "
                f"(batched_heads={'set' if self._batched_heads is not None else 'None'}); refusing"
            )
        context = torch.empty(
            batch,
            query_heads,
            query_len,
            self._value_dim,
            device=queries.device,
            dtype=queries.dtype,
        )
        per_query_streams: list[list[Any]] = [[] for _ in range(query_len)]
        for batch_idx in range(batch):
            for kv_head in range(self.num_key_value_heads):
                stream = self._streams[batch_idx * self.num_key_value_heads + kv_head]
                q_start = kv_head * self.num_key_value_groups
                q_end = q_start + self.num_key_value_groups
                group_queries = queries[batch_idx, q_start:q_end].transpose(0, 1).reshape(
                    query_len * self.num_key_value_groups,
                    key_dim,
                )
                group_outputs = stream.forward_many(group_queries)
                for query_pos in range(query_len):
                    row_start = query_pos * self.num_key_value_groups
                    row_end = row_start + self.num_key_value_groups
                    current_outputs = group_outputs[row_start:row_end]
                    current_context = torch.stack(
                        [item.output for item in current_outputs],
                        dim=0,
                    )
                    context[batch_idx, q_start:q_end, query_pos, :] = current_context
                    per_query_streams[query_pos].extend(current_outputs)

        return _GroupedQueryDecodeOutput(
            context=context,
            per_query=tuple(
                _GroupedQueryAttentionOutput(
                    output=context[:, :, query_pos, :],
                    per_stream=tuple(per_query_streams[query_pos]),
                )
                for query_pos in range(query_len)
            ),
        )

    def _scan_retrieval_logits(
        self,
        bh: dict,
        lut: torch.Tensor,
        list_scores: torch.Tensor,
        *,
        fused_row_max: bool,
        pair_tables: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Multihead LUT scan. On H20, optionally fold block max into row_max.

        The fused Hopper kernel still materializes ``[H,G,N]`` scores (needed
        for ``topk`` and per-list background mass). It only removes the later
        ``amax(dim=2)`` over retrieval tokens.
        """
        from pq_hsa.kernels.device import resolve_lut_backend

        backend = self.index_config.kernel_backend
        packed = bh["packed"]
        resolved = resolve_lut_backend(backend, packed.device)
        if fused_row_max and resolved == "h20":
            from pq_hsa.kernels.triton_lut_scan_h20 import (
                reduce_block_online_softmax,
                score_packed_4bit_lut_multihead_fused_stats_h20,
            )

            scores, block_max, block_expsum = (
                score_packed_4bit_lut_multihead_fused_stats_h20(
                    packed,
                    lut,
                    bh["list_ids"],
                    list_scores,
                    num_subspaces=bh["M"],
                    block_size=bh["block_size"],
                    token_scale=bh["knorm"],
                    pair_tables=pair_tables,
                )
            )
            # Fold the per-block stats in one CUDA launch instead of the
            # 5 aten kernels behind amax + exp + mul + sum. Default OFF.
            if os.environ.get("PQ_HSA_FUSED_BLOCKRED", "0") == "1":
                try:
                    from pq_hsa.kernels.cuda.pq_fused_h20 import block_reduce as _cu_block_reduce

                    row_max, exp_sum = _cu_block_reduce(block_max, block_expsum)
                    return scores, row_max, exp_sum
                except Exception as _br_exc:
                    self._block_reduce_error = f"{type(_br_exc).__name__}: {_br_exc}"
            row_max, exp_sum = reduce_block_online_softmax(block_max, block_expsum)
            return scores, row_max, exp_sum
        scores = score_packed_4bit_lut_multihead_list_bias(
            packed,
            lut,
            bh["list_ids"],
            list_scores,
            num_subspaces=bh["M"],
            block_size=bh["block_size"],
            token_scale=bh["knorm"],
            backend=backend,
        )
        return scores, None, None

    # ------------------------------------------------------------------
    # (opt-in, PQ_HSA_OFFLOAD_FAST=1): fast offload gather for the
    # non-shared-base (kv_storage="cpu") branches of
    # `_forward_many_batched_heads`.
    # ------------------------------------------------------------------

    def _fast_offload_full_gather(
        self,
        H: int,
        sink_end: int,
        local_start: int,
        local_count: int,
        full_count: int,
        D: int,
        dev: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sink+local read for the offload (kv_storage='cpu') path. The
        default branch this replaces already does ONE combined `.to(dev)`
        call (not per-head), but the source is a freshly-`cat`'d, unpinned
        CPU tensor, so that single H2D is still a silent blocking copy. This
        copies the same per-head slices into a reused PINNED staging buffer
        first so the H2D is a real non-blocking transfer.
        """
        vdim = self._streams[0].cache.values.shape[-1]
        dtype = self._streams[0].cache.keys.dtype
        buf_shape_k = (H, full_count, D)
        buf_shape_v = (H, full_count, vdim)
        need_alloc = (
            getattr(self, "_fast_full_key_pin", None) is None
            or tuple(self._fast_full_key_pin.shape) != buf_shape_k
            or self._fast_full_key_pin.dtype != dtype
        )
        if need_alloc:
            pin = torch.cuda.is_available()
            key_pin = torch.empty(buf_shape_k, dtype=dtype)
            val_pin = torch.empty(buf_shape_v, dtype=dtype)
            if pin:
                key_pin = key_pin.pin_memory()
                val_pin = val_pin.pin_memory()
            self._fast_full_key_pin = key_pin
            self._fast_full_val_pin = val_pin

        key_pin = self._fast_full_key_pin
        val_pin = self._fast_full_val_pin
        for h in range(H):
            ck = self._streams[h].cache.keys
            cv = self._streams[h].cache.values
            key_pin[h, :sink_end].copy_(ck[:sink_end])
            key_pin[h, sink_end:full_count].copy_(ck[local_start : local_start + local_count])
            val_pin[h, :sink_end].copy_(cv[:sink_end])
            val_pin[h, sink_end:full_count].copy_(cv[local_start : local_start + local_count])

        is_cuda = dev.type == "cuda"
        full_keys = key_pin.to(dev, non_blocking=is_cuda)
        full_values = val_pin.to(dev, non_blocking=is_cuda)
        self._fast_offload_full_gather_calls = getattr(self, "_fast_offload_full_gather_calls", 0) + 1
        self._fast_offload_full_gather_rows = (
            getattr(self, "_fast_offload_full_gather_rows", 0) + H * full_count
        )
        return full_keys, full_values

    def _fast_offload_exact_gather(
        self,
        exact_global: torch.Tensor,
        H: int,
        G: int,
        topk: int,
        D: int,
        vdim: int,
        dev: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """PQ-retrieval exact-rerank read for the offload (kv_storage='cpu')
        path -- THE per-layer, per-decode-step gather that profiling traced
        the 5.9 s/token offload bottleneck to: the
        replaced loop did H separate CPU fancy-index + blocking `.to(dev)`
        calls (H = num_key_value_heads), once per layer, every decode step,
        with no batching across heads and no pinned staging.

        Fix: one combined D2H of every head's indices (instead of H), H
        cheap CPU-only `index_select` calls into a reused pinned staging
        buffer (no device sync per call), then ONE non-blocking H2D
        transfer for all H heads together.
        """
        if topk <= 0:
            dtype = self._streams[0].cache.keys.dtype
            empty_k = torch.empty(H, G, 0, D, device=dev, dtype=dtype)
            empty_v = torch.empty(H, G, 0, vdim, device=dev, dtype=dtype)
            return empty_k, empty_v

        dtype = self._streams[0].cache.keys.dtype
        buf_shape_k = (H, G, topk, D)
        buf_shape_v = (H, G, topk, vdim)
        need_alloc = (
            getattr(self, "_fast_exact_key_pin", None) is None
            or tuple(self._fast_exact_key_pin.shape) != buf_shape_k
            or self._fast_exact_key_pin.dtype != dtype
        )
        if need_alloc:
            pin = torch.cuda.is_available()
            key_pin = torch.empty(buf_shape_k, dtype=dtype)
            val_pin = torch.empty(buf_shape_v, dtype=dtype)
            if pin:
                key_pin = key_pin.pin_memory()
                val_pin = val_pin.pin_memory()
            self._fast_exact_key_pin = key_pin
            self._fast_exact_val_pin = val_pin

        key_pin = self._fast_exact_key_pin
        val_pin = self._fast_exact_val_pin

        # One combined D2H for every head's indices instead of H separate ones.
        exact_global_cpu = exact_global.to(device="cpu", dtype=torch.long)
        for h in range(H):
            ck = self._streams[h].cache.keys
            cv = self._streams[h].cache.values
            gh = exact_global_cpu[h].reshape(-1)
            torch.index_select(ck, 0, gh, out=key_pin[h].view(G * topk, D))
            torch.index_select(cv, 0, gh, out=val_pin[h].view(G * topk, vdim))

        is_cuda = dev.type == "cuda"
        exact_keys = key_pin.to(dev, non_blocking=is_cuda)
        exact_values = val_pin.to(dev, non_blocking=is_cuda)
        self._fast_offload_exact_gather_calls = getattr(self, "_fast_offload_exact_gather_calls", 0) + 1
        self._fast_offload_exact_gather_rows = (
            getattr(self, "_fast_offload_exact_gather_rows", 0) + H * G * topk
        )
        return exact_keys, exact_values

    def _forward_many_batched_heads(
        self,
        queries: torch.Tensor,
    ) -> _GroupedQueryDecodeOutput | None:
        """Head-batched decode for the steady-state hybrid/centroid/all_pq config.

        queries: [1, num_query_heads, 1, D]. Returns None to fall back to the
        per-head loop if the live cache state no longer matches the snapshot
        (e.g. an index refresh changed the retrieval region mid-decode).
        """
        bh = self._batched_heads
        H = bh["H"]
        G = self.num_key_value_groups
        D = self._key_dim
        dev = bh["packed"].device
        dtype = queries.dtype

        first_cache = self._streams[0].cache
        # Snapshot is only valid while the retrieval region stays frozen.
        if int(first_cache.regions.retrieval.numel()) != bh["indexed_retrieval_len"]:
            # Task 3a: try to rebuild the snapshot from the new index state so the
            # batched path resumes on the NEXT step (correctness: fall back this step).
            self._maybe_build_batched_heads()
            return None
        N = bh["N"]
        # Task 3b: retrieval_start was read from the snapshot but never used in
        # this function (the mapping goes via bh["retrieval_global"] instead).
        # Removed to avoid a dead local that confused profiling.
        # Live full region (sink + local window), identical index layout per head.
        # Regions are contiguous ranges (sink = [0, sink_end), local =
        # [length - local_count, length)), so the full-region KV is read via
        # slice views + one cat instead of per-head index gathers.
        regions = first_cache.regions
        sink_end = int(regions.sink.numel())
        local_count = int(regions.local.numel())
        local_start = int(first_cache._length) - local_count
        full_count = sink_end + local_count

        scale = 1.0 / sqrt(D)
        M = bh["M"]
        subdim = bh["subdim"]

        # queries -> [H, G, D]
        q = queries[0, :, 0, :].reshape(H, G, D).to(dev)
        # PQ codebooks / coarse centroids live in rotated space; rotate the
        # query once for PQ-domain scoring. Full-region and exact-rerank
        # logits below use the raw q (they read raw keys).
        q_pq = q if bh["rotation"] is None else q.matmul(bh["rotation"].to(dtype))


        # --- (opt-in): cross-step retrieval reuse decision ---------------
        _use_approx_topk = getattr(self, "_approx_topk", False)
        _e42_on = (
            _E42_REUSE
            and not _use_approx_topk
            and os.environ.get("PQ_HSA_GROUP_SHARED_RETRIEVAL", "0") != "1"
            and not (_E21_C2_MASS or os.environ.get("PQ_HSA_E21_C2_MASS", "0") == "1")
        )
        _e42_reuse = False
        _e42_st = getattr(self, "_e42_state", None)
        _e42_topk = min(max(self._streams[0]._resolve_retrieval_topk(N), 0), N)
        _e42_qn = None
        _e42_ev0 = None
        if _e42_on:
            if not _E42_PRINTED["first"]:
                _E42_PRINTED["first"] = True
                print(f"[REUSE] cross-step retrieval reuse active on eager batched-heads path "
                      f"(tau={_E42_TAU}, r_max={_E42_RMAX}, agg={_E42_AGG}, variant=all-or-nothing-per-layer)", flush=True)
            if _E42_TIMING:
                _e42_ev0 = torch.cuda.Event(enable_timing=True)
                _e42_ev1 = torch.cuda.Event(enable_timing=True)
                _e42_ev0.record()
            _E42_STATS["calls"] += 1
            _e42_qn = torch.nn.functional.normalize(q.float(), dim=-1)  # [H, G, D]
            if _e42_st is None:
                _E42_STATS["first_retrievals"] += 1
            elif (_e42_st["bh"] is not bh) or (_e42_st["N"] != N) or (_e42_st["topk"] != _e42_topk):
                _E42_STATS["forced_invalidate"] += 1
            elif _e42_st["age"] >= _E42_RMAX:
                _E42_STATS["forced_rmax"] += 1
            else:
                _e42_cos = (_e42_qn * _e42_st["qn"]).sum(-1)  # [H, G] per-row query drift
                if _E42_AGG == "mean":
                    _e42_cos_min = float(_e42_cos.mean())
                elif _E42_AGG.startswith("q"):
                    # q0.9 -> reuse if >= 90% of the rows have cos >= tau
                    _e42_cos_min = float(torch.quantile(_e42_cos.flatten(), 1.0 - float(_E42_AGG[1:])))
                else:
                    _e42_cos_min = float(_e42_cos.min())
                _E42_STATS["cos_min_sum"] += _e42_cos_min
                _E42_STATS["cos_min_n"] += 1
                _E42_STATS["cos_rowmean_sum"] += float(_e42_cos.mean())
                if _e42_cos_min >= _E42_TAU:
                    _e42_reuse = True
                else:
                    _E42_STATS["forced_drift"] += 1
            if _E42_STATS["calls"] % 4096 == 0:
                print(_e42_summary_line(), flush=True)
                _e42_export_stats()
        if _e42_reuse:
            # Skip steps 1-4: re-use the stored exact top-k set and stored PQ
            # background quantities (all computed from the stored query).
            _E42_STATS["reuse_hits"] += 1
            _e42_st["age"] += 1
            _gsr_eager = False
            _c2_on = False
            _c2_anchor_mode = "mean"
            num_lists = bh["num_lists"]
            list_max_f32 = None
            list_expsum_f32 = None
            topk = _e42_topk
            approx_logits = None
            _approx_dtype = _e42_st["approx_dtype"]
            exact_local = _e42_st["exact_local"]
            exact_approx_logits = _e42_st["exact_approx_logits"]
            fused_retrieval_row_max = _e42_st["ret_row_max"]
            fused_retrieval_exp_sum = None
        else:
            _approx_dtype = None

            _t12_mark("pq.lut_build")
            # --- 1) One LUT for all heads/groups: [H, G, M, 16] ------------------
            # codebooks [H, M, K, subdim]; q reshaped to [H, G, M, subdim].
            codebooks = bh["codebooks"]
            q_sub = q_pq.reshape(H, G, M, subdim)
            # (opt-in, default OFF): group-shared retrieval on the EAGER
            # path -- the same semantics as the branch in _cg_forward_static
            # (scan / top-k / list-mass from the group-mean query, Gr=1 rows, then
            # broadcast to the G query heads; exact top-k attention and sink+local
            # keep each head's own q).  Until this branch existed, GSR only lived in
            # the CUDA-graph body, so any HF-harness run whose capture failed
            # (8B/30B) silently evaluated plain per-head retrieval.
            _gsr_eager = os.environ.get("PQ_HSA_GROUP_SHARED_RETRIEVAL", "0") == "1" and G > 1
            if _gsr_eager and getattr(self, "_approx_topk", False):
                raise NotImplementedError(
                    "PQ_HSA_GROUP_SHARED_RETRIEVAL=1 is not supported together with _approx_topk (Phase 2)"
                )
            if _gsr_eager:
                q_pq_r = q_pq.mean(dim=1, keepdim=True)          # [H, 1, D]
                q_sub_r = q_pq_r.reshape(H, 1, M, subdim)
            else:
                q_pq_r, q_sub_r = q_pq, q_sub
            # lut[h,g,m,k] = sum_d q_sub[h,g,m,d] * codebooks[h,m,k,d]
            # The attention scale is folded into the LUT and list bias so the scan
            # output is already in logit units (saves one [H,G,N] elementwise op).
            lut, list_scores = _lut_and_list_scores(
                q_sub_r, codebooks, q_pq_r, bh["coarse"], scale, dtype
            )

            # --- 3) One multihead scan over frozen retrieval codes: [H, G, N] ----
            # Default path: H20 fused scan also returns retrieval row_max (block
            # reduce), so we skip the later amax over N. Phase-2 approx top-k
            # still needs per-list max, so it keeps the plain scan.
            _use_approx_topk = getattr(self, "_approx_topk", False)
            _t12_mark("pq.lut_scan")
            approx_logits, fused_retrieval_row_max, fused_retrieval_exp_sum = self._scan_retrieval_logits(
                bh, lut, list_scores, fused_row_max=not _use_approx_topk
            )  # [H, G, N], already scaled

            # --- 3b) Per-list stats -------------------------------------------
            # When _approx_topk=True (Phase 2) we need per-list max to select
            # the top-nprobe lists, so we run the Welford single-pass kernel that
            # computes (list_max, list_expsum_rel_local_max) simultaneously.
            # When _approx_topk=False (Phase 1 / default) we skip per-list max
            # and instead compute the global row_max cheaply via amax(dim=2), then
            # call the faster list_exp_sums kernel (which only needs expsum given
            # a known row_max).  This avoids the +56µs overhead of the Welford
            # kernel in the common-case (exact-only) path.
            import triton as _triton
            num_lists = bh["num_lists"]
            _avg_list_len = max(1, N // num_lists)
            _stats_block_size = int(_triton.next_power_of_2(min(max(_avg_list_len, 32), 512)))
            if _use_approx_topk:
                list_max_f32, list_expsum_f32 = list_stats_sorted_multihead_triton(
                    approx_logits,
                    bh["inv_offsets"],
                    num_lists=num_lists,
                    block_size=_stats_block_size,
                )  # [H,G,L] fp32 each; list_expsum relative to per-list local max
                # list_max_f32 is used in step 4 to select top-nprobe lists.
            else:
                list_max_f32 = None  # computed later from approx_logits.amax

            # --- 4) top-k exact tokens (batched over H*G) -----------------------
            _t12_mark("pq.topk")
            topk = self._streams[0]._resolve_retrieval_topk(N)
            topk = min(max(topk, 0), N)
            if topk > 0:
                if getattr(self, "_approx_topk", False):
                    # Phase 2: IVF-list-pruned approximate top-k.
                    # Use per-list max scores (from Phase 1 list_max_f32) to select
                    # the top-nprobe lists per (h,g). Only score tokens within those
                    # lists (nprobe × avg_list_size tokens << N). Then aten::topk on
                    # this reduced candidate set (~16K at 256K, 16x less than N).
                    # Recall >= 0.85 since list_max correlates with token quality.
                    _nprobe = min(bh["nprobe"] if bh["nprobe"] is not None else num_lists, num_lists)
                    # list_max_f32 [H,G,L]: use to select top-nprobe lists per (h,g)
                    _, _probe_list_ids = torch.topk(
                        list_max_f32, k=_nprobe, dim=2, sorted=False
                    )  # [H,G,nprobe]
                    # Gather token ranges for probed lists using inv_offsets [H,L+1]
                    _inv_off = bh["inv_offsets"]  # [H, L+1]
                    _off_hg = _inv_off.unsqueeze(1).expand(H, G, num_lists + 1)
                    _p_starts = torch.gather(_off_hg, 2, _probe_list_ids)      # [H,G,nprobe]
                    _p_ends = torch.gather(_off_hg, 2, _probe_list_ids + 1)    # [H,G,nprobe]
                    _p_sizes = _p_ends - _p_starts                             # [H,G,nprobe]
                    # Task 3c: use precomputed static slot size (frozen at snapshot build
                    # time from inv_offsets) to avoid a per-step GPU->CPU sync (.item()).
                    _slots = bh.get("_static_slots") or max(1, int(_p_sizes.max().item()))
                    # Build candidate positions: [H,G,nprobe,slots]
                    _slot_ar = torch.arange(_slots, device=dev)
                    _pos = _p_starts.unsqueeze(-1) + _slot_ar                  # [H,G,nprobe,slots]
                    _valid = _slot_ar < _p_sizes.unsqueeze(-1)                 # [H,G,nprobe,slots]
                    _pos = torch.where(_valid, _pos, torch.zeros_like(_pos))
                    _C = _nprobe * _slots
                    # Gather approx logit scores for probed tokens [H,G,C]
                    _pos_flat = _pos.reshape(H, G, _C)
                    _logits_hgn = approx_logits  # [H, G, N]
                    _cand_logits = torch.gather(_logits_hgn, 2, _pos_flat)    # [H,G,C]
                    _valid_c = _valid.reshape(H, G, _C)
                    _cand_logits = torch.where(_valid_c, _cand_logits,
                        torch.full_like(_cand_logits, float("-inf")))
                    # topk within candidate set
                    _topk_c = min(topk, _C)
                    _top_cand_logits, _top_cand_pos = torch.topk(
                        _cand_logits, k=_topk_c, dim=2, sorted=False
                    )  # [H,G,topk]
                    exact_local = torch.gather(_pos_flat, 2, _top_cand_pos)   # [H,G,topk]
                    exact_approx_logits = _top_cand_logits
                else:
                    # Selection only feeds set-membership and elementwise exp
                    # corrections, so the top-k does not need to be sorted.
                    exact_approx_logits, exact_local = _retrieval_exact_topk(
                        approx_logits, topk
                    )  # [H, G, topk] local retrieval indices
            else:
                exact_local = torch.empty(H, G, 0, device=dev, dtype=torch.long)
                exact_approx_logits = torch.empty(H, G, 0, device=dev, dtype=approx_logits.dtype)

            if _gsr_eager:
                # Broadcast the per-KV-head (Gr=1) retrieval result to the
                # G query heads so everything downstream keeps its [H, G, *] layout.
                approx_logits = approx_logits.expand(H, G, N).contiguous()
                if fused_retrieval_row_max is not None:
                    fused_retrieval_row_max = fused_retrieval_row_max.expand(H, G).contiguous()
                if fused_retrieval_exp_sum is not None:
                    fused_retrieval_exp_sum = fused_retrieval_exp_sum.expand(H, G).contiguous()
                if exact_local.shape[1] != G:
                    exact_local = exact_local.expand(H, G, -1).contiguous()
                    exact_approx_logits = exact_approx_logits.expand(H, G, -1).contiguous()
                _gsr_hits = getattr(self, "_gsr_eager_hits", 0) + 1
                self._gsr_eager_hits = _gsr_hits
                if _gsr_hits == 1:
                    print("[GSR-EAGER] group-shared retrieval active on eager batched-heads path", flush=True)

            # --- Arm C2 (opt-in, PQ_HSA_E21_C2_MASS=1; default path untouched) ---
            # Neutralize per-token / per-list PQ information in the background
            # mass and denominator AFTER exact selection: every retrieval-region
            # logit is replaced by one per-row constant (mean PQ score of the
            # already-selected exact set), so background mass = count * exp(const)
            # and the exact-set correction uses the same constant. Selection
            # (exact_local) is unchanged. This is the batched-heads counterpart of
            # run_ablation_arm._install_pqfree_mass_patch(), which only wraps the
            # sparse_attention fallback path and is never reached here.
            #-C2b: PQ_HSA_E21_C2_ANCHOR picks WHICH per-row constant is used --
            # "mean" (default) is the original C2 anchor and is applied right here;
            # "boundary" is arm C2b and is applied further down, once exact_logits
            # (the true rerank scores) exist. Nothing between the two sites reads
            # approx_logits / exact_approx_logits / the fused retrieval stats, so
            # the two placements are equivalent.
            _c2_on = _E21_C2_MASS or os.environ.get("PQ_HSA_E21_C2_MASS", "0") == "1"  # read lazily: run_ablation_arm sets the env after import
            _c2_anchor_mode = "mean"
            if _c2_on:
                _c2_anchor_mode = (
                    os.environ.get("PQ_HSA_E21_C2_ANCHOR", _E21_C2_ANCHOR) or "mean"
                ).strip().lower()
                if _c2_anchor_mode not in ("mean", "boundary"):
                    raise RuntimeError(
                        f"PQ_HSA_E21_C2_ANCHOR={_c2_anchor_mode!r} not understood; expected "
                        "'mean' (default, original C2 arm) or 'boundary' (arm C2b)"
                    )
                if _use_approx_topk:
                    raise RuntimeError("PQ_HSA_E21_C2_MASS=1 is not supported together with _approx_topk (Phase 2)")
            if _c2_on and _c2_anchor_mode == "mean":
                if topk > 0:
                    _c2_anchor = exact_approx_logits.float().mean(dim=2, keepdim=True)
                else:
                    _c2_anchor = approx_logits.float().mean(dim=2, keepdim=True)
                _c2_anchor = _c2_anchor.to(approx_logits.dtype)
                approx_logits = _c2_anchor.expand_as(approx_logits).contiguous()
                if topk > 0:
                    exact_approx_logits = _c2_anchor.expand_as(exact_approx_logits).contiguous()
                fused_retrieval_row_max = None
                fused_retrieval_exp_sum = None
                list_max_f32 = None
                _c2_hits = getattr(self, "_e21_c2_hits", 0) + 1
                self._e21_c2_hits = _c2_hits
                if _c2_hits == 1:
                    print("[variant-C2] PQ-free background mass active on batched-heads path", flush=True)

            _approx_dtype = approx_logits.dtype
        # --- 5) full-region logits (live): [H, G, full_count] ---------------
        _t12_mark("pq.exact_gather_rerank")
        # full keys/values per head, read live from the growing cache.
        # Fast path: when all streams share a [H, cap, D] base buffer, we can
        # read sink/local via direct 2-D slices (no Python loop, no per-head cat).
        # Fallback: per-head slice views + one cat over dim=0.
        _sb_keys = bh.get("shared_base_keys")
        _sb_vals = bh.get("shared_base_vals")
        _paged_kv = (
            (_paged_kv_enabled() or getattr(self, "_shared_base_ring", False))
            and self._paged_kv_cache is not None
        )
        if os.environ.get("PQ_HSA_PAGED_KV_DEBUG", "0") == "1":
            print(f"[paged-kv DEBUG] enter _forward_many_batched_heads paged_kv_enabled="
                  f"{_paged_kv_enabled()} cache_set={self._paged_kv_cache is not None} "
                  f"_paged_kv={_paged_kv} sink_end={sink_end} local_start={local_start} "
                  f"local_count={local_count} full_count={full_count}", flush=True)
        if _paged_kv and full_count > 0:
            # Sink + local window read straight off vLLM's paged KV via
            # block_table (two small contiguous ranges), instead of slicing
            # the sidecar's own resident buffer.
            _sink_k, _sink_v = gather_flashattn_kv_range(
                self._paged_kv_cache, self._paged_attn_metadata, self._paged_seq_idx,
                0, sink_end,
            )
            _local_k, _local_v = gather_flashattn_kv_range(
                self._paged_kv_cache, self._paged_attn_metadata, self._paged_seq_idx,
                local_start, local_start + local_count,
            )
            full_keys = torch.cat([_sink_k, _local_k], dim=1).to(dtype)   # [H, full_count, D]
            full_values = torch.cat([_sink_v, _local_v], dim=1).to(dtype)  # [H, full_count, vdim_]
            if os.environ.get("PQ_HSA_PAGED_KV_DEBUG", "0") == "1" and _sb_keys is not None:
                _ref_sink_k = _sb_keys[:, :sink_end, :]
                _ref_local_k = _sb_keys[:, local_start:local_start + local_count, :]
                _ref_full_keys = torch.cat([_ref_sink_k, _ref_local_k], dim=1)
                _ref_sink_v = _sb_vals[:, :sink_end, :]
                _ref_local_v = _sb_vals[:, local_start:local_start + local_count, :]
                _ref_full_values = torch.cat([_ref_sink_v, _ref_local_v], dim=1)
                _dk = (full_keys.float() - _ref_full_keys.float()).abs().max().item()
                _dv = (full_values.float() - _ref_full_values.float()).abs().max().item()
                print(f"[paged-kv DEBUG] full_k/v raw diff: dk={_dk:.3e} dv={_dv:.3e} "
                      f"sink_end={sink_end} local_start={local_start} local_count={local_count} "
                      f"seq_len={int(first_cache._length)}",
                      flush=True)
                if _dk > 1e-6:
                    from benchmarks.vllm_backend.paged_kv_fa import gather_flashattn_kv as _gfkv
                    _fk_full, _fv_full, _sl = _gfkv(self._paged_kv_cache, self._paged_attn_metadata, self._paged_seq_idx)
                    _fk_full = _fk_full[0]  # [H, seq_len, D]
                    _fv_full = _fv_full[0]
                    _ref_sink_full = _fk_full[:, :sink_end, :]
                    _ref_local_full = _fk_full[:, local_start:local_start + local_count, :]
                    _full_via_old = torch.cat([_ref_sink_full, _ref_local_full], dim=1)
                    _d_old_vs_shared = (_full_via_old.float() - _ref_full_keys.float()).abs().amax(dim=(0, 2))
                    _d_old_vs_new = (_full_via_old.float() - full_keys.float()).abs().amax(dim=(0, 2))
                    _bad_old_vs_shared = (_d_old_vs_shared > 1e-6).nonzero().reshape(-1).tolist()
                    _bad_old_vs_new = (_d_old_vs_new > 1e-6).nonzero().reshape(-1).tolist()
                    print(f"[paged-kv DEBUG3] old-trusted-gather vs shared_base bad_rows={_bad_old_vs_shared[:10]} "
                          f"old-trusted-gather vs new-range-gather bad_rows={_bad_old_vs_new[:10]} seq_len_old={_sl}",
                          flush=True)

                    _row_diff = (full_keys.float() - _ref_full_keys.float()).abs().amax(dim=(0, 2))
                    _bad_rows = (_row_diff > 1e-6).nonzero().reshape(-1).tolist()
                    print(f"[paged-kv DEBUG]   bad local rows (offset within full_count)={_bad_rows[:10]} "
                          f"of {full_count} (token_idx = local_start-sink_end+row if row>=sink_end)",
                          flush=True)
        elif _sb_keys is not None and full_count > 0:
            # Shared base: extract sink and local as 3-D slices, then cat on dim=1.
            _sink_k = _sb_keys[:, :sink_end, :]           # [H, sink_end, D]
            _local_k = _sb_keys[:, local_start:local_start + local_count, :]  # [H, local, D]
            full_keys = torch.cat([_sink_k, _local_k], dim=1)   # [H, full_count, D]
            _sink_v = _sb_vals[:, :sink_end, :]
            _local_v = _sb_vals[:, local_start:local_start + local_count, :]
            full_values = torch.cat([_sink_v, _local_v], dim=1)  # [H, full_count, vdim_]
        elif full_count > 0:
            if (
                os.environ.get("PQ_HSA_OFFLOAD_FAST", "0") == "1"
                and self._streams[0].cache.kv_storage == "cpu"
            ):
                # (opt-in): the plain path below concatenates fresh
                # (unpinned) per-head CPU slices and does one `.to(dev)` with
                # no non_blocking -- correct but the H2D copy is a silent
                # blocking transfer from unpinned memory. Route the same
                # per-head slices into a reused PINNED staging buffer so the
                # H2D is a real non-blocking transfer.
                full_keys, full_values = self._fast_offload_full_gather(
                    H, sink_end, local_start, local_count, full_count, D, dev,
                )
            else:
                key_views = []
                value_views = []
                for h in range(H):
                    ck = self._streams[h].cache.keys
                    cv = self._streams[h].cache.values
                    key_views.append(ck[:sink_end])
                    key_views.append(ck[local_start : local_start + local_count])
                    value_views.append(cv[:sink_end])
                    value_views.append(cv[local_start : local_start + local_count])
                full_keys = torch.cat(key_views, dim=0).to(dev).reshape(H, full_count, D)
                full_values = torch.cat(value_views, dim=0).to(dev).reshape(H, full_count, -1)
        else:
            full_keys = torch.empty(H, 0, D, device=dev, dtype=dtype)
            full_values = torch.empty(H, 0, self._value_dim, device=dev, dtype=dtype)
        full_logits = torch.einsum("hgd,hfd->hgf", q, full_keys) * scale  # [H,G,full]

        # --- 6) exact rerank: gather top-k retrieval keys/values LIVE from the
        # per-stream caches (only topk rows/head, no full [H,N,D] duplicate) ---
        vdim = full_values.shape[-1]
        if topk > 0:
            ret_global = bh["retrieval_global"]  # [H, N] retrieval-local -> global
            # exact_local [H,G,topk] -> global token ids per head
            exact_global = torch.gather(
                ret_global.unsqueeze(1).expand(H, G, N), 2, exact_local
            )  # [H,G,topk]
            _head_offsets = bh.get("head_offsets")
            if _paged_kv:
                # Exact-rerank candidate gather straight off vLLM's
                # paged KV via block_table -- no flat [H*cap, D] shared-base
                # buffer read. exact_global is already global token ids
                # ([H,G,topk], values in [0, seq_len)), matching
                # gather_flashattn_kv_at_indices' expected leading-dim-H layout.
                exact_keys, exact_values = gather_flashattn_kv_at_indices(
                    self._paged_kv_cache, self._paged_attn_metadata,
                    self._paged_seq_idx, exact_global,
                )
                exact_keys = exact_keys.to(dtype)
                exact_values = exact_values.to(dtype)
                if os.environ.get("PQ_HSA_PAGED_KV_DEBUG", "0") == "1" and _sb_keys is not None and _head_offsets is not None:
                    _cap_dbg = bh["shared_base_cap"]
                    _flat_k_dbg = _sb_keys.reshape(H * _cap_dbg, D)
                    _flat_v_dbg = _sb_vals.reshape(H * _cap_dbg, vdim)
                    _flat_idx_dbg = (exact_global + _head_offsets).to(dtype=torch.int32).reshape(-1)
                    _ref_ek = _flat_k_dbg[_flat_idx_dbg].reshape(H, G, topk, D)
                    _ref_ev = _flat_v_dbg[_flat_idx_dbg].reshape(H, G, topk, vdim)
                    _dek = (exact_keys.float() - _ref_ek.float()).abs().max().item()
                    _dev = (exact_values.float() - _ref_ev.float()).abs().max().item()
                    print(f"[paged-kv DEBUG] exact_k/v raw diff: dek={_dek:.3e} dev={_dev:.3e} topk={topk}",
                          flush=True)
            elif _sb_keys is not None and _head_offsets is not None:
                # Flat gather: one index into H*cap instead of H per-head gathers.
                _cap = bh["shared_base_cap"]
                flat_k = _sb_keys.reshape(H * _cap, D)   # [H*cap, D]
                flat_v = _sb_vals.reshape(H * _cap, vdim) # [H*cap, vdim]
                # exact_global [H,G,topk] + head_offsets [H,1,1] -> [H,G,topk]
                # int32 index for the large flat gather (low-hanging).
                flat_idx = (exact_global + _head_offsets).to(dtype=torch.int32).reshape(-1)
                exact_keys = flat_k[flat_idx].reshape(H, G, topk, D)
                exact_values = flat_v[flat_idx].reshape(H, G, topk, vdim)
            elif (
                os.environ.get("PQ_HSA_OFFLOAD_FAST", "0") == "1"
                and self._streams[0].cache.kv_storage == "cpu"
            ):
                # (opt-in): this is the ACTUAL per-layer, per-decode-step
                # offload gather (kv_storage="cpu" -> _sb_keys is None, so
                # neither the paged-KV nor shared-base-flat-gather branches
                # above apply) -- the 5.9 s/token offload bottleneck
                # traces to exactly this loop: H separate CPU
                # fancy-index + blocking `.to(dev)` calls (H = num_key_value_
                # heads), once per layer, every decode step.
                # pq_hsa/attention/kv_cache.py's SparseKVCache.fetch_kv fast
                # path (also gated by PQ_HSA_OFFLOAD_FAST=1) is NOT reached
                # here -- share_gqa_kv_cache=True decode uses this head-batched
                # adapter method directly, bypassing IVFPQSparseAttention.
                # forward/_forward_hybrid entirely. Fix: one combined D2H of
                # all heads' indices, H cheap CPU-only index_selects into a
                # reused pinned staging buffer, one non-blocking H2D for all
                # H heads together.
                exact_keys, exact_values = self._fast_offload_exact_gather(
                    exact_global, H, G, topk, D, vdim, dev,
                )
            else:
                exact_keys_list = []
                exact_values_list = []
                for h in range(H):
                    ck = self._streams[h].cache.keys
                    cv = self._streams[h].cache.values
                    gh = exact_global[h].reshape(-1).to(ck.device)  # [G*topk]
                    exact_keys_list.append(ck[gh].to(dev).reshape(G, topk, D))
                    exact_values_list.append(cv[gh].to(dev).reshape(G, topk, vdim))
                exact_keys = torch.stack(exact_keys_list, dim=0)      # [H,G,topk,D]
                exact_values = torch.stack(exact_values_list, dim=0)  # [H,G,topk,vdim]
            # Batched matvec instead of mul+sum avoids materializing a
            # [H,G,topk,D] product tensor.
            exact_logits = (
                torch.matmul(exact_keys, q.unsqueeze(-1)).squeeze(-1) * scale
            )  # [H,G,topk]
        else:
            exact_keys = torch.empty(H, G, 0, D, device=dev, dtype=dtype)
            exact_values = torch.empty(H, G, 0, vdim, device=dev, dtype=dtype)
            exact_logits = torch.empty(H, G, 0, device=dev, dtype=_approx_dtype)

        # --- Arm C2b (opt-in, PQ_HSA_E21_C2_ANCHOR=boundary) ------------
        # Exactly the transformation of the C2 "mean" block above (same
        # broadcast/expand/dtype path, same cleared fused stats, selection
        # untouched), but the per-row constant is the MINIMUM EXACT rerank
        # logit over the already-selected set -- "every unselected token
        # scores as badly as the worst token we kept". That anchor reads no PQ
        # score at all (the mean anchor uses the exact set's PQ
        # approximations), and since the selected set holds the top-scoring
        # tokens it is the tightest PQ-free upper bound on the background
        # mass, where the mean anchor over-estimates it by construction.
        if _c2_on and _c2_anchor_mode == "boundary":
            if topk > 0:
                _c2_anchor = exact_logits.float().min(dim=2, keepdim=True).values
            else:
                # topk==0 (arm D only) has no selected set; fall back to the
                # tightest available constant over the retrieval region.
                _c2_anchor = approx_logits.float().min(dim=2, keepdim=True).values
            _c2_anchor = _c2_anchor.to(approx_logits.dtype)
            approx_logits = _c2_anchor.expand_as(approx_logits).contiguous()
            if topk > 0:
                exact_approx_logits = _c2_anchor.expand_as(exact_approx_logits).contiguous()
            fused_retrieval_row_max = None
            fused_retrieval_exp_sum = None
            list_max_f32 = None
            _c2_hits = getattr(self, "_e21_c2_hits", 0) + 1
            self._e21_c2_hits = _c2_hits
            if _c2_hits == 1:
                print("[variant-C2:boundary] PQ-free background mass "
                      "(anchor = min exact rerank logit of the selected set) "
                      "active on batched-heads path", flush=True)

        # --- 7) centroid hybrid softmax (batched over [H,G]) ----------------
        _t12_mark("pq.list_stats_denom")
        if _use_approx_topk:
            # Phase 2 path: list_max_f32 already computed (Welford kernel).
            # Derive retrieval row_max from per-list maxes (O(L) not O(N)).
            retrieval_row_max = list_max_f32.amax(dim=2).to(full_logits.dtype)
        elif fused_retrieval_row_max is not None:
            retrieval_row_max = fused_retrieval_row_max.to(full_logits.dtype)
        else:
            # Default path (Phase 1 or no approx): fast O(N) global max.
            retrieval_row_max = approx_logits.amax(dim=2).to(full_logits.dtype)

        # row_max over full + retrieval + exact.
        row_max = torch.maximum(
            full_logits.max(dim=2).values,
            retrieval_row_max,
        )  # [H,G] in working dtype
        if topk > 0:
            row_max = torch.maximum(row_max, exact_logits.max(dim=2).values)

        full_exp = torch.exp(full_logits - row_max.unsqueeze(-1))
        full_exp_sum = full_exp.sum(dim=2)  # [H,G]

        if _use_approx_topk:
            # Phase 2: background_mass[h,g,l] = list_expsum_f32[h,g,l] *
            # exp(list_max_f32[h,g,l] - row_max[h,g])  (two-pass trick).
            background_mass_f32 = list_expsum_f32 * torch.exp(
                list_max_f32 - row_max.float().unsqueeze(-1)
            )  # [H,G,L] fp32
        elif _e42_reuse:
            # Stored per-list PQ background mass (relative to the stored joint
            # row max) rescaled to the CURRENT joint row max (online-softmax trick).
            background_mass_f32 = _e42_st["bg_ref"] * torch.exp(
                _e42_st["ref_max"] - row_max.float()
            ).unsqueeze(-1)
        else:
            # Default path: compute per-list expsum directly given row_max.
            # list_exp_sums_sorted_multihead_triton returns sum(exp(x-row_max))
            # per list, which is already in the right form for background_mass.
            background_mass_f32 = list_exp_sums_sorted_multihead_triton(
                approx_logits,
                row_max,
                bh["inv_offsets"],
                num_lists=num_lists,
            )  # [H,G,L] float32

        if fused_retrieval_exp_sum is not None:
            # Online-softmax rescale: fused exp_sum is relative to retrieval
            # row_max, while denom uses the joint max over full + retrieval + exact.
            retrieval_exp_sum = (
                fused_retrieval_exp_sum
                * torch.exp(
                    fused_retrieval_row_max.float() - row_max.float()
                )
            ).to(full_exp.dtype)
        else:
            retrieval_exp_sum = background_mass_f32.sum(dim=2).to(full_exp.dtype)

        if _e42_on and not _e42_reuse:
            # Remember this retrieval for the following steps.
            _E42_STATS["retrievals"] += 1
            if _E42_JACCARD and _e42_st is not None and _e42_st["bh"] is bh and topk > 0 \
                    and _e42_st["exact_local"].shape == exact_local.shape:
                # Diagnostic: how much does the exact top-k SET change between two
                # consecutive retrievals (per row Jaccard, mean over [H,G])?  This is
                # the quantity that bounds what any set-reuse scheme can achieve.
                _old_mask = torch.zeros(H, G, N, device=dev, dtype=torch.bool)
                _old_mask.scatter_(2, _e42_st["exact_local"], True)
                _inter = torch.gather(_old_mask, 2, exact_local).sum(dim=2).float()  # [H,G]
                _jac = _inter / (2.0 * topk - _inter)
                _E42_STATS["jaccard_sum"] += float(_jac.mean())
                _E42_STATS["jaccard_min_sum"] += float(_jac.min())
                _E42_STATS["jaccard_n"] += 1
                # per-row picture: what fraction of [H,G] rows would a row-level scheme reuse?
                _cos_rows = (_e42_qn * _e42_st["qn"]).sum(-1)
                _E42_STATS["rows_cos_ge_tau_sum"] += float((_cos_rows >= _E42_TAU_DIAG).float().mean())
                _E42_STATS["rows_jac_ge_09_sum"] += float((_jac >= 0.9).float().mean())
                _E42_STATS["rows_jac_ge_08_sum"] += float((_jac >= 0.8).float().mean())
                _E42_STATS["rows_both_sum"] += float(((_cos_rows >= _E42_TAU_DIAG) & (_jac >= 0.9)).float().mean())
                if _e42_st["age"] == 0:
                    _E42_STATS["jaccard_consec_sum"] += float(_jac.mean())
                    _E42_STATS["jaccard_consec_n"] += 1
            self._e42_state = {
                "bh": bh, "N": N, "topk": topk, "age": 0, "qn": _e42_qn,
                "exact_local": exact_local, "exact_approx_logits": exact_approx_logits,
                "ret_row_max": retrieval_row_max.clone(),
                "bg_ref": background_mass_f32.clone(),
                "ref_max": row_max.float().clone(),
                "approx_dtype": _approx_dtype,
            }
        if topk > 0:
            old_exact_exp = torch.exp(exact_approx_logits - row_max.unsqueeze(-1))
            exact_exp = torch.exp(exact_logits - row_max.unsqueeze(-1))
            exact_exp_sum = exact_exp.sum(dim=2)
        else:
            old_exact_exp = torch.zeros(H, G, 0, device=dev, dtype=full_exp.dtype)
            exact_exp = old_exact_exp
            exact_exp_sum = torch.zeros(H, G, device=dev, dtype=full_exp.dtype)

        denom = full_exp_sum + retrieval_exp_sum - old_exact_exp.sum(dim=2) + exact_exp_sum
        denom = denom.clamp_min(torch.finfo(denom.dtype).tiny)  # [H,G]

        _t12_mark("pq.hybrid_softmax")
        # full-region output: [H,G,D]
        full_output = torch.einsum(
            "hgf,hfd->hgd", full_exp / denom.unsqueeze(-1), full_values
        )
        if topk > 0:
            exact_weights = exact_exp / denom.unsqueeze(-1)  # [H,G,topk]
            exact_output = torch.matmul(
                exact_weights.unsqueeze(2).to(exact_values.dtype), exact_values
            ).squeeze(2)  # [H,G,vdim]
        else:
            exact_output = torch.zeros(H, G, vdim, device=dev, dtype=dtype)

        # background (centroid) mass per IVF list, minus the exact tokens'
        # approximate contribution (those are recomputed exactly above).
        list_ids = bh["list_ids"]  # [H, N], sorted
        background_mass = background_mass_f32.to(full_exp.dtype)
        if topk > 0:
            exact_list_ids = torch.gather(
                list_ids.unsqueeze(1).expand(H, G, N), 2, exact_local
            )  # [H,G,topk]
            background_mass.scatter_add_(2, exact_list_ids, -old_exact_exp)
        if _E21_BGMASS_LOG or os.environ.get("PQ_HSA_E21_BGMASS_LOG", "0") == "1":
            _e21_bgmass_report(self, background_mass, exact_exp_sum, full_exp_sum, denom)
        background_weight = background_mass / denom.unsqueeze(-1)  # [H,G,num_lists]
        background_output = torch.einsum(
            "hgl,hld->hgd", background_weight, bh["value_centroids"]
        )

        output = full_output + exact_output + background_output  # [H,G,D]
        # --- Arm T (opt-in, PQ_HSA_E21_TRUNC_ALLPQ=1; default path untouched) ---
        if _E21_T2_DIAG:
            _e21_t2_diag(
                self, bh, q, scale, approx_logits, row_max, exact_local, exact_logits,
                exact_approx_logits, full_logits, denom, background_mass, first_cache, topk,
            )
        if int(_E21_TRUNC_ALLPQ) + int(_E21_BG_UNSELECTED_MEAN) > 1:
            raise RuntimeError("[variants] arms T / Bx are mutually exclusive")
        if _E21_BG_UNSELECTED_MEAN:
            output = _e21_bx_epilogue(
                self, output, full_output, exact_output, bh, exact_local, exact_values,
                background_weight,
            )
        elif _E21_TRUNC_ALLPQ or _E21_PROBE:
            output = _e21_trunc_epilogue(
                self, output, full_logits, full_values, exact_logits, exact_values,
                exact_global if topk > 0 else None, background_mass, denom,
            )

        _t12_mark("pq.attn_end")
        # --- 8) pack context [1, num_query_heads, 1, vdim] ------------------
        context = output.reshape(1, H * G, 1, vdim).to(dtype=dtype)

        # Lightweight per-stream metadata so attention stats stay populated.
        denominator_count = full_count + N
        outputs = []
        empty_long = torch.empty(0, device=dev, dtype=torch.long)
        empty_score = torch.empty(0, device=dev, dtype=_approx_dtype)
        for _ in range(H * G):
            outputs.append(
                AttentionOutput(
                    output=None,
                    indices=empty_long,
                    logits=empty_score,
                    weights=empty_score,
                    approx_retrieval_logits=empty_score,
                    exact_retrieval_logits=None,
                    search_result=None,
                    denominator_indices=None,
                    denominator_logits=None,
                    denominator_weights=None,
                    selected_count=int(full_count + topk),
                    denominator_count=int(denominator_count),
                    approx_retrieval_count=int(N),
                    exact_retrieval_count=int(topk),
                    candidate_count=int(N),
                    probed_list_count=0,
                )
            )
        if _e42_ev0 is not None:
            _e42_ev1.record()
            _e42_ev1.synchronize()
            _e42_ms = _e42_ev0.elapsed_time(_e42_ev1)
            if _e42_reuse:
                _E42_STATS["t_reuse_ms"] += _e42_ms
                _E42_STATS["n_reuse_timed"] += 1
            else:
                _E42_STATS["t_retr_ms"] += _e42_ms
                _E42_STATS["n_retr_timed"] += 1
        return _GroupedQueryDecodeOutput(
            context=context,
            per_query=(
                _GroupedQueryAttentionOutput(
                    output=context[:, :, 0, :],
                    per_stream=tuple(outputs),
                ),
            ),
        )

    def _forward_many_batched_heads_candidate(
        self,
        queries: torch.Tensor,
    ) -> _GroupedQueryDecodeOutput | None:
        """Candidate-pruned head-batched decode for very long context (256K+).

        Instead of scanning all N retrieval tokens (O(N), the bottleneck that
        makes the full-scan path scale linearly with context), this scores only
        the tokens in the nprobe IVF lists each query probes, capped to a fixed
        candidate budget C. Cost is O(H*G*C), decoupled from context length.

        The softmax denominator is approximate: it covers the probed candidates
        exactly plus a per-IVF-list centroid background mass over ALL lists
        (reusing value_centroids), so unprobed lists still contribute background
        mass via their centroid — this keeps the denominator close to the full
        all_pq denominator without an O(N) scan.

        queries: [1, num_query_heads, 1, D]. Returns None to fall back.
        """
        # direction_normalize is not supported on the candidate path (PyTorch gather
        # loop does not apply key_norms; let the full batched-scan path handle it).
        if self.index_config.direction_normalize:
            return None
        bh = self._batched_heads
        H = bh["H"]
        G = self.num_key_value_groups
        D = self._key_dim
        dev = bh["packed"].device
        dtype = queries.dtype

        first_cache = self._streams[0].cache
        if int(first_cache.regions.retrieval.numel()) != bh["indexed_retrieval_len"]:
            # Task 3a: rebuild snapshot so the batched path resumes next step.
            self._maybe_build_batched_heads()
            return None
        N = bh["N"]
        num_lists = bh["num_lists"]
        M = bh["M"]
        subdim = bh["subdim"]
        scale = 1.0 / sqrt(D)

        nprobe = bh["nprobe"] if bh["nprobe"] is not None else num_lists
        nprobe = min(max(int(nprobe), 1), num_lists)
        budget = bh["candidate_budget"]
        C = min(int(budget) if budget is not None else N, N)
        if C <= 0:
            return None

        regions = first_cache.regions
        q = queries[0, :, 0, :].reshape(H, G, D).to(dev)

        # --- 1) LUT [H,G,M,16] and coarse list scores [H,G,num_lists] -------
        codebooks = bh["codebooks"]
        q_pq = q if bh["rotation"] is None else q.matmul(bh["rotation"].to(q.dtype))
        q_sub = q_pq.reshape(H, G, M, subdim)
        if _pq_hsa_fp16_lut():
            cb = codebooks if codebooks.dtype == q_sub.dtype else codebooks.to(dtype=q_sub.dtype)
            cs = bh["coarse"] if bh["coarse"].dtype == q_pq.dtype else bh["coarse"].to(dtype=q_pq.dtype)
            lut = torch.einsum("hgmd,hmkd->hgmk", q_sub, cb)
            list_scores = torch.einsum("hgd,hld->hgl", q_pq, cs)
        else:
            lut = torch.einsum("hgmd,hmkd->hgmk", q_sub.float(), codebooks.float())  # f32
            list_scores = torch.einsum("hgd,hld->hgl", q_pq.float(), bh["coarse"].float())  # [H,G,L]

        # --- 2) probe top-nprobe lists per (h,g) ----------------------------
        probed = torch.topk(list_scores, k=nprobe, dim=2).indices  # [H,G,nprobe]

        # --- 3) gather a fixed C candidates from the probed lists -----------
        # Per-list fixed slot budget so the candidate tensor is regular [H,G,C].
        slots = max(1, C // nprobe)
        C = slots * nprobe
        inv_off = bh["inv_offsets"]      # [H, L+1]
        inv_idx = bh["inv_indices"]      # [H, N]  retrieval-local token ids
        # start offset and length of each probed list, per (h,g)
        off_h = inv_off.unsqueeze(1).expand(H, G, num_lists + 1)  # [H,G,L+1]
        list_start = torch.gather(off_h, 2, probed)               # [H,G,nprobe]
        list_end = torch.gather(off_h, 2, probed + 1)             # [H,G,nprobe]
        list_len = list_end - list_start                          # [H,G,nprobe]
        slot_ar = torch.arange(slots, device=dev)                 # [slots]
        # positions into inv_idx[h]: start + slot, valid while slot < list_len
        pos = list_start.unsqueeze(-1) + slot_ar                  # [H,G,nprobe,slots]
        valid = slot_ar < list_len.unsqueeze(-1)                  # [H,G,nprobe,slots]
        pos = torch.where(valid, pos, torch.zeros_like(pos))
        cand_local = torch.gather(
            inv_idx.unsqueeze(1).expand(H, G, N),
            2,
            pos.reshape(H, G, C),
        )  # [H,G,C] retrieval-local token ids
        valid_c = valid.reshape(H, G, C)

        # --- 4) score ONLY the candidates (O(H*G*C)) ------------------------
        # gather packed codes for candidates: packed [H,N,W] -> [H,G,C,W]
        packed = bh["packed"]                                    # [H,N,W]
        W = packed.shape[-1]
        cand_packed = torch.gather(
            packed.unsqueeze(1).expand(H, G, N, W),
            2,
            cand_local.unsqueeze(-1).expand(H, G, C, W),
        )  # [H,G,C,W] uint8
        # PQ score via LUT: unpack 4-bit codes and sum lut over subspaces.
        # codes[...,m] in [0,16); packed stores 2 codes per byte.
        cand_scores = torch.zeros(H, G, C, device=dev, dtype=torch.float32)
        for m in range(M):
            byte = cand_packed[..., m // 2].to(torch.int64)
            code = (byte & 0x0F) if (m % 2 == 0) else (byte >> 4)  # [H,G,C]
            lut_m = lut[:, :, m, :]                                # [H,G,16]
            cand_scores += torch.gather(lut_m, 2, code)            # gather per code
        # add coarse list bias for each candidate's list
        cand_list_ids = torch.gather(
            bh["list_ids"].unsqueeze(1).expand(H, G, N), 2, cand_local
        )  # [H,G,C]
        cand_scores += torch.gather(list_scores, 2, cand_list_ids)
        neg_inf = torch.finfo(torch.float32).min
        cand_scores = torch.where(valid_c, cand_scores, torch.full_like(cand_scores, neg_inf))
        cand_logits = (cand_scores * scale).to(dtype)             # [H,G,C]

        # --- 5) top-k exact tokens among candidates (O(C)) ------------------
        topk = self._streams[0]._resolve_retrieval_topk(N)
        topk = min(max(topk, 0), C)
        if topk > 0:
            exact_cand_logits, exact_pos = torch.topk(cand_logits, k=topk, dim=2, sorted=True)
            exact_local = torch.gather(cand_local, 2, exact_pos)  # [H,G,topk]
        else:
            exact_pos = torch.empty(H, G, 0, device=dev, dtype=torch.long)
            exact_local = torch.empty(H, G, 0, device=dev, dtype=torch.long)
            exact_cand_logits = torch.empty(H, G, 0, device=dev, dtype=cand_logits.dtype)

        # --- 6) full region (live), exact rerank on gathered candidates -----
        sink_end_c = int(regions.sink.numel())
        local_count_c = int(regions.local.numel())
        local_start_c = int(first_cache._length) - local_count_c
        full_count = sink_end_c + local_count_c

        _sb_keys_c = bh.get("shared_base_keys")
        _sb_vals_c = bh.get("shared_base_vals")
        if _sb_keys_c is not None and full_count > 0:
            _sink_k_c = _sb_keys_c[:, :sink_end_c, :]
            _local_k_c = _sb_keys_c[:, local_start_c:local_start_c + local_count_c, :]
            full_keys = torch.cat([_sink_k_c, _local_k_c], dim=1)   # [H, full_count, D]
            _sink_v_c = _sb_vals_c[:, :sink_end_c, :]
            _local_v_c = _sb_vals_c[:, local_start_c:local_start_c + local_count_c, :]
            full_values = torch.cat([_sink_v_c, _local_v_c], dim=1)
        else:
            if full_count > 0:
                full_idx = torch.cat([regions.sink, regions.local], dim=0)
                full_keys = torch.stack(
                    [self._streams[h].cache.keys[full_idx.to(self._streams[h].cache.keys.device)].to(dev)
                     for h in range(H)], dim=0
                )
                full_values = torch.stack(
                    [self._streams[h].cache.values[full_idx.to(self._streams[h].cache.values.device)].to(dev)
                     for h in range(H)], dim=0
                )
            else:
                full_keys = torch.empty(H, 0, D, device=dev, dtype=dtype)
                full_values = torch.empty(H, 0, self._value_dim, device=dev, dtype=dtype)

        full_logits = torch.einsum("hgd,hfd->hgf", q, full_keys) * scale  # [H,G,full]
        vdim = full_values.shape[-1]

        if topk > 0:
            ret_global = bh["retrieval_global"]        # [H, N] local -> global
            exact_global = torch.gather(
                ret_global.unsqueeze(1).expand(H, G, N), 2, exact_local
            )  # [H,G,topk]
            _head_offsets_c = bh.get("head_offsets")
            if _sb_keys_c is not None and _head_offsets_c is not None:
                _cap_c = bh["shared_base_cap"]
                flat_k_c = _sb_keys_c.reshape(H * _cap_c, D)
                flat_v_c = _sb_vals_c.reshape(H * _cap_c, vdim)
                flat_idx_c = (exact_global + _head_offsets_c).reshape(-1)
                exact_keys = flat_k_c[flat_idx_c].reshape(H, G, topk, D)
                exact_values = flat_v_c[flat_idx_c].reshape(H, G, topk, vdim)
            else:
                exact_keys_list = []
                exact_values_list = []
                for h in range(H):
                    ck = self._streams[h].cache.keys
                    cv = self._streams[h].cache.values
                    gh = exact_global[h].reshape(-1).to(ck.device)
                    exact_keys_list.append(ck[gh].to(dev).reshape(G, topk, D))
                    exact_values_list.append(cv[gh].to(dev).reshape(G, topk, vdim))
                exact_keys = torch.stack(exact_keys_list, dim=0)      # [H,G,topk,D]
                exact_values = torch.stack(exact_values_list, dim=0)  # [H,G,topk,vdim]
            exact_logits = (exact_keys * q.unsqueeze(2)).sum(-1) * scale  # [H,G,topk]
        else:
            exact_values = torch.empty(H, G, 0, self._value_dim, device=dev, dtype=dtype)
            exact_logits = torch.empty(H, G, 0, device=dev, dtype=cand_logits.dtype)

        # --- 7) centroid hybrid softmax over candidates + centroid background
        row_max = full_logits.max(dim=2).values
        cand_max = cand_logits.max(dim=2).values
        row_max = torch.maximum(row_max, cand_max)
        if topk > 0:
            row_max = torch.maximum(row_max, exact_logits.max(dim=2).values)

        full_exp = torch.exp(full_logits - row_max.unsqueeze(-1))
        full_exp_sum = full_exp.sum(dim=2)
        # candidate exp (masked invalid to 0)
        cand_exp = torch.exp(cand_logits.float() - row_max.unsqueeze(-1).float())
        cand_exp = torch.where(valid_c, cand_exp, torch.zeros_like(cand_exp))

        if topk > 0:
            old_exact_exp = torch.exp(exact_cand_logits.float() - row_max.unsqueeze(-1).float())
            exact_exp = torch.exp(exact_logits.float() - row_max.unsqueeze(-1).float())
            exact_exp_sum = exact_exp.sum(dim=2)
        else:
            old_exact_exp = torch.zeros(H, G, 0, device=dev, dtype=torch.float32)
            exact_exp = old_exact_exp
            exact_exp_sum = torch.zeros(H, G, device=dev, dtype=torch.float32)

        # background centroid mass: sum over ALL lists of (list_exp approx).
        # Approximate each list's exp mass by its member count * exp(centroid
        # score). Here we use the probed-candidate exp directly for probed lists
        # and rely on centroid value approximation for the rest via value_centroids.
        # denominator = full + candidate approx mass + exact correction.
        cand_exp_sum = cand_exp.sum(dim=2)
        denom = full_exp_sum + cand_exp_sum - old_exact_exp.sum(dim=2) + exact_exp_sum
        denom = denom.clamp_min(torch.finfo(torch.float32).tiny)

        full_output = torch.einsum(
            "hgf,hfd->hgd", (full_exp / denom.unsqueeze(-1)).to(dtype), full_values
        )
        if topk > 0:
            exact_weights = (exact_exp / denom.unsqueeze(-1)).to(dtype)
            exact_output = (exact_weights.unsqueeze(-1) * exact_values).sum(dim=2)
        else:
            exact_output = torch.zeros(H, G, vdim, device=dev, dtype=dtype)

        # background: aggregate candidate exp per IVF list into centroid values,
        # minus the exact tokens' candidate contribution (recomputed exactly).
        background_mass = torch.zeros(H, G, num_lists, device=dev, dtype=torch.float32)
        background_mass.scatter_add_(2, cand_list_ids, cand_exp)
        if topk > 0:
            exact_list_ids = torch.gather(cand_list_ids, 2, exact_pos)  # [H,G,topk]
            background_mass.scatter_add_(2, exact_list_ids, -old_exact_exp)
        background_weight = (background_mass / denom.unsqueeze(-1)).to(dtype)
        background_output = torch.einsum(
            "hgl,hld->hgd", background_weight, bh["value_centroids"]
        )

        output = full_output + exact_output + background_output  # [H,G,D]
        context = output.reshape(1, H * G, 1, vdim).to(dtype=dtype)

        denominator_count = full_count + C
        outputs = []
        empty_long = torch.empty(0, device=dev, dtype=torch.long)
        empty_score = torch.empty(0, device=dev, dtype=cand_logits.dtype)
        for _ in range(H * G):
            outputs.append(
                AttentionOutput(
                    output=None,
                    indices=empty_long,
                    logits=empty_score,
                    weights=empty_score,
                    approx_retrieval_logits=empty_score,
                    exact_retrieval_logits=None,
                    search_result=None,
                    denominator_indices=None,
                    denominator_logits=None,
                    denominator_weights=None,
                    selected_count=int(full_count + topk),
                    denominator_count=int(denominator_count),
                    approx_retrieval_count=int(C),
                    exact_retrieval_count=int(topk),
                    candidate_count=int(C),
                    probed_list_count=int(nprobe),
                )
            )
        return _GroupedQueryDecodeOutput(
            context=context,
            per_query=(
                _GroupedQueryAttentionOutput(
                    output=context[:, :, 0, :],
                    per_stream=tuple(outputs),
                ),
            ),
        )

    # ------------------------------------------------------------------
    # CUDA-graph decode (Task B)
    # ------------------------------------------------------------------

    def _cg_eligible(self) -> bool:
        """Return True if the batched-heads path is active and CUDA graph is enabled."""
        return (
            self._use_cuda_graph
            and self._batched_heads is not None
            and self._shared_base_keys is not None
            and not self._shared_base_fallback
            and self._leading_shape is not None
            and self._leading_shape[0] == 1
            and self._key_dim is not None
            and self._value_dim is not None
        )

    def _cg_invalidate(self) -> None:
        """Discard the captured graph (called on snapshot rebuild)."""
        self._cg = None
        # New snapshot epoch: allow fresh capture attempts.
        self._cg_capture_failed = False
        self._cg_capture_tries = 0

    def _cg_try_capture(self, q_sample: torch.Tensor) -> bool:
        """Warm up and capture the CUDA graph for the batched-heads forward.

        Returns True if capture succeeded, False on any error.
        """
        if not self._cg_eligible():
            return False

        bh = self._batched_heads
        assert bh is not None
        H = bh["H"]
        G = self.num_key_value_groups
        D = self._key_dim
        vD = self._value_dim
        assert D is not None and vD is not None
        dev = bh["packed"].device
        dtype = q_sample.dtype

        # Compute static full-region capacity.
        interval = self.attention_config.index_update_interval
        lw = self.attention_config.local_window
        sink_tokens = self.attention_config.sink_tokens
        first_cache = self._streams[0].cache
        sink_end = min(sink_tokens, first_cache._length)
        # FULL_CAP covers sink + local_window + index_update_interval (max local
        # size). PQ_CG_CAP_SLACK extends it so the graph survives past one
        # deferred interval when no flush is applied (the no-update arm
        # fell back to eager after Delta steps). Padding slots are masked, so
        # slack only costs a slightly larger full-region einsum.
        FULL_CAP = sink_end + lw + interval + int(os.environ.get("PQ_CG_CAP_SLACK", "256"))

        try:
            # Allocate static buffers (filled below from live cache state).
            cg_full_k = torch.zeros(H, FULL_CAP, D, device=dev, dtype=dtype)
            cg_full_v = torch.zeros(H, FULL_CAP, vD, device=dev, dtype=dtype)
            # Mask: 0 for valid slots, -inf for padding.
            neg_inf = float("-inf")
            # Mask stored in float32 to avoid fp16 representation issues.
            cg_mask = torch.full((H, G, FULL_CAP), neg_inf, device=dev, dtype=torch.float32)
            cg_q = torch.zeros(1, H * G, 1, D, device=dev, dtype=dtype)
            cg_out = torch.zeros(1, H * G, 1, vD, device=dev, dtype=dtype)

            # Fill static buffers from live cache state.
            self._cg_refill_buffers(cg_full_k, cg_full_v, cg_mask, sink_end)

            # Track how many valid full-region slots are currently filled.
            regions = first_cache.regions
            local_count = int(regions.local.numel())
            cg_valid_full = sink_end + local_count

            # Copy sample query into static buffer.
            cg_q.copy_(q_sample)

            # Warmup + capture must run with the adapter's device current:
            # torch.cuda.graph captures the CURRENT device's stream, so on
            # multi-GPU models (device_map=auto) capturing from cuda:0 while
            # the tensors live on cuda:N records an empty graph and replay
            # silently produces garbage.
            # First capture in a process warms cuBLAS/Triton with 3 eager runs;
            # RE-captures after a deferred flush reuse the same kernels on the
            # same shapes (N grows by Delta), so one warmup run suffices. At
            # 128K each eager static forward costs ~3 ms x 32 layers per flush.
            warmup_iters = 3 if not getattr(self, "_cg_captured_once", False) else int(
                os.environ.get("PQ_CG_RECAPTURE_WARMUP", "0")
            )
            _prof = os.environ.get("PQ_HSA_GRAPH_PROF", "0") == "1"
            _tw = time.perf_counter() if _prof else 0.0
            with torch.cuda.device(dev):
                torch.cuda.synchronize(dev)
                for _ in range(warmup_iters):
                    _ = self._cg_forward_static(
                        bh, H, G, D, vD, dev, dtype,
                        cg_full_k, cg_full_v, cg_mask, cg_q, cg_valid_full,
                    )
                torch.cuda.synchronize(dev)
            if _prof:
                try:
                    from benchmarks.vllm_backend.pq_hsa_decode_runtime import STATS as _ST

                    _ST["graph_cap_warmup_s"] += time.perf_counter() - _tw
                except Exception:
                    pass

            _tc = time.perf_counter() if _prof else 0.0
            with torch.cuda.device(dev):
                graph = torch.cuda.CUDAGraph()
                # Explicit capture stream on the adapter's device plus
                # thread_local error mode: with device_map pipelines the main
                # thread has in-flight work on other GPUs, and global-mode
                # capture from a non-default device records an empty graph.
                # The capture stream must OUTLIVE the graph: the capture bakes
                # stream-associated state (e.g. cuBLAS workspaces) whose
                # buffers are released when the stream is GC'd, after which
                # replays read reused memory and go non-finite mid-run.
                capture_stream = torch.cuda.Stream(device=dev)
                # torch.cuda.graph() performs a global synchronize, gc.collect,
                # and empty_cache on every __enter__. Capturing one graph per
                # transformer layer made those Python GC scans cost seconds per
                # request. Buffers are already allocated and warmed above, so
                # capture directly on the long-lived side stream.
                with torch.cuda.stream(capture_stream):
                    graph.capture_begin(capture_error_mode="thread_local")
                    try:
                        cg_result = self._cg_forward_static(
                            bh, H, G, D, vD, dev, dtype,
                            cg_full_k, cg_full_v, cg_mask, cg_q, cg_valid_full,
                        )
                        cg_out.copy_(cg_result)
                    finally:
                        graph.capture_end()
            if _prof:
                try:
                    from benchmarks.vllm_backend.pq_hsa_decode_runtime import STATS as _ST

                    _ST["graph_cap_capture_s"] += time.perf_counter() - _tc
                except Exception:
                    pass

            # Self-validate the capture: an empty graph (wrong device/stream)
            # replays as a no-op, so poison the output buffer, replay, and
            # compare against a fresh eager run of the same static forward.
            # After the first validated capture on this adapter, re-captures
            # (same code, same device, shapes grown by Delta) keep the poison +
            # finite check but skip the eager reference forward.
            full_validation = not getattr(self, "_cg_captured_once", False)
            _tv = time.perf_counter() if _prof else 0.0
            with torch.cuda.device(dev):
                cg_out.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize(dev)
                if full_validation:
                    reference = self._cg_forward_static(
                        bh, H, G, D, vD, dev, dtype,
                        cg_full_k, cg_full_v, cg_mask, cg_q, cg_valid_full,
                    )
            if not torch.isfinite(cg_out).all() or (
                full_validation
                and not torch.allclose(
                    cg_out.float(), reference.float(), rtol=2e-2, atol=2e-2
                )
            ):
                if os.environ.get("PQ_CG_DEBUG"):
                    print(
                        f"[PQ_CG_DEBUG] capture validation FAILED dev={dev} "
                        f"finite={bool(torch.isfinite(cg_out).all())} "
                        f"current_dev={torch.cuda.current_device()}",
                        flush=True,
                    )
                self._cg = None
                return False
            if os.environ.get("PQ_CG_DEBUG"):
                print(f"[PQ_CG_DEBUG] capture OK dev={dev}", flush=True)
            if _prof:
                try:
                    from benchmarks.vllm_backend.pq_hsa_decode_runtime import STATS as _ST

                    _ST["graph_cap_validate_s"] += time.perf_counter() - _tv
                except Exception:
                    pass

            self._cg = {
                "graph": graph,
                "capture_stream": capture_stream,
                "full_k": cg_full_k,
                "full_v": cg_full_v,
                "mask": cg_mask,
                "q": cg_q,
                "out": cg_out,
                "sink_end": sink_end,
                "valid_full": cg_valid_full,
                "FULL_CAP": FULL_CAP,
                "dtype": dtype,
            }
            self._cg_captured_once = True
            return True

        except Exception:
            if os.environ.get("PQ_CG_DEBUG"):
                import traceback

                print(f"[PQ_CG_DEBUG] capture raised dev={dev}", flush=True)
                traceback.print_exc()
            self._cg = None
            return False

    def _cg_refill_buffers(
        self,
        full_k: torch.Tensor,
        full_v: torch.Tensor,
        mask: torch.Tensor,
        sink_end: int,
    ) -> None:
        """Fill static full-region buffers from the live shared-base cache.

        Under PAGED_ATTEND=1 (ring mode), _shared_base_keys/_vals no
        longer hold real historical content at absolute positions (see
        _paged_ring_shrink), so sink+local are read straight out of vLLM's
        own paged kv_cache instead, via the same gather_flashattn_kv_range
        helper _apply_pending_batched already uses for the flush/training
        range. =0 / non-ring path is byte-for-byte unchanged.
        """
        H = full_k.shape[0]
        G = mask.shape[1]
        first_cache = self._streams[0].cache
        regions = first_cache.regions
        local_count = int(regions.local.numel())
        local_start = int(first_cache._length) - local_count

        neg_inf = float("-inf")
        _ring = getattr(self, "_shared_base_ring", False)
        if _ring:
            assert self._paged_kv_cache is not None, (
                "ring mode active but set_paged_kv_context() was "
                "never called on this adapter before the first graph capture"
            )
            if sink_end > 0:
                sk, sv = gather_flashattn_kv_range(
                    self._paged_kv_cache, self._paged_attn_metadata,
                    self._paged_seq_idx, 0, sink_end,
                )
                full_k[:, :sink_end, :].copy_(sk.to(full_k.dtype))
                full_v[:, :sink_end, :].copy_(sv.to(full_v.dtype))
                mask[:, :, :sink_end] = 0.0
            if local_count > 0:
                lk, lv = gather_flashattn_kv_range(
                    self._paged_kv_cache, self._paged_attn_metadata,
                    self._paged_seq_idx, local_start, local_start + local_count,
                )
                full_k[:, sink_end:sink_end + local_count, :].copy_(lk.to(full_k.dtype))
                full_v[:, sink_end:sink_end + local_count, :].copy_(lv.to(full_v.dtype))
                mask[:, :, sink_end:sink_end + local_count] = 0.0
        else:
            assert self._shared_base_keys is not None
            assert self._shared_base_vals is not None
            # Fill sink region (positions 0..sink_end-1 in static buf).
            if sink_end > 0:
                full_k[:, :sink_end, :].copy_(self._shared_base_keys[:, :sink_end, :])
                full_v[:, :sink_end, :].copy_(self._shared_base_vals[:, :sink_end, :])
                mask[:, :, :sink_end] = 0.0  # valid

            # Fill local region (positions sink_end..sink_end+local_count-1 in static buf).
            if local_count > 0:
                full_k[:, sink_end:sink_end + local_count, :].copy_(
                    self._shared_base_keys[:, local_start:local_start + local_count, :]
                )
                full_v[:, sink_end:sink_end + local_count, :].copy_(
                    self._shared_base_vals[:, local_start:local_start + local_count, :]
                )
                mask[:, :, sink_end:sink_end + local_count] = 0.0  # valid

        # Remaining slots are padding (-inf already set on allocation).
        FULL_CAP = full_k.shape[1]
        valid_end = sink_end + local_count
        if valid_end < FULL_CAP:
            mask[:, :, valid_end:] = neg_inf

    def _cg_forward_static(
        self,
        bh: dict,
        H: int, G: int, D: int, vD: int,
        dev: torch.device,
        dtype: torch.dtype,
        full_k: torch.Tensor,   # [H, FULL_CAP, D] — static
        full_v: torch.Tensor,   # [H, FULL_CAP, vD] — static
        mask: torch.Tensor,     # [H, G, FULL_CAP] — static, additive logit mask
        q: torch.Tensor,        # [1, HG, 1, D] — static
        valid_full: int,        # MUST be a Python int captured at graph creation time
    ) -> torch.Tensor:
        """Pure-tensor forward pass over static buffers; captures cleanly into a graph.

        All Python branching / shape-dependent logic is resolved BEFORE this
        function (valid_full, topk, etc. are captured as compile-time constants
        in the graph's closure). Returns context [1, H*G, 1, vD].
        """
        from math import sqrt as _sqrt
        scale = 1.0 / _sqrt(D)
        M = bh["M"]
        subdim = bh["subdim"]
        N = bh["N"]
        num_lists = bh["num_lists"]

        # q: [1, HG, 1, D] -> [H, G, D]
        q_hg = q[0, :, 0, :].reshape(H, G, D)

        # --- LUT and coarse scores ------------------------------------------
        codebooks = bh["codebooks"]
        # Rotate queries into PQ space when rotation is enabled (see the eager
        # forward); raw q_hg keeps feeding full-region/exact logits below.
        q_pq = q_hg if bh["rotation"] is None else q_hg.matmul(bh["rotation"].to(dtype))
        q_sub = q_pq.reshape(H, G, M, subdim)
        # (opt-in, default OFF): group-shared retrieval.  The PQ scan / top-k /
        # list-mass are computed ONCE per KV head from the group-mean query (Gr=1
        # rows instead of G), then broadcast to the G query heads; the exact
        # top-k attention and the full (sink+local) region still use each head's
        # own query.  Changes numerics (retrieval + background shared inside a GQA
        # group) -- quality-gated separately; the default path is untouched.
        _gsr = os.environ.get("PQ_HSA_GROUP_SHARED_RETRIEVAL", "0") == "1" and G > 1
        Gr = 1 if _gsr else G
        if _gsr:
            # Receipt (numeric no-op, inside the already opt-in `_gsr` branch):
            # capture-time first-hit counter proving GSR really reached the CUDA-graph
            # body on this adapter.  Read back per rank by the harness receipt; same
            # semantics as `_cuda_attend_only_hits`.
            _gsr_g_hits = getattr(self, "_gsr_graph_hits", 0) + 1
            self._gsr_graph_hits = _gsr_g_hits
            if _gsr_g_hits == 1:
                print("[GSR-GRAPH] group-shared retrieval active in _cg_forward_static "
                      "(capture); G=%d -> Gr=1" % int(G), flush=True)
            q_pq_r = q_pq.mean(dim=1, keepdim=True)
            q_sub_r = q_pq_r.reshape(H, 1, M, subdim)
        else:
            q_pq_r, q_sub_r = q_pq, q_sub
        # One CUDA launch emits the pair LUT + list-score table that the
        # scan actually consumes, replacing 2 GEMMs + 2 scalar muls + the Triton
        # pair-build kernel.  Default OFF (PQ_HSA_FUSED_LUTPREP).
        _pair_tables = None
        lut = None
        list_scores = None
        if (
            os.environ.get("PQ_HSA_FUSED_LUTPREP", "0") == "1"
            and _pq_hsa_fp16_lut()
            and codebooks.dtype == dtype
            and bh["coarse"].dtype == dtype
            and M % 2 == 0
            # Pq_lut_prep emits the PAIR tables, which only the Triton
            # pair kernel consumes, and `tl.arange(0, GROUPS)` needs a power-of-two
            # group dim (triton_lut_scan_h20.py:232 / :385).  Handing pair tables to
            # a non-power-of-two G raises INSIDE the graph body and kills the CUDA
            # graph capture, so keep the aten/Triton LUT build for those shapes.
            # No effect for G=4 (the adopted shape) or G=8.
            and (G & (G - 1)) == 0
            and os.environ.get("PQ_HSA_CUDA_FUSED", "0") != "1"
            and os.environ.get("PQ_HSA_FUSED_RADIX", "0") != "1"
            and os.environ.get("PQ_HSA_FUSED_FULL", "0") != "1"
        ):
            try:
                from pq_hsa.kernels.cuda.pq_fused_h20 import lut_prep as _cu_lut_prep

                if _gsr:
                    # GSR-lean: kG=1 instance of the same lut_prep kernel.
                    if os.environ.get("PQ_HSA_GSR_LEAN", "0") == "1":
                        from pq_hsa.kernels.cuda.pq_fused_h20 import lut_prep_g1 as _cu_lut_prep_g1
                        _pair_tables = _cu_lut_prep_g1(q_pq_r.contiguous(), codebooks, bh["coarse"], scale)
                    else:
                        raise RuntimeError("group-shared retrieval: no kG=1 lut_prep instance")
                else:
                    _pair_tables = _cu_lut_prep(q_pq_r, codebooks, bh["coarse"], scale)
            except Exception as _lp_exc:
                self._lut_prep_error = f"{type(_lp_exc).__name__}: {_lp_exc}"
                _pair_tables = None
        if _pair_tables is None:
            lut, list_scores = _lut_and_list_scores(
                q_sub_r, codebooks, q_pq_r, bh["coarse"], scale, dtype
            )

        _radix_sel = None
        # CUDA fused scan+select+attend. Default OFF until the 0.1 ms/layer gate.
        if os.environ.get("PQ_HSA_CUDA_FUSED", "0") == "1":
            topk_cu = min(max(self._streams[0]._resolve_retrieval_topk(N), 0), N)
            if topk_cu > 0:
                try:
                    from pq_hsa.kernels.cuda.pq_fused_h20 import cuda_fused_decode

                    fused_cu_ctx = cuda_fused_decode(
                        q_hg,
                        lut,
                        list_scores,
                        bh["packed"],
                        bh["list_ids"],
                        num_subspaces=M,
                        k=topk_cu,
                        token_scale=bh.get("knorm"),
                        full_k=full_k,
                        full_v=full_v,
                        mask=mask,
                        retrieval_global=bh["retrieval_global"],
                        shared_base_k=bh.get("shared_base_keys"),
                        shared_base_v=bh.get("shared_base_vals"),
                        shared_base_cap=int(bh.get("shared_base_cap", 0) or 0),
                        value_centroids=bh["value_centroids"],
                        scale=scale,
                    )
                    if fused_cu_ctx is not None:
                        self._cuda_fused_hits = int(getattr(self, "_cuda_fused_hits", 0)) + 1
                        try:
                            from benchmarks.vllm_backend.pq_hsa_decode_runtime import STATS as _CST
                            _CST["cuda_fused_calls"] = int(_CST.get("cuda_fused_calls") or 0) + 1
                        except Exception:
                            pass
                        return fused_cu_ctx
                except Exception as _cu_exc:
                    self._cuda_fused_error = f"{type(_cu_exc).__name__}: {_cu_exc}"
                    if os.environ.get("PQ_HSA_FUSED_DEBUG", "0") == "1":
                        print(f"[pq_hsa] cuda fused fallback: {self._cuda_fused_error}", flush=True)

        # V2: radix-select (exact top-k, no [H,G,N]). Default OFF.
        # PQ_HSA_FUSED_RADIX_EPILOGUE=1 (default when RADIX is on) hands off to
        # fused_hybrid_epilogue_h20. =0 continues the proven aten hybrid using
        # the selected index set and rescaled list masses.
        if os.environ.get("PQ_HSA_FUSED_RADIX", "0") == "1":
            topk_rx = min(max(self._streams[0]._resolve_retrieval_topk(N), 0), N)
            if topk_rx > 0:
                try:
                    from pq_hsa.kernels.fused_radix_select_h20 import (
                        fused_radix_decode_h20,
                        fused_radix_select_h20,
                    )

                    if os.environ.get("PQ_HSA_FUSED_RADIX_EPILOGUE", "1") == "1":
                        fused_radix_ctx = fused_radix_decode_h20(
                            q_hg,
                            lut,
                            list_scores,
                            bh["packed"],
                            bh["list_ids"],
                            num_subspaces=M,
                            k=topk_rx,
                            token_scale=bh.get("knorm"),
                            full_k=full_k,
                            full_v=full_v,
                            mask=mask,
                            retrieval_global=bh["retrieval_global"],
                            shared_base_k=bh.get("shared_base_keys"),
                            shared_base_v=bh.get("shared_base_vals"),
                            shared_base_cap=int(bh.get("shared_base_cap", 0) or 0),
                            value_centroids=bh["value_centroids"],
                            scale=scale,
                            inv_offsets=bh.get("inv_offsets"),
                            inv_indices=bh.get("inv_indices"),
                        )
                        if fused_radix_ctx is not None:
                            self._fused_radix_hits = int(getattr(self, "_fused_radix_hits", 0)) + 1
                            try:
                                from benchmarks.vllm_backend.pq_hsa_decode_runtime import STATS as _RST
                                _RST["fused_radix_calls"] = int(_RST.get("fused_radix_calls") or 0) + 1
                            except Exception:
                                pass
                            return fused_radix_ctx
                    else:
                        _radix_sel = fused_radix_select_h20(
                            bh["packed"],
                            lut,
                            bh["list_ids"],
                            list_scores,
                            num_subspaces=M,
                            k=topk_rx,
                            token_scale=bh.get("knorm"),
                            inv_offsets=bh.get("inv_offsets"),
                            inv_indices=bh.get("inv_indices"),
                        )
                        self._fused_radix_hits = int(getattr(self, "_fused_radix_hits", 0)) + 1
                        try:
                            from benchmarks.vllm_backend.pq_hsa_decode_runtime import STATS as _RST
                            _RST["fused_radix_calls"] = int(_RST.get("fused_radix_calls") or 0) + 1
                        except Exception:
                            pass
                except Exception as _rx_exc:
                    self._fused_radix_error = f"{type(_rx_exc).__name__}: {_rx_exc}"
                    _radix_sel = None
                    if os.environ.get("PQ_HSA_FUSED_DEBUG", "0") == "1":
                        print(f"[pq_hsa] fused radix fallback: {self._fused_radix_error}", flush=True)

        # Full fused scan+topk+list-mass+epilogue. Default OFF.
        if os.environ.get("PQ_HSA_FUSED_FULL", "0") == "1":
            topk_ff = min(max(self._streams[0]._resolve_retrieval_topk(N), 0), N)
            if topk_ff > 0:
                try:
                    from pq_hsa.kernels.fused_decode_full_h20 import fused_full_decode_h20

                    fused_full_ctx = fused_full_decode_h20(
                        q_hg,
                        lut,
                        list_scores,
                        bh["packed"],
                        bh["list_ids"],
                        num_subspaces=M,
                        k=topk_ff,
                        token_scale=bh.get("knorm"),
                        full_k=full_k,
                        full_v=full_v,
                        mask=mask,
                        retrieval_global=bh["retrieval_global"],
                        shared_base_k=bh.get("shared_base_keys"),
                        shared_base_v=bh.get("shared_base_vals"),
                        shared_base_cap=int(bh.get("shared_base_cap", 0) or 0),
                        value_centroids=bh["value_centroids"],
                        scale=scale,
                    )
                    if fused_full_ctx is not None:
                        self._fused_full_hits = int(getattr(self, "_fused_full_hits", 0)) + 1
                        try:
                            from benchmarks.vllm_backend.pq_hsa_decode_runtime import STATS as _RST
                            _RST["fused_full_calls"] = int(_RST.get("fused_full_calls") or 0) + 1
                        except Exception:
                            pass
                        return fused_full_ctx
                except Exception as _ff_exc:
                    self._fused_full_error = f"{type(_ff_exc).__name__}: {_ff_exc}"
                    if os.environ.get("PQ_HSA_FUSED_DEBUG", "0") == "1":
                        print(f"[pq_hsa] fused full fallback: {self._fused_full_error}", flush=True)

        # --- Retrieval scan -------------------------------------------------
        _radix_list_mass = None
        if _radix_sel is not None:
            topk = int(_radix_sel["topk_idx"].shape[-1])
            exact_local = _radix_sel["topk_idx"]
            exact_approx_logits = _radix_sel["topk_val"].to(dtype)
            fused_retrieval_row_max = _radix_sel["row_max"]
            fused_retrieval_exp_sum = _radix_sel["ret_exp_sum"]
            _radix_list_mass = _radix_sel["list_mass"]
            approx_logits = None
            _par_mass = None
            _par_stream = None
        else:
            approx_logits, fused_retrieval_row_max, fused_retrieval_exp_sum = self._scan_retrieval_logits(
                bh, lut, list_scores, fused_row_max=True, pair_tables=_pair_tables
            )  # [H, G, N]
            # B: slack padding must not enter top-k / list mass.
            _rm = bh.get("ret_mask")
            if _rm is not None:
                if _rm.dtype != approx_logits.dtype:
                    _rm = _rm.to(dtype=approx_logits.dtype)
                    bh["ret_mask"] = _rm
                approx_logits.add_(_rm)

            # The per-list background mass is a second, independent
            # read of the [H,G,N] scores.  It does not depend on top-k, so it can
            # run CONCURRENTLY with aten mbtopk on a side stream forked from (and
            # joined back into) the capture stream.  Pure scheduling change: the
            # kernel, its inputs and its output are byte-identical.
            _par_mass = None
            _par_stream = None
            if (
                os.environ.get("PQ_HSA_PARSTREAM", "0") == "1"
                and os.environ.get("PQ_HSA_CUDA_ATTEND_ONLY", "0") == "1"
                and approx_logits is not None
                and fused_retrieval_row_max is not None
            ):
                try:
                    _par_stream = getattr(self, "_t83r_side_stream", None)
                    if _par_stream is None:
                        _par_stream = torch.cuda.Stream(device=approx_logits.device)
                        self._t83r_side_stream = _par_stream
                    _cur_stream = torch.cuda.current_stream()
                    _par_stream.wait_stream(_cur_stream)
                    with torch.cuda.stream(_par_stream):
                        _par_mass = list_exp_sums_sorted_multihead_triton(
                            approx_logits, fused_retrieval_row_max,
                            bh["inv_offsets"], num_lists=num_lists,
                        )
                except Exception as _ps_exc:
                    self._parstream_error = f"{type(_ps_exc).__name__}: {_ps_exc}"
                    _par_mass = None
                    _par_stream = None

            # --- top-k exact tokens --------------------------------------------
            topk = self._streams[0]._resolve_retrieval_topk(N)
            topk = min(max(topk, 0), N)
            if topk > 0:
                # Exact tail-histogram select instead of aten mbtopk over all
                # N. Default OFF (PQ_HSA_SCAN_TOPK). Returns None -> aten fallback.
                _tt = None
                # Single-kernel exact radix top-k (int32 idx / fp32 val) -- only on the
                # CUDA attend-only path, which consumes those dtypes directly.
                if (
                    os.environ.get("PQ_HSA_TOPK_RADIX", "0") == "1"
                    and os.environ.get("PQ_HSA_CUDA_ATTEND_ONLY", "0") == "1"
                    and approx_logits.dtype == torch.float16
                ):
                    try:
                        from pq_hsa.kernels.cuda.pq_fused_h20 import topk_radix as _topk_radix
                        _tt = _topk_radix(approx_logits, topk)
                        self._topk_radix_hits = int(getattr(self, "_topk_radix_hits", 0)) + 1
                    except Exception as _rx_exc2:
                        self._topk_radix_error = f"{type(_rx_exc2).__name__}: {_rx_exc2}"
                        _tt = None
                if _tt is None and os.environ.get("PQ_HSA_SCAN_TOPK", "0") == "1" and fused_retrieval_row_max is not None:
                    try:
                        from pq_hsa.kernels.triton_tail_topk_h20 import select_tail_topk

                        _tt = select_tail_topk(
                            approx_logits, fused_retrieval_row_max, topk,
                            check=(os.environ.get("PQ_HSA_SCAN_TOPK_CHECK", "0") == "1"),
                        )
                    except Exception as _tt_exc:
                        self._scan_topk_error = f"{type(_tt_exc).__name__}: {_tt_exc}"
                        _tt = None
                if _tt is not None:
                    exact_approx_logits, exact_local = _tt
                else:
                    exact_approx_logits, exact_local = _retrieval_exact_topk(
                        approx_logits, topk
                    )
            else:
                exact_local = torch.empty(H, Gr, 0, device=dev, dtype=torch.long)
                exact_approx_logits = torch.empty(
                    H, Gr, 0, device=dev,
                    dtype=(lut.dtype if lut is not None else approx_logits.dtype))

        _gsr_lean = _gsr and os.environ.get("PQ_HSA_GSR_LEAN", "0") == "1" \
            and os.environ.get("PQ_HSA_CUDA_ATTEND_ONLY", "0") == "1" and topk > 0
        _gsr_expanded = False
        if _gsr and not _gsr_lean:
            # Broadcast the per-KV-head retrieval result to the G query heads.
            _gsr_expanded = True
            exact_local = exact_local.expand(H, G, -1).contiguous()
            exact_approx_logits = exact_approx_logits.expand(H, G, -1).contiguous()
            if fused_retrieval_row_max is not None:
                fused_retrieval_row_max = fused_retrieval_row_max.expand(H, G).contiguous()
            if fused_retrieval_exp_sum is not None:
                fused_retrieval_exp_sum = fused_retrieval_exp_sum.expand(H, G).contiguous()

        # CUDA "attend-only" epilogue. Keeps the Triton pair-LUT scan and
        # the aten mbtopk above (both still the fastest known implementation of
        # their step) and replaces ONLY the epilogue with pq_exact_attend, whose
        # reduce stage is the multi-warp kernel. Default OFF.
        if (
            os.environ.get("PQ_HSA_CUDA_ATTEND_ONLY", "0") == "1"
            and topk > 0
            and fused_retrieval_row_max is not None
            and fused_retrieval_exp_sum is not None
            and approx_logits is not None
        ):
            try:
                if _par_mass is not None:
                    # Join the side stream before the epilogue reads it.
                    torch.cuda.current_stream().wait_stream(_par_stream)
                    _mass_ret = _par_mass
                else:
                    _mass_ret = list_exp_sums_sorted_multihead_triton(
                        approx_logits,
                        (fused_retrieval_row_max[:, :1].contiguous() if _gsr
                         else fused_retrieval_row_max),
                        bh["inv_offsets"],
                        num_lists=num_lists,
                    )
                if _gsr_lean:
                    # One launch -> int32 idx, fp32 val, row_max, exp_sum, mass (all [H,G,*]).
                    from pq_hsa.kernels.cuda.pq_fused_h20 import gsr_expand as _gsr_expand
                    (exact_local, exact_approx_logits, fused_retrieval_row_max,
                     fused_retrieval_exp_sum, _mass_ret) = _gsr_expand(
                        exact_local, exact_approx_logits,
                        fused_retrieval_row_max[:, :1], fused_retrieval_exp_sum[:, :1],
                        _mass_ret, G)
                    _gsr_expanded = True
                    self._gsr_lean_hits = int(getattr(self, "_gsr_lean_hits", 0)) + 1
                elif _gsr:
                    _mass_ret = _mass_ret.expand(H, G, -1).contiguous()
                # Opt-in paged-KV variant of the exact-gather read (see
                # _paged_attend_enabled()). Falls through to the frozen
                # cuda_attend_only() whenever the flag is off, the paged-KV
                # context has not been set on this adapter, or the paged
                # kernel is unavailable/errors -- byte-identical to before
                # in every one of those cases.
                _ao_ctx = None
                if _paged_attend_enabled() and self._paged_kv_cache is not None:
                    from pq_hsa.kernels.cuda.pq_fused_h20 import cuda_attend_only_paged

                    # Receipt (opt-in path only): the paged kernel calls
                    # .contiguous() on vLLM's page; a non-contiguous page (a
                    # stride-order-permuted allocation) would silently copy the
                    # whole KV cache every step. Print the layout once.
                    _pk_hits = int(getattr(self, "_e40c_paged_kv_hits", 0)) + 1
                    self._e40c_paged_kv_hits = _pk_hits
                    if _pk_hits == 1:
                        _pk = self._paged_kv_cache
                        print(f"[paged] kv_cache dim={_pk.dim()} shape={tuple(_pk.shape)} "
                              f"strides={tuple(_pk.stride())} contiguous={_pk.is_contiguous()} dtype={_pk.dtype}", flush=True)
                    _bt_row = self._paged_attn_metadata.block_table[self._paged_seq_idx]
                    _ao_ctx = cuda_attend_only_paged(
                        bh,
                        q_hg,
                        self._paged_kv_cache,
                        _bt_row,
                        full_k,
                        full_v,
                        mask,
                        exact_local,
                        exact_approx_logits,
                        _mass_ret,
                        fused_retrieval_row_max,
                        fused_retrieval_exp_sum,
                        scale,
                    )
                if _ao_ctx is None:
                    from pq_hsa.kernels.cuda.pq_fused_h20 import cuda_attend_only

                    if os.environ.get("PQ_HSA_E42_CG_STORE", "0") == "1":
                        # (opt-in, diagnostic): stash the retrieval-step inputs of the
                        # attend tail so a standalone script can time the "reuse step"
                        # body (= this tail alone) at a given (H, G) shape.
                        self._e42_cg_store = {
                            "q_hg": q_hg, "full_k": full_k, "full_v": full_v, "mask": mask,
                            "exact_local": exact_local, "exact_approx_logits": exact_approx_logits,
                            "mass": _mass_ret, "rmax": fused_retrieval_row_max,
                            "esum": fused_retrieval_exp_sum, "scale": scale,
                        }
                    _ao_ctx = cuda_attend_only(
                        bh,
                        q_hg,
                        full_k,
                        full_v,
                        mask,
                        exact_local,
                        exact_approx_logits,
                        _mass_ret,
                        fused_retrieval_row_max,
                        fused_retrieval_exp_sum,
                        scale,
                    )
                if _ao_ctx is not None:
                    self._cuda_attend_only_hits = int(
                        getattr(self, "_cuda_attend_only_hits", 0)) + 1
                    try:
                        from benchmarks.vllm_backend.pq_hsa_decode_runtime import STATS as _AST
                        _AST["cuda_attend_only_calls"] = int(
                            _AST.get("cuda_attend_only_calls") or 0) + 1
                    except Exception:
                        pass
                    return _ao_ctx
            except Exception as _ao_exc:
                if _gsr and not _gsr_expanded:
                    # GSR-lean failed before broadcasting: restore the layout so the
                    # aten fallbacks below see [H,G,*] tensors.
                    exact_local = exact_local.expand(H, G, -1).contiguous()
                    exact_approx_logits = exact_approx_logits.expand(H, G, -1).contiguous()
                    if fused_retrieval_row_max is not None and fused_retrieval_row_max.shape[1] != G:
                        fused_retrieval_row_max = fused_retrieval_row_max.expand(H, G).contiguous()
                    if fused_retrieval_exp_sum is not None and fused_retrieval_exp_sum.shape[1] != G:
                        fused_retrieval_exp_sum = fused_retrieval_exp_sum.expand(H, G).contiguous()
                    _gsr_expanded = True
                self._cuda_attend_only_error = f"{type(_ao_exc).__name__}: {_ao_exc}"
                if os.environ.get("PQ_HSA_FUSED_DEBUG", "0") == "1":
                    print(f"[pq_hsa] cuda attend-only fallback: {self._cuda_attend_only_error}",
                          flush=True)

        # Fused post-scan hybrid (exact gather + full + softmax + centroids).
        # Exact torch.topk is kept (no _approx_topk). Falls back to the aten graph.
        if (
            os.environ.get("PQ_HSA_FUSED_DECODE", "0") == "1"
            and topk > 0
            and fused_retrieval_row_max is not None
            and fused_retrieval_exp_sum is not None
        ):
            try:
                from pq_hsa.kernels.fused_decode_h20 import fused_hybrid_epilogue_h20

                retrieval_row_max = fused_retrieval_row_max
                if _radix_list_mass is not None:
                    background_mass_f32 = _radix_list_mass
                else:
                    background_mass_f32 = list_exp_sums_sorted_multihead_triton(
                        approx_logits, retrieval_row_max, bh["inv_offsets"], num_lists=num_lists,
                    )
                fused_ctx = fused_hybrid_epilogue_h20(
                    q_hg,
                    full_k,
                    full_v,
                    mask,
                    exact_local,
                    exact_approx_logits,
                    bh["retrieval_global"],
                    bh.get("shared_base_keys"),
                    bh.get("shared_base_vals"),
                    bh.get("head_offsets"),
                    int(bh.get("shared_base_cap", 0) or 0),
                    bh["list_ids"],
                    background_mass_f32,
                    bh["value_centroids"],
                    retrieval_row_max,
                    fused_retrieval_exp_sum,
                    scale,
                )
                if fused_ctx is not None:
                    self._fused_decode_hits = int(getattr(self, "_fused_decode_hits", 0)) + 1
                    try:
                        from benchmarks.vllm_backend.pq_hsa_decode_runtime import STATS as _RST
                        _RST["fused_decode_calls"] = int(_RST.get("fused_decode_calls") or 0) + 1
                    except Exception:
                        pass
                    return fused_ctx
            except Exception as _fused_exc:
                self._fused_decode_error = f"{type(_fused_exc).__name__}: {_fused_exc}"
                if os.environ.get("PQ_HSA_FUSED_DEBUG", "0") == "1":
                    print(f"[pq_hsa] fused decode fallback: {self._fused_decode_error}", flush=True)

        # --- Full region logits with static mask ----------------------------
        # Use the FULL_CAP-sized static tensors; mask handles padding.
        # Compute in float32 to prevent fp16 overflow (q @ huge_K = inf)
        # which would cause inf + (-inf) = NaN when mask is applied.
        FULL_CAP = full_k.shape[1]
        full_logits = (
            torch.einsum("hgd,hfd->hgf", q_hg.float(), full_k[:, :FULL_CAP, :].float()) * scale
            + mask.float()  # padding slots: logit -> -inf -> exp=0, robust to huge K values
        ).to(dtype)

        # --- Exact rerank gather (before row_max!) ---------------------------
        # The TRUE exact logits must participate in row_max, exactly like the
        # eager batched path.  Under direction_normalize the PQ approximation
        # can undershoot the true logit by more than ln(65504/1) ~ 11.09, so
        # exp(exact_logits - approx_row_max) overflows fp16 to inf and one
        # (h, g) row of the output turns NaN (denominator inf / inf).
        if topk > 0:
            ret_global = bh["retrieval_global"]
            exact_global = torch.gather(
                ret_global.unsqueeze(1).expand(H, G, N), 2, exact_local
            )
            # Flat gather into shared base buffers.
            _head_offsets = bh.get("head_offsets")
            _sb_keys = bh.get("shared_base_keys")
            _sb_vals = bh.get("shared_base_vals")
            _cap = bh.get("shared_base_cap", 0)
            if _sb_keys is not None and _head_offsets is not None and _cap > 0:
                flat_k = _sb_keys.reshape(H * _cap, D)
                flat_v = _sb_vals.reshape(H * _cap, vD)
                flat_idx = (exact_global + _head_offsets).to(dtype=torch.int32).reshape(-1)
                exact_keys = flat_k[flat_idx].reshape(H, G, topk, D)
                exact_values = flat_v[flat_idx].reshape(H, G, topk, vD)
            else:
                exact_keys_list = []
                exact_values_list = []
                for h in range(H):
                    ck = self._streams[h].cache.keys
                    cv = self._streams[h].cache.values
                    gh = exact_global[h].reshape(-1)
                    exact_keys_list.append(ck[gh].reshape(G, topk, D))
                    exact_values_list.append(cv[gh].reshape(G, topk, vD))
                exact_keys = torch.stack(exact_keys_list, dim=0)
                exact_values = torch.stack(exact_values_list, dim=0)
            exact_logits = (
                torch.matmul(exact_keys, q_hg.unsqueeze(-1)).squeeze(-1) * scale
            )
        else:
            exact_values = torch.empty(H, G, 0, vD, device=dev, dtype=dtype)
            exact_logits = torch.empty(H, G, 0, device=dev, dtype=approx_logits.dtype)

        # --- row_max over full + retrieval + TRUE exact, then softmax masses -
        if fused_retrieval_row_max is not None:
            retrieval_row_max = fused_retrieval_row_max.to(full_logits.dtype)
        else:
            retrieval_row_max = approx_logits.amax(dim=2).to(full_logits.dtype)
        row_max = torch.maximum(
            full_logits.amax(dim=2),  # -inf padded slots don't affect max
            retrieval_row_max,
        )
        if topk > 0:
            row_max = torch.maximum(row_max, exact_approx_logits.amax(dim=2))
            row_max = torch.maximum(row_max, exact_logits.max(dim=2).values)

        full_exp = torch.exp(full_logits - row_max.unsqueeze(-1))
        # Padded slots: exp(-inf - row_max) = 0, so they don't contribute.
        full_exp_sum = full_exp.sum(dim=2)

        if _radix_list_mass is not None:
            # Masses were accumulated at retrieval row_max; rescale to the
            # hybrid row_max (full + exact may be larger).
            _scale_m = torch.exp(
                fused_retrieval_row_max.float() - row_max.float()
            ).unsqueeze(-1)
            background_mass_f32 = _radix_list_mass * _scale_m
        else:
            # Fix: in the aten tail (non-ATTEND_ONLY) path,
            # approx_logits still has the un-expanded Gr=1 group dim under GSR
            # (only exact_local/exact_approx_logits/row_max/exp_sum were
            # broadcast to G above), so row_max must be sliced back down to
            # match before this call -- mirrors the ATTEND_ONLY slice at
            # `fused_retrieval_row_max[:, :1]` a few dozen lines up. Without
            # this, list_exp_sums_sorted_multihead_triton raises
            # "row_max must have shape (H, 1), got (H, G)" on every layer
            # whenever GSR is combined with a CUDA-graph capture.
            _row_max_scan = row_max[:, :1].contiguous() if _gsr_expanded else row_max
            background_mass_f32 = list_exp_sums_sorted_multihead_triton(
                approx_logits, _row_max_scan, bh["inv_offsets"], num_lists=num_lists,
            )  # [H,Gr,L] float32; re-broadcast to [H,G,L] below under GSR
            if _gsr_expanded:
                background_mass_f32 = background_mass_f32.expand(H, G, -1).contiguous()
        if fused_retrieval_exp_sum is not None:
            retrieval_exp_sum = (
                fused_retrieval_exp_sum
                * torch.exp(fused_retrieval_row_max.float() - row_max.float())
            ).to(full_exp.dtype)
        else:
            retrieval_exp_sum = background_mass_f32.sum(dim=2).to(full_exp.dtype)

        if topk > 0:
            old_exact_exp = torch.exp(exact_approx_logits - row_max.unsqueeze(-1))
            exact_exp = torch.exp(exact_logits - row_max.unsqueeze(-1))
            exact_exp_sum = exact_exp.sum(dim=2)
        else:
            old_exact_exp = torch.zeros(H, G, 0, device=dev, dtype=full_exp.dtype)
            exact_exp = old_exact_exp
            exact_exp_sum = torch.zeros(H, G, device=dev, dtype=full_exp.dtype)

        denom = full_exp_sum + retrieval_exp_sum - old_exact_exp.sum(dim=2) + exact_exp_sum
        denom = denom.clamp_min(torch.finfo(denom.dtype).tiny)

        full_output = torch.einsum("hgf,hfd->hgd", full_exp / denom.unsqueeze(-1), full_v[:, :FULL_CAP, :])
        if topk > 0:
            exact_weights = exact_exp / denom.unsqueeze(-1)
            exact_output = torch.matmul(
                exact_weights.unsqueeze(2).to(exact_values.dtype), exact_values
            ).squeeze(2)
        else:
            exact_output = torch.zeros(H, G, vD, device=dev, dtype=dtype)

        list_ids = bh["list_ids"]
        background_mass = background_mass_f32.to(full_exp.dtype)
        if topk > 0:
            exact_list_ids = torch.gather(
                list_ids.unsqueeze(1).expand(H, G, N), 2, exact_local
            )
            background_mass.scatter_add_(2, exact_list_ids, -old_exact_exp)
        background_weight = background_mass / denom.unsqueeze(-1)
        background_output = torch.einsum("hgl,hld->hgd", background_weight, bh["value_centroids"])

        output = full_output + exact_output + background_output
        context = output.reshape(1, H * G, 1, vD).to(dtype=dtype)
        return context

    def forward_context_fast(self, queries: torch.Tensor) -> torch.Tensor:
        """Steady-decode fast path: graph replay returning the context ONLY.

        ``queries`` is ``[1, H*G, D]``. Returns ``[1, H*G, D]``. The returned
        tensor aliases the graph's static output buffer, so the caller must
        consume (copy) it before the next replay. Falls back to the full
        ``forward()`` path when the graph is unavailable.

        This exists because the wrapped vLLM decode only ever reads
        ``.context``: building 32 ``AttentionOutput`` dataclasses plus a
        defensive ``clone()`` per layer per token measured ~3.6 ms/tok of host
        time at 128K (profiled).
        """
        # Fast lane: reuse the cached static views, skip unsqueeze/slice.
        _lrp = getattr(self, "_lean_replay_state", None)
        if _lrp is not None and len(_lrp) == 11 and _lrp[0] is self._cg:
            (_cgL, _fcL, _ilenL, _sinkL, _capL, _qbufL, _graphL, _outL, _evArmed,
             _q3L, _o3L) = _lrp
            if (
                self._streams[0].cache is _fcL
                and getattr(self, "_ga_k", None) is None
                and not int(getattr(self, "_cg_sentinel_interval", 0))
            ):
                _regL = _fcL.regions
                if (
                    int(_regL.retrieval.numel()) == _ilenL
                    and _sinkL + int(_regL.local.numel()) <= _capL
                ):
                    # The event flag must NOT be cached -- the harness
                    # flips PQ_HSA_STEADY_EVENTS on only around the measurement
                    # window, and a stale cached False silently drops the
                    # `replay_gpu` span.  Under HOST_SYNC_FIX=1 this is one sticky
                    # getattr once the flag has latched.
                    if self._t83r_hsf:
                        _evL = bool(getattr(self, "_steady_events", False))
                        if not _evL and os.environ.get("PQ_HSA_STEADY_EVENTS", "0") == "1":
                            self._steady_events = True
                            _evL = True
                    else:
                        _evL = os.environ.get("PQ_HSA_STEADY_EVENTS", "0") == "1"
                    if _evL:
                        from benchmarks.cuda_event_log import span_end, span_start

                        span_start("replay_gpu")
                    if getattr(self, "_q_staged_cg", None) is _cgL:
                        self._q_staged_cg = None
                    else:
                        _q3L.copy_(queries)
                    _graphL.replay()
                    if _evL:
                        span_end("replay_gpu")
                    return _o3L
        if not getattr(self, "_candidate_pruned", False):
            raw = self._cg_replay_raw(queries.unsqueeze(2))
            if raw is not None:
                return raw[:, :, 0, :]
        return self.forward(queries).context

    def _cg_replay_raw(self, queries: torch.Tensor) -> torch.Tensor | None:
        """Copy ``queries`` into the static buffer and replay the graph.

        Returns the static output buffer ``[1, H*G, 1, vD]`` (NOT cloned), or
        None when the caller must fall back to the eager path.
        """
        # Steady-state fast lane.  Everything the general path below
        # re-derives every step (env lookups, dict lookups, getattr chains, the
        # cuda device context manager) is hoisted into a cached tuple keyed on the
        # graph object; only the two checks that can actually change per step --
        # snapshot staleness and full-buffer overflow -- are still evaluated.
        # Any mismatch falls through to the untouched general path.
        # Default OFF (PQ_HSA_LEAN_REPLAY).
        _lrp = getattr(self, "_lean_replay_state", None)
        if _lrp is not None and _lrp[0] is self._cg:
            (_cgL, _fcL, _ilenL, _sinkL, _capL, _qbufL, _graphL, _outL, _evArmed) = _lrp[:9]
            if (
                self._streams[0].cache is _fcL
                and getattr(self, "_ga_k", None) is None
                and not int(getattr(self, "_cg_sentinel_interval", 0))
            ):
                _regL = _fcL.regions
                if (
                    int(_regL.retrieval.numel()) == _ilenL
                    and _sinkL + int(_regL.local.numel()) <= _capL
                ):
                    # The event flag must NOT be cached -- the harness
                    # flips PQ_HSA_STEADY_EVENTS on only around the measurement
                    # window, and a stale cached False silently drops the
                    # `replay_gpu` span.  Under HOST_SYNC_FIX=1 this is one sticky
                    # getattr once the flag has latched.
                    if self._t83r_hsf:
                        _evL = bool(getattr(self, "_steady_events", False))
                        if not _evL and os.environ.get("PQ_HSA_STEADY_EVENTS", "0") == "1":
                            self._steady_events = True
                            _evL = True
                    else:
                        _evL = os.environ.get("PQ_HSA_STEADY_EVENTS", "0") == "1"
                    if _evL:
                        from benchmarks.cuda_event_log import span_end, span_start

                        span_start("replay_gpu")
                    if getattr(self, "_q_staged_cg", None) is _cgL:
                        self._q_staged_cg = None
                    else:
                        _qbufL.copy_(queries)
                    _graphL.replay()
                    if _evL:
                        span_end("replay_gpu")
                    return _outL
        if not self._use_cuda_graph:
            return None

        bh = self._batched_heads
        if bh is None:
            return None

        first_cache = self._streams[0].cache
        if int(first_cache.regions.retrieval.numel()) != bh["indexed_retrieval_len"]:
            # Snapshot stale: rebuild. B may keep the graph when N fits N_CAP.
            self._maybe_build_batched_heads()
            bh = self._batched_heads
            if bh is None or self._cg is None:
                return None

        # Lazily capture the graph if not yet captured. First-attempt capture
        # failures are often transient under multi-GPU device_map (the replay
        # validation sees NaN once, then a retry succeeds), so retry a bounded
        # number of times before giving up for this snapshot epoch — a
        # permanently-eager layer costs ~1 ms/step, far more than the retries.
        if self._cg is None:
            tries = getattr(self, "_cg_capture_tries", 0)
            if getattr(self, "_cg_capture_failed", False) or tries >= 3:
                return None
            ok = self._cg_try_capture(queries)
            if not ok:
                self._cg_capture_tries = tries + 1
                return None
            self._cg_capture_tries = 0

        cg = self._cg
        if cg is None:
            return None

        # Per-step maintenance OUTSIDE the graph:
        # 1. Copy query into the static buffer.
        # 2. The K/V for the current append were already written into the
        #    shared-base buffer and the static full buffers are kept in sync
        #    (updated in _cg_update_after_append, called from append()).
        #    Check that we haven't drifted.
        regions = first_cache.regions
        local_count = int(regions.local.numel())
        expected_valid = cg["sink_end"] + local_count
        if expected_valid > cg["FULL_CAP"]:
            # Buffer overflow; fall back eagerly.
            self._cg_invalidate()
            return None

        # CUDA-event span around in-graph GPU (q.copy_ + replay).
        # Off unless PQ_HSA_STEADY_EVENTS=1. Host checks above stay outside.
        # HOST_SYNC_FIX=1 uses the instance flag (refreshed if env
        # flips on after construction so A/B events still work).
        if os.environ.get("PQ_HSA_HOST_SYNC_FIX", "1") == "1":
            _ev = bool(getattr(self, "_steady_events", False))
            if not _ev and os.environ.get("PQ_HSA_STEADY_EVENTS", "0") == "1":
                self._steady_events = True
                _ev = True
        else:
            _ev = os.environ.get("PQ_HSA_STEADY_EVENTS", "0") == "1"
        if _ev:
            from benchmarks.cuda_event_log import enable, is_enabled, span_end, span_start

            if not is_enabled():
                enable()
            span_start("replay_gpu")
        ga_k = getattr(self, "_ga_k", None)
        if ga_k is not None:
            _t83e_append_scatter(
                self._shared_base_keys,
                self._shared_base_vals,
                cg["full_k"],
                cg["full_v"],
                cg["mask"],
                ga_k,
                self._ga_v,
                int(self._ga_pos),
                int(self._ga_spos),
            ) or self._cg_deferred_append_eager()
            self._ga_k = None
            self._ga_v = None
        if getattr(self, "_q_staged_cg", None) is cg:
            self._q_staged_cg = None      # Q already staged by lean_append
        else:
            cg["q"].copy_(queries)
        with torch.cuda.device(cg["q"].device):
            cg["graph"].replay()
        if _ev:
            span_end("replay_gpu")
        # Arm the fast lane for the next step once every per-step
        # invariant above has been validated at least once for THIS graph object.
        if (
            os.environ.get("PQ_HSA_LEAN_REPLAY", "0") == "1"
            and getattr(self, "_ga_k", None) is None
            and not int(getattr(self, "_cg_sentinel_interval", 0))
        ):
            _qb = cg["q"]
            _ob = cg["out"]
            # HOST_SYNC_FIX is a protocol constant for the life of the process.
            self._t83r_hsf = os.environ.get("PQ_HSA_HOST_SYNC_FIX", "1") == "1"
            self._lean_replay_state = (
                cg, first_cache, int(bh["indexed_retrieval_len"]),
                int(cg["sink_end"]), int(cg["FULL_CAP"]),
                _qb, cg["graph"], _ob, bool(_ev),
                # The [1,HG,D] views of the two static buffers are built
                # once here instead of once per layer per step (unsqueeze + slice).
                _qb[:, :, 0, :], _ob[:, :, 0, :],
            )

        # Post-replay sentinel: if the replay ever produces non-finite output,
        # disable the graph, report once, and let this step run eagerly so the
        # result stays correct. The isfinite check forces a GPU sync, so it
        # runs every `_cg_sentinel_interval` steps (up to that many corrupted
        # steps can slip through before detection; PPL metrics surface them).
        self._cg_step_counter = getattr(self, "_cg_step_counter", 0) + 1
        interval = int(getattr(self, "_cg_sentinel_interval", 0))
        if interval and self._cg_step_counter % interval == 0 and not bool(
            torch.isfinite(cg["out"]).all()
        ):
            if os.environ.get("PQ_CG_DEBUG"):
                first_cache_dbg = self._streams[0].cache
                parts = [
                    "[PQ_CG_DEBUG] replay produced non-finite output; "
                    f"dev={cg['q'].device} length={int(first_cache_dbg._length)} "
                    f"valid_full={cg['valid_full']} FULL_CAP={cg['FULL_CAP']} "
                    f"step_counter={self._cg_step_counter}"
                ]
                for name in ("full_k", "full_v", "mask", "q", "out"):
                    t = cg[name].float()
                    finite = t[torch.isfinite(t)]
                    parts.append(
                        f"  {name}: finite={int(torch.isfinite(t).sum())}/{t.numel()} "
                        f"maxabs_finite={float(finite.abs().max()) if finite.numel() else float('nan'):.3e}"
                    )
                knorm_dbg = self._batched_heads.get("knorm") if self._batched_heads else None
                if knorm_dbg is not None:
                    kf = knorm_dbg.float()
                    parts.append(
                        f"  knorm: finite={int(torch.isfinite(kf).sum())}/{kf.numel()} "
                        f"maxabs={float(kf.abs().max()):.3e} min={float(kf.min()):.3e}"
                    )
                # Decisive probe: re-run the SAME static forward eagerly with
                # the same buffers. Finite eager result => replay-specific
                # (baked kernel state) divergence; non-finite => a genuine
                # numerical event reproducible outside the graph.
                try:
                    bh_dbg = self._batched_heads
                    with torch.cuda.device(cg["q"].device):
                        eager_ref = self._cg_forward_static(
                            bh_dbg,
                            bh_dbg["H"],
                            self.num_key_value_groups,
                            self._key_dim,
                            self._value_dim,
                            cg["q"].device,
                            cg["dtype"],
                            cg["full_k"],
                            cg["full_v"],
                            cg["mask"],
                            cg["q"],
                            cg["valid_full"],
                        )
                    parts.append(
                        "  eager-static recompute finite="
                        f"{int(torch.isfinite(eager_ref).sum())}/{eager_ref.numel()}"
                    )
                except Exception as exc:  # pragma: no cover - debug only
                    parts.append(f"  eager-static recompute raised: {exc!r}")
                print("\n".join(parts), flush=True)
            self._cg_invalidate()
            self._cg_capture_failed = True
            return None

        return cg["out"]

    def _forward_many_batched_heads_graph(
        self,
        queries: torch.Tensor,
    ) -> _GroupedQueryDecodeOutput | None:
        """CUDA-graph replay of the batched-heads decode.

        Returns None to fall back to eager if graph is not ready or invalid.
        """
        raw = self._cg_replay_raw(queries)
        if raw is None:
            return None
        bh = self._batched_heads
        cg = self._cg
        assert bh is not None and cg is not None
        first_cache = self._streams[0].cache

        # Build lightweight return value.
        context = raw.clone()  # callers may retain this
        H = bh["H"]
        G = self.num_key_value_groups
        N = bh["N"]
        full_count = cg["sink_end"] + int(first_cache.regions.local.numel())
        topk = self._streams[0]._resolve_retrieval_topk(N)
        topk = min(max(topk, 0), N)
        denominator_count = full_count + N
        empty_long = torch.empty(0, device=cg["q"].device, dtype=torch.long)
        empty_score = torch.empty(0, device=cg["q"].device, dtype=cg["dtype"])
        outputs = [
            AttentionOutput(
                output=None, indices=empty_long, logits=empty_score,
                weights=empty_score, approx_retrieval_logits=empty_score,
                exact_retrieval_logits=None, search_result=None,
                denominator_indices=None, denominator_logits=None,
                denominator_weights=None,
                selected_count=int(full_count + topk),
                denominator_count=int(denominator_count),
                approx_retrieval_count=int(N),
                exact_retrieval_count=int(topk),
                candidate_count=int(N), probed_list_count=0,
            )
            for _ in range(H * G)
        ]
        return _GroupedQueryDecodeOutput(
            context=context,
            per_query=(
                _GroupedQueryAttentionOutput(
                    output=context[:, :, 0, :],
                    per_stream=tuple(outputs),
                ),
            ),
        )

    def _cg_deferred_append_eager(self) -> None:
        """Fallback when D deferred scatter misses the fused kernel."""
        cg = self._cg
        if cg is None or getattr(self, "_ga_k", None) is None:
            return
        pos = int(self._ga_pos)
        spos = int(self._ga_spos)
        self._shared_base_keys[:, pos, :] = self._ga_k
        self._shared_base_vals[:, pos, :] = self._ga_v
        cg["full_k"][:, spos, :].copy_(self._ga_k)
        cg["full_v"][:, spos, :].copy_(self._ga_v)
        cg["mask"][:, :, spos] = 0.0

    def _cg_update_after_append(self) -> None:
        """After a shared-base append, sync the new row into the static graph buffers.

        Called from append() when the vectorized shared-base path was used.
        The new token row is already in self._shared_base_keys/vals at position
        (first_cache._length - 1). We copy it into the static full-region buffers
        and mark the mask slot as valid (0).
        """
        cg = self._cg
        if cg is None:
            return

        first_cache = self._streams[0].cache
        sink_end = cg["sink_end"]
        FULL_CAP = cg["FULL_CAP"]
        static_pos = int(cg["valid_full"])
        if static_pos >= FULL_CAP:
            self._cg_invalidate()
            return

        assert self._shared_base_keys is not None
        assert self._shared_base_vals is not None
        src_pos = self._paged_ring_pos(int(first_cache._length) - 1)
        cg["full_k"][:, static_pos, :].copy_(self._shared_base_keys[:, src_pos, :])
        cg["full_v"][:, static_pos, :].copy_(self._shared_base_vals[:, src_pos, :])
        cg["mask"][:, :, static_pos] = 0.0
        cg["valid_full"] = static_pos + 1

    def _wmg_eligible(self) -> bool:
        return (
            self._batched_heads is not None
            and self._shared_base_keys is not None
            and not getattr(self, "_shared_base_fallback", True)
            and self._leading_shape is not None
            and self._leading_shape[0] == 1
            and self._key_dim is not None
            and self._value_dim is not None
        )

    def _wmg_prepare_buffers(self) -> bool:
        """Allocate padded full-region buffers for an outer whole-model graph."""
        if getattr(self, "_wmg_ready", False) and self._wmg_eligible():
            return True
        if not self._wmg_eligible():
            return False
        bh = self._batched_heads
        assert bh is not None
        H = bh["H"]
        G = self.num_key_value_groups
        D = self._key_dim
        vD = self._value_dim
        assert D is not None and vD is not None
        dev = bh["packed"].device
        dtype = self._shared_base_keys.dtype
        interval = self.attention_config.index_update_interval
        lw = self.attention_config.local_window
        sink_tokens = self.attention_config.sink_tokens
        first_cache = self._streams[0].cache
        sink_end = min(sink_tokens, first_cache._length)
        full_cap = sink_end + lw + interval
        self._wmg_full_k = torch.zeros(H, full_cap, D, device=dev, dtype=dtype)
        self._wmg_full_v = torch.zeros(H, full_cap, vD, device=dev, dtype=dtype)
        self._wmg_mask = torch.full((H, G, full_cap), float("-inf"), device=dev, dtype=torch.float32)
        self._wmg_mask_zero = torch.zeros(H, G, 1, device=dev, dtype=torch.float32)
        self._wmg_write_idx = torch.zeros(1, device=dev, dtype=torch.long)
        self._wmg_static_pos = torch.zeros(1, device=dev, dtype=torch.long)
        self._wmg_sink_end = sink_end
        self._wmg_full_cap = full_cap
        self._wmg_dtype = dtype
        self._cg_refill_buffers(self._wmg_full_k, self._wmg_full_v, self._wmg_mask, sink_end)
        self._wmg_ready = True
        return True

    def _wmg_invalidate(self) -> None:
        self._wmg_ready = False

    def _wmg_set_write_slots(self) -> None:
        first_cache = self._streams[0].cache
        self._wmg_write_idx.fill_(self._paged_ring_pos(int(first_cache._length)))
        local_count = int(first_cache.regions.local.numel()) + 1
        static_pos = self._wmg_sink_end + local_count - 1
        if static_pos >= self._wmg_full_cap:
            raise RuntimeError(
                f"WMG full-region overflow static_pos={static_pos} cap={self._wmg_full_cap}"
            )
        self._wmg_static_pos.fill_(static_pos)

    def _wmg_attend(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Write current K/V into static slots and run the static batched forward.

        queries/keys/values are [1, heads, 1, dim]. Returns context [1, q_heads, 1, dim].
        """
        bh = self._batched_heads
        assert bh is not None
        H = bh["H"]
        G = self.num_key_value_groups
        D = self._key_dim
        vD = self._value_dim
        assert D is not None and vD is not None
        k = keys[0, :, 0, :]
        v = values[0, :, 0, :]
        self._shared_base_keys.index_copy_(1, self._wmg_write_idx, k.unsqueeze(1))
        self._shared_base_vals.index_copy_(1, self._wmg_write_idx, v.unsqueeze(1))
        self._wmg_full_k.index_copy_(1, self._wmg_static_pos, k.unsqueeze(1))
        self._wmg_full_v.index_copy_(1, self._wmg_static_pos, v.unsqueeze(1))
        self._wmg_mask.index_copy_(2, self._wmg_static_pos, self._wmg_mask_zero)
        return self._cg_forward_static(
            bh,
            H,
            G,
            D,
            vD,
            self._wmg_full_k.device,
            self._wmg_dtype,
            self._wmg_full_k,
            self._wmg_full_v,
            self._wmg_mask,
            queries.contiguous(),
            0,
        )

    def _wmg_notify_after_replay(self) -> None:
        for stream in self._streams:
            stream.cache._notify_external_append()

    def _check_built(self) -> None:
        if not self._streams:
            raise RuntimeError("build_cache must be called before using GQAIVFPQDecodeAttentionAdapter")


def _normalize_gqa_queries(queries: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if queries.ndim == 3:
        return queries.unsqueeze(-2), False
    if queries.ndim == 4:
        return queries, True
    raise ValueError("queries must be [batch, heads, dim] or [batch, heads, query_len, dim]")


def _squeeze_gqa_decode_token(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor
    if tensor.ndim == 4 and tensor.shape[-2] == 1:
        return tensor.squeeze(-2)
    raise ValueError(f"{name} must be [batch, heads, dim] or [batch, heads, 1, dim]")


def _collect_patch_metrics(
    patched: list[torch.nn.Module],
    model: torch.nn.Module,
    *,
    include_per_layer: bool = False,
) -> dict[str, Any]:
    total_stats = _empty_attention_stats()
    memory = _empty_memory_stats()
    profile = {}
    prebuild_profile = {}
    per_layer = []

    for attn in patched:
        stats = dict(attn._e2e_pq_stats)
        layer_memory = _collect_adapter_memory(attn._e2e_pq_adapter)
        layer_profile = _collect_adapter_profile(attn._e2e_pq_adapter)
        layer_prebuild_profile = dict(getattr(attn, "_e2e_pq_prebuild_profile", {}))
        _accumulate_dict(memory, layer_memory)
        _accumulate_dict(profile, layer_profile)
        _accumulate_dict(prebuild_profile, layer_prebuild_profile)
        _accumulate_dict(total_stats, stats)
        if include_per_layer:
            per_layer.append(
                {
                    "layer": int(attn._e2e_pq_layer_idx),
                    "stats": _finalize_attention_stats(stats),
                    "memory": _finalize_memory_stats(layer_memory),
                    "profile": _finalize_profile_stats(layer_profile),
                    "prebuild_profile": _finalize_profile_stats(layer_prebuild_profile),
                }
            )

    metrics = {
        "summary": _finalize_attention_stats(total_stats),
        "memory": _finalize_memory_stats(memory),
        "num_layers": len(model.model.layers),
    }
    if include_per_layer:
        metrics["per_layer"] = per_layer
    if profile:
        metrics["profile"] = _finalize_profile_stats(profile)
    if prebuild_profile:
        metrics["prebuild_profile"] = _finalize_profile_stats(prebuild_profile)
    return metrics


def _empty_attention_stats() -> dict[str, float]:
    return {
        "attention_call_count": 0,
        "stream_query_count": 0,
        "attention_ms": 0.0,
        "selected_token_total": 0,
        "denominator_token_total": 0,
        "retrieval_token_total": 0,
        "exact_retrieval_token_total": 0,
        "candidate_token_total": 0,
        "probed_list_total": 0,
        "approx_score_token_total": 0,
        "search_query_count": 0,
        "tokens_with_retrieval": 0,
    }


def _update_attention_stats(
    stats: dict[str, float],
    adapter_output: Any,
    elapsed_ms: float,
) -> None:
    stats["attention_call_count"] += 1
    stats["attention_ms"] += elapsed_ms
    for query_output in adapter_output.per_query:
        stats["stream_query_count"] += len(query_output.per_stream)
        for stream_output in query_output.per_stream:
            selected_count = getattr(stream_output, "selected_count", None)
            if selected_count is None:
                selected_count = int(stream_output.indices.numel())
            stats["selected_token_total"] += int(selected_count)

            denominator_count = getattr(stream_output, "denominator_count", None)
            if denominator_count is None and stream_output.denominator_indices is not None:
                denominator_count = int(stream_output.denominator_indices.numel())
            if denominator_count is not None:
                stats["denominator_token_total"] += int(denominator_count)

            approx_count = getattr(stream_output, "approx_retrieval_count", None)
            if approx_count is None and stream_output.approx_retrieval_logits is not None:
                approx_count = int(stream_output.approx_retrieval_logits.numel())
            if approx_count is not None:
                stats["approx_score_token_total"] += approx_count
                stats["retrieval_token_total"] += approx_count
                if approx_count > 0:
                    stats["tokens_with_retrieval"] += 1

            exact_count = getattr(stream_output, "exact_retrieval_count", None)
            if exact_count is None and stream_output.exact_retrieval_logits is not None:
                exact_count = int(stream_output.exact_retrieval_logits.numel())
            if exact_count is not None:
                stats["exact_retrieval_token_total"] += int(exact_count)

            candidate_count = getattr(stream_output, "candidate_count", None)
            probed_list_count = getattr(stream_output, "probed_list_count", None)
            if candidate_count is not None or stream_output.search_result is not None:
                stats["search_query_count"] += 1
                if candidate_count is None:
                    candidate_count = int(stream_output.search_result.candidate_indices.numel())
                if probed_list_count is None:
                    probed_list_count = int(stream_output.search_result.probed_lists.numel())
                stats["candidate_token_total"] += int(candidate_count)
                stats["probed_list_total"] += int(probed_list_count)


def _finalize_attention_stats(stats: dict[str, float]) -> dict[str, float]:
    stream_queries = max(1, int(stats["stream_query_count"]))
    search_queries = max(1, int(stats["search_query_count"]))
    retrieval_total = max(1, int(stats["retrieval_token_total"]))
    denominator_total = max(1, int(stats["denominator_token_total"]))
    selected_total = int(stats["selected_token_total"])
    exact_total = int(stats["exact_retrieval_token_total"])
    return {
        **stats,
        "avg_attention_ms_per_layer_call": stats["attention_ms"]
        / max(1, int(stats["attention_call_count"])),
        "avg_selected_tokens_per_stream_query": selected_total / stream_queries,
        "avg_denominator_tokens_per_stream_query": stats["denominator_token_total"]
        / stream_queries,
        "avg_retrieval_tokens_per_stream_query": stats["retrieval_token_total"]
        / stream_queries,
        "avg_exact_retrieval_tokens_per_stream_query": exact_total / stream_queries,
        "avg_candidate_tokens_per_search": stats["candidate_token_total"] / search_queries,
        "avg_probed_lists_per_search": stats["probed_list_total"] / search_queries,
        "exact_retrieval_ratio_over_retrieval": exact_total / retrieval_total,
        "materialized_token_ratio_over_denominator": selected_total / denominator_total,
    }


def _empty_memory_stats() -> dict[str, int]:
    return {
        "key_bytes": 0,
        "value_bytes": 0,
        "index_bytes": 0,
        "retrieval_full_key_bytes": 0,
        "compressed_key_metadata_bytes": 0,
        "dense_kv_bytes": 0,
        "dense_key_bytes": 0,
        "dense_value_bytes": 0,
        "effective_key_bytes_without_retrieval_full_keys": 0,
        "kv_fetches": 0,
        "key_fetches": 0,
        "value_fetches": 0,
        "fetched_key_rows": 0,
        "fetched_value_rows": 0,
        "index_rebuilds": 0,
        "index_adds": 0,
        "online_codebook_updates": 0,
        "codebook_refreshes": 0,
        "deferred_index_updates": 0,
    }


def _collect_adapter_memory(adapter: IVFPQDecodeAttentionAdapter | None) -> dict[str, int]:
    memory = _empty_memory_stats()
    if adapter is None:
        return memory
    seen_index_storages: set[tuple[str, int | None, int, int]] = set()
    for stream in adapter.streams:
        if stream.cache is None:
            continue
        stats = stream.cache.memory_footprint()
        duplicate_index_bytes = _duplicate_shared_index_bytes(
            stream.cache.index,
            seen_index_storages,
        )
        physical_index_bytes = max(0, int(stats.index_bytes) - duplicate_index_bytes)
        memory["key_bytes"] += int(stats.key_bytes)
        memory["value_bytes"] += int(stats.value_bytes)
        memory["index_bytes"] += physical_index_bytes
        memory["retrieval_full_key_bytes"] += int(stats.retrieval_full_key_bytes)
        memory["compressed_key_metadata_bytes"] += max(
            0,
            int(stats.compressed_key_metadata_bytes) - duplicate_index_bytes,
        )
        memory["dense_key_bytes"] += int(stats.key_bytes)
        memory["dense_value_bytes"] += int(stats.value_bytes)
        memory["dense_kv_bytes"] += int(stats.key_bytes + stats.value_bytes)
        memory["effective_key_bytes_without_retrieval_full_keys"] += int(
            stats.key_bytes - stats.retrieval_full_key_bytes + stats.compressed_key_metadata_bytes
        )
        memory["kv_fetches"] += int(stream.cache.kv_fetches)
        memory["key_fetches"] += int(stream.cache.key_fetches)
        memory["value_fetches"] += int(stream.cache.value_fetches)
        memory["fetched_key_rows"] += int(stream.cache.fetched_key_rows)
        memory["fetched_value_rows"] += int(stream.cache.fetched_value_rows)
        memory["index_rebuilds"] += int(stream.cache.index_rebuilds)
        memory["index_adds"] += int(stream.cache.index_adds)
        memory["online_codebook_updates"] += int(stream.cache.online_codebook_updates)
        memory["codebook_refreshes"] += int(stream.cache.codebook_refreshes)
        memory["deferred_index_updates"] += int(getattr(stream.cache, "deferred_index_updates", 0))
    return memory


def _duplicate_shared_index_bytes(
    index: IVFPQIndex | None,
    seen_storages: set[tuple[str, int | None, int, int]],
) -> int:
    if index is None:
        return 0
    duplicate_bytes = _duplicate_tensor_bytes(index.coarse_centroids, seen_storages)
    if index.pq is not None:
        duplicate_bytes += _duplicate_tensor_bytes(index.pq.codebooks, seen_storages)
    return duplicate_bytes


def _duplicate_tensor_bytes(
    tensor: torch.Tensor | None,
    seen_storages: set[tuple[str, int | None, int, int]],
) -> int:
    if tensor is None:
        return 0
    storage = tensor.untyped_storage()
    key = (
        tensor.device.type,
        tensor.device.index,
        int(storage.data_ptr()),
        int(storage.nbytes()),
    )
    if key in seen_storages:
        return tensor.numel() * tensor.element_size()
    seen_storages.add(key)
    return 0


def _collect_adapter_profile(adapter: Any | None) -> dict[str, float]:
    profile: dict[str, float] = {}
    if adapter is None:
        return profile
    for key, value in getattr(adapter, "shared_codebook_profile", {}).items():
        profile[key] = profile.get(key, 0.0) + float(value)
    for stream in getattr(adapter, "streams", ()):
        for key, value in getattr(stream, "profile_stats", {}).items():
            profile[key] = profile.get(key, 0.0) + float(value)
        cache = getattr(stream, "cache", None)
        if cache is None:
            continue
        for key, value in getattr(cache, "profile_stats", {}).items():
            profile[key] = profile.get(key, 0.0) + float(value)
        index = getattr(cache, "index", None)
        if index is None:
            continue
        for key, value in getattr(index, "profile_stats", {}).items():
            profile[key] = profile.get(key, 0.0) + float(value)
    return profile


def _finalize_profile_stats(profile: dict[str, float]) -> dict[str, float]:
    finalized = dict(profile)
    for key, value in list(profile.items()):
        if not key.endswith("_ms"):
            continue
        prefix = key[: -len("_ms")]
        calls = profile.get(f"{prefix}_calls", 0.0)
        if calls > 0:
            finalized[f"{prefix}_avg_ms"] = value / calls
    return finalized


def _finalize_memory_stats(memory: dict[str, int]) -> dict[str, float | int]:
    dense_kv = max(1, memory["dense_kv_bytes"])
    dense_key = max(1, memory["dense_key_bytes"])
    retrieval_full_key = max(1, memory["retrieval_full_key_bytes"])
    physical_total = memory["key_bytes"] + memory["value_bytes"] + memory["index_bytes"]
    return {
        **memory,
        "physical_total_bytes": physical_total,
        "physical_total_vs_dense_kv_ratio": physical_total / dense_kv,
        "pq_metadata_vs_retrieval_full_key_ratio": memory["compressed_key_metadata_bytes"]
        / retrieval_full_key,
        "effective_key_ratio_without_retrieval_full_keys": memory[
            "effective_key_bytes_without_retrieval_full_keys"
        ]
        / dense_key,
    }


def _compare_decode_metrics(dense: dict[str, Any], sparse: dict[str, Any]) -> dict[str, float]:
    dense_logits = dense["logits"]
    sparse_logits = sparse["logits"]
    dense_probs = torch.softmax(dense_logits, dim=-1)
    sparse_log_probs = torch.log_softmax(sparse_logits, dim=-1)
    dense_log_probs = torch.log_softmax(dense_logits, dim=-1)
    kl = (dense_probs * (dense_log_probs - sparse_log_probs)).sum(dim=-1).mean()
    rel_l2 = (
        torch.linalg.vector_norm((sparse_logits - dense_logits).float(), dim=-1)
        / torch.linalg.vector_norm(dense_logits.float(), dim=-1).clamp_min(1e-12)
    ).mean()
    dense_top1 = dense_logits.argmax(dim=-1)
    sparse_top1 = sparse_logits.argmax(dim=-1)
    dense_top10 = torch.topk(dense_logits, k=min(10, dense_logits.shape[-1]), dim=-1).indices
    sparse_top10 = torch.topk(sparse_logits, k=min(10, sparse_logits.shape[-1]), dim=-1).indices
    top1_agreement = (dense_top1 == sparse_top1).float().mean()
    dense_top1_in_sparse_top10 = (sparse_top10 == dense_top1.unsqueeze(-1)).any(dim=-1).float().mean()
    sparse_top1_in_dense_top10 = (dense_top10 == sparse_top1.unsqueeze(-1)).any(dim=-1).float().mean()
    return {
        "nll_delta": float(sparse["nll"] - dense["nll"]),
        "ppl_delta": float(sparse["ppl"] - dense["ppl"]),
        "ppl_ratio": float(sparse["ppl"] / dense["ppl"]) if dense["ppl"] != 0 else math.inf,
        "logit_rel_l2": float(rel_l2.item()),
        "dense_to_sparse_kl": float(kl.item()),
        "top1_agreement": float(top1_agreement.item()),
        "dense_top1_in_sparse_top10": float(dense_top1_in_sparse_top10.item()),
        "sparse_top1_in_dense_top10": float(sparse_top1_in_dense_top10.item()),
    }


def _strip_logits(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "logits"}


def _sweep_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    base_index = {
        "num_lists": _one_int(args.num_lists),
        "nprobe": _one_int(args.nprobe),
        "num_subspaces": _one_int(args.subspaces),
        "num_bits": _one_int(args.bits),
        "residual": True,
        "coarse_max_iter": _one_int(args.coarse_iter),
        "pq_max_iter": _one_int(args.pq_iter),
        "pack_codes": args.pack_codes,
        "topk_block_size": _one_optional_int(args.topk_block_size),
        "kernel_backend": _one_str(args.kernel_backend),
        "rotation": _one_str(args.rotation),
        "direction_normalize": args.direction_normalize,
        "online_codebook_lr": _one_float(args.online_codebook_lr),
        "seed": args.seed,
    }
    base_attention = {
        "sink_tokens": _one_int(args.sink),
        "local_window": _one_int(args.local_window),
        "retrieval_topk": _one_int(args.retrieval_topk),
        "retrieval_top_fraction": _one_optional_float(args.retrieval_top_fraction),
        "retrieval_top_p": _one_optional_float(args.retrieval_top_p),
        "retrieval_top_p_scope": _one_str(args.retrieval_top_p_scope),
        "nprobe": _one_int(args.nprobe),
        "candidate_budget": _one_optional_int(args.candidate_budget),
        "exact_rerank": args.exact_rerank,
        "scale": None,
        "mode": "hybrid",
        "hybrid_value_mode": _one_str(args.hybrid_value_mode),
        "hybrid_topk_source": _one_str(args.hybrid_topk_source),
        "hybrid_denominator_source": _one_str(args.hybrid_denominator_source),
        "index_update_interval": _one_int(args.index_update_interval),
        "index_update_strategy": _one_str(args.index_update_strategy),
        "codebook_refresh_interval": _one_optional_int(args.codebook_refresh_interval),
        "kv_storage": args.kv_storage,
        "pin_offloaded_kv": args.pin_offloaded_kv,
        "prefetch_full_kv": args.prefetch_full_kv,
        "collect_attention_details": args.collect_attention_details,
        "profile_attention_components": args.profile_attention_components,
    }

    if args.grid_json is not None:
        with open(args.grid_json, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError("--grid-json must contain a JSON array")
        configs = []
        for idx, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError("each grid JSON item must be an object")
            index = dict(base_index)
            attention = dict(base_attention)
            name = str(item.get("name", f"cfg{idx:03d}"))
            _apply_grid_item(index, attention, item)
            _sync_shared_fields(index, attention)
            configs.append({"name": name, "index": index, "attention": attention})
        return configs

    grid_fields = {
        "num_lists": _parse_int_list(args.num_lists),
        "nprobe": _parse_int_list(args.nprobe),
        "num_subspaces": _parse_int_list(args.subspaces),
        "num_bits": _parse_int_list(args.bits),
        "coarse_max_iter": _parse_int_list(args.coarse_iter),
        "pq_max_iter": _parse_int_list(args.pq_iter),
        "topk_block_size": _parse_optional_int_list(args.topk_block_size),
        "kernel_backend": _parse_str_list(args.kernel_backend),
        "rotation": _parse_str_list(args.rotation),
        "online_codebook_lr": _parse_float_list(args.online_codebook_lr),
        "sink_tokens": _parse_int_list(args.sink),
        "local_window": _parse_int_list(args.local_window),
        "retrieval_topk": _parse_int_list(args.retrieval_topk),
        "retrieval_top_fraction": _parse_optional_float_list(args.retrieval_top_fraction),
        "retrieval_top_p": _parse_optional_float_list(args.retrieval_top_p),
        "retrieval_top_p_scope": _parse_str_list(args.retrieval_top_p_scope),
        "candidate_budget": _parse_optional_int_list(args.candidate_budget),
        "hybrid_value_mode": _parse_str_list(args.hybrid_value_mode),
        "hybrid_topk_source": _parse_str_list(args.hybrid_topk_source),
        "hybrid_denominator_source": _parse_str_list(args.hybrid_denominator_source),
        "index_update_interval": _parse_int_list(args.index_update_interval),
        "index_update_strategy": _parse_str_list(args.index_update_strategy),
        "codebook_refresh_interval": _parse_optional_int_list(args.codebook_refresh_interval),
    }

    configs = []
    keys = list(grid_fields)
    for idx, values in enumerate(itertools.product(*(grid_fields[key] for key in keys))):
        flat = dict(zip(keys, values, strict=True))
        index = dict(base_index)
        attention = dict(base_attention)
        for key, value in flat.items():
            if key in SWEEP_INDEX_KEYS:
                index[key] = value
            if key in SWEEP_ATTENTION_KEYS:
                attention[key] = value
        _sync_shared_fields(index, attention)
        name = (
            f"cfg{idx:03d}_L{index['num_lists']}_P{index['nprobe']}_"
            f"M{index['num_subspaces']}_B{index['num_bits']}_"
            f"R{attention['retrieval_top_fraction']}_C{attention['candidate_budget']}"
        )
        configs.append({"name": name, "index": index, "attention": attention})
    return configs


def _apply_grid_item(
    index: dict[str, Any],
    attention: dict[str, Any],
    item: dict[str, Any],
) -> None:
    nested_index = item.get("index", {})
    nested_attention = item.get("attention", {})
    if nested_index is not None and not isinstance(nested_index, dict):
        raise ValueError("grid item 'index' must be an object")
    if nested_attention is not None and not isinstance(nested_attention, dict):
        raise ValueError("grid item 'attention' must be an object")

    for key, value in item.items():
        if key in {"name", "index", "attention"}:
            continue
        normalized = _normalize_alias(key)
        if normalized in SWEEP_INDEX_KEYS:
            index[normalized] = value
        if normalized in SWEEP_ATTENTION_KEYS:
            attention[normalized] = value

    for key, value in nested_index.items():
        index[_normalize_alias(key)] = value
    for key, value in nested_attention.items():
        attention[_normalize_alias(key)] = value


def _normalize_alias(key: str) -> str:
    aliases = {
        "lists": "num_lists",
        "num-lists": "num_lists",
        "subspaces": "num_subspaces",
        "bits": "num_bits",
        "coarse_iter": "coarse_max_iter",
        "pq_iter": "pq_max_iter",
        "sink": "sink_tokens",
        "retrieval_fraction": "retrieval_top_fraction",
        "top_fraction": "retrieval_top_fraction",
        "topk": "retrieval_topk",
    }
    return aliases.get(key, key.replace("-", "_"))


def _sync_shared_fields(index: dict[str, Any], attention: dict[str, Any]) -> None:
    attention["nprobe"] = index["nprobe"]
    if attention.get("retrieval_top_fraction") is not None and attention.get("retrieval_top_p") is not None:
        raise ValueError("retrieval_top_fraction and retrieval_top_p are mutually exclusive")


def _parse_int_list(value: str) -> list[int]:
    return [int(item) for item in _split_list(value)]


def _parse_optional_int_list(value: str) -> list[int | None]:
    return [None if _is_none(item) else int(item) for item in _split_list(value)]


def _parse_float_list(value: str) -> list[float]:
    return [float(item) for item in _split_list(value)]


def _parse_optional_float_list(value: str) -> list[float | None]:
    return [None if _is_none(item) else float(item) for item in _split_list(value)]


def _parse_str_list(value: str) -> list[str]:
    return _split_list(value)


def _split_list(value: str) -> list[str]:
    items = [item.strip() for item in str(value).split(",")]
    return [item for item in items if item != ""]


def _is_none(value: str) -> bool:
    return value.lower() in {"none", "null", "nil"}


def _one_int(value: str) -> int:
    return _parse_int_list(value)[0]


def _one_optional_int(value: str) -> int | None:
    return _parse_optional_int_list(value)[0]


def _one_float(value: str) -> float:
    return _parse_float_list(value)[0]


def _one_optional_float(value: str) -> float | None:
    return _parse_optional_float_list(value)[0]


def _one_str(value: str) -> str:
    return _parse_str_list(value)[0]


def _profile_metric(profile: dict[str, Any], key: str) -> float:
    value = profile.get(key, 0.0)
    if value is None:
        return 0.0
    return float(value)


def _summary_row(result: dict[str, Any]) -> dict[str, Any]:
    pq = result["pq_hsa"]
    comparison = result["comparison"]
    stats = pq["attention_stats"]["summary"]
    memory = pq["attention_stats"]["memory"]
    decode_profile = pq["attention_stats"].get("profile", {})
    prebuild_profile = pq["attention_stats"].get("prebuild_profile", {})
    index_config = pq["index_config"]
    attention_config = pq["attention_config"]
    dense_ms_per_token = result["dense"]["ms_per_token"]
    pq_ms_per_token = pq["ms_per_token"]
    pq_total_ms_per_token = pq.get("ms_per_token_with_prebuild", pq_ms_per_token)
    shared_codebook_build_ms = _profile_metric(
        prebuild_profile,
        "shared_codebook_index_build_total_ms",
    )
    shared_stream_build_ms = _profile_metric(
        prebuild_profile,
        "index_build_shared_total_ms",
    )
    per_stream_index_build_ms = _profile_metric(
        prebuild_profile,
        "index_build_total_ms",
    )
    return {
        "config_id": result["config_id"],
        "name": result["name"],
        "share_layer_pq_codebook": result.get("dataset", {}).get(
            "share_layer_pq_codebook",
            False,
        ),
        "dense_ppl": result["dense"]["ppl"],
        "pq_ppl": pq["ppl"],
        "ppl_ratio": comparison["ppl_ratio"],
        "nll_delta": comparison["nll_delta"],
        "logit_rel_l2": comparison["logit_rel_l2"],
        "dense_to_sparse_kl": comparison["dense_to_sparse_kl"],
        "top1_agreement": comparison["top1_agreement"],
        "exact_retrieval_ratio": stats["exact_retrieval_ratio_over_retrieval"],
        "avg_retrieval_tokens": stats["avg_retrieval_tokens_per_stream_query"],
        "avg_exact_retrieval_tokens": stats["avg_exact_retrieval_tokens_per_stream_query"],
        "avg_candidate_tokens": stats["avg_candidate_tokens_per_search"],
        "avg_probed_lists": stats["avg_probed_lists_per_search"],
        "pq_eval_ms": pq["eval_ms"],
        "pq_ms_per_token": pq["ms_per_token"],
        "sparse_prebuild_ms": pq.get("prebuild_ms", 0.0),
        "sparse_warmup_ms": pq.get("warmup_ms", 0.0),
        "pq_total_with_prebuild_ms": pq.get("total_with_prebuild_ms", pq["eval_ms"]),
        "pq_total_with_prebuild_and_warmup_ms": pq.get(
            "total_with_prebuild_and_warmup_ms",
            pq.get("total_with_prebuild_ms", pq["eval_ms"]),
        ),
        "pq_ms_per_token_with_prebuild": pq_total_ms_per_token,
        "pq_ms_per_token_with_prebuild_and_warmup": pq.get(
            "ms_per_token_with_prebuild_and_warmup",
            pq_total_ms_per_token,
        ),
        "dense_ms_per_token": dense_ms_per_token,
        "pq_vs_dense_speedup": (
            dense_ms_per_token / pq_ms_per_token if pq_ms_per_token != 0 else math.inf
        ),
        "pq_vs_dense_slowdown": (
            pq_ms_per_token / dense_ms_per_token if dense_ms_per_token != 0 else math.inf
        ),
        "pq_total_vs_dense_speedup": (
            dense_ms_per_token / pq_total_ms_per_token
            if pq_total_ms_per_token != 0
            else math.inf
        ),
        "pq_total_vs_dense_slowdown": (
            pq_total_ms_per_token / dense_ms_per_token
            if dense_ms_per_token != 0
            else math.inf
        ),
        "attention_ms": stats["attention_ms"],
        "prebuild_cache_build_index_total_ms": _profile_metric(
            prebuild_profile,
            "cache_build_index_total_ms",
        ),
        "prebuild_index_build_total_ms": _profile_metric(
            prebuild_profile,
            "index_build_total_ms",
        ),
        "prebuild_shared_codebook_build_total_ms": shared_codebook_build_ms,
        "prebuild_shared_stream_build_total_ms": shared_stream_build_ms,
        "prebuild_effective_index_build_total_ms": (
            per_stream_index_build_ms + shared_codebook_build_ms + shared_stream_build_ms
        ),
        "prebuild_index_coarse_kmeans_ms": _profile_metric(
            prebuild_profile,
            "index_build_coarse_kmeans_ms",
        ),
        "prebuild_shared_codebook_coarse_kmeans_ms": _profile_metric(
            prebuild_profile,
            "shared_codebook_index_build_coarse_kmeans_ms",
        ),
        "prebuild_index_pq_train_ms": _profile_metric(
            prebuild_profile,
            "index_build_pq_train_ms",
        ),
        "prebuild_shared_codebook_pq_train_ms": _profile_metric(
            prebuild_profile,
            "shared_codebook_index_build_pq_train_ms",
        ),
        "prebuild_index_pq_encode_ms": _profile_metric(
            prebuild_profile,
            "index_build_pq_encode_ms",
        ),
        "prebuild_shared_codebook_pq_encode_ms": _profile_metric(
            prebuild_profile,
            "shared_codebook_index_build_pq_encode_ms",
        ),
        "prebuild_shared_stream_pq_encode_ms": _profile_metric(
            prebuild_profile,
            "index_build_shared_pq_encode_ms",
        ),
        "prebuild_shared_stream_assign_lists_ms": _profile_metric(
            prebuild_profile,
            "index_build_shared_assign_lists_ms",
        ),
        "prebuild_index_inverted_lists_ms": _profile_metric(
            prebuild_profile,
            "index_build_inverted_lists_ms",
        ),
        "prebuild_shared_stream_inverted_lists_ms": _profile_metric(
            prebuild_profile,
            "index_build_shared_inverted_lists_ms",
        ),
        "prebuild_cache_fetch_keys_ms": _profile_metric(
            prebuild_profile,
            "cache_fetch_keys_ms",
        ),
        "prebuild_cache_fetch_values_ms": _profile_metric(
            prebuild_profile,
            "cache_fetch_values_ms",
        ),
        "prebuild_cache_value_centroids_ms": _profile_metric(
            prebuild_profile,
            "cache_build_value_centroids_ms",
        ),
        "decode_batched_retrieval_scores_ms": _profile_metric(
            decode_profile,
            "batched_retrieval_scores_ms",
        ),
        "decode_index_pq_compute_lut_ms": _profile_metric(
            decode_profile,
            "index_pq_compute_lut_ms",
        ),
        "decode_index_pq_lut_scan_ms": _profile_metric(
            decode_profile,
            "index_pq_lut_scan_ms",
        ),
        "decode_index_coarse_list_scores_ms": _profile_metric(
            decode_profile,
            "index_coarse_list_scores_ms",
        ),
        "decode_select_tokens_ms": _profile_metric(
            decode_profile,
            "select_tokens_ms",
        ),
        "decode_fetch_kv_wrapper_ms": _profile_metric(
            decode_profile,
            "fetch_kv_ms",
        ),
        "decode_cache_fetch_kv_total_ms": _profile_metric(
            decode_profile,
            "cache_fetch_kv_total_ms",
        ),
        "decode_cache_fetch_kv_keys_ms": _profile_metric(
            decode_profile,
            "cache_fetch_kv_keys_ms",
        ),
        "decode_cache_fetch_kv_values_ms": _profile_metric(
            decode_profile,
            "cache_fetch_kv_values_ms",
        ),
        "decode_vectorized_softmax_ms": _profile_metric(
            decode_profile,
            "vectorized_softmax_ms",
        ),
        "decode_vectorized_exact_logits_ms": _profile_metric(
            decode_profile,
            "vectorized_exact_logits_ms",
        ),
        "decode_vectorized_exact_output_ms": _profile_metric(
            decode_profile,
            "vectorized_exact_output_ms",
        ),
        "num_lists": index_config["num_lists"],
        "nprobe": index_config["nprobe"],
        "subspaces": index_config["num_subspaces"],
        "bits": index_config["num_bits"],
        "candidate_budget": attention_config["candidate_budget"],
        "retrieval_top_fraction": attention_config["retrieval_top_fraction"],
        "retrieval_topk": attention_config["retrieval_topk"],
        "local_window": attention_config["local_window"],
        "hybrid_value_mode": attention_config["hybrid_value_mode"],
        "hybrid_topk_source": attention_config["hybrid_topk_source"],
        "hybrid_denominator_source": attention_config["hybrid_denominator_source"],
        "collect_attention_details": attention_config["collect_attention_details"],
        "profile_attention_components": attention_config["profile_attention_components"],
        "index_update_strategy": attention_config["index_update_strategy"],
        "index_update_interval": attention_config["index_update_interval"],
        "physical_total_vs_dense_kv_ratio": memory["physical_total_vs_dense_kv_ratio"],
        "pq_metadata_vs_retrieval_full_key_ratio": memory[
            "pq_metadata_vs_retrieval_full_key_ratio"
        ],
        "effective_key_ratio_without_retrieval_full_keys": memory[
            "effective_key_ratio_without_retrieval_full_keys"
        ],
        "index_rebuilds": memory["index_rebuilds"],
        "index_adds": memory["index_adds"],
        "online_codebook_updates": memory["online_codebook_updates"],
        "codebook_refreshes": memory["codebook_refreshes"],
        "deferred_index_updates": memory["deferred_index_updates"],
    }


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_tokens(
    tokenizer: Any,
    *,
    dataset_name: str,
    dataset_config: str,
    split: str,
    text_lines: int,
    token_count: int,
) -> torch.Tensor:
    local_parquet = os.environ.get("PQ_DATASET_PARQUET")
    if local_parquet:
        dataset = load_dataset("parquet", data_files=local_parquet, split="train")
    else:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    lines = [row["text"] for row in dataset if row.get("text") and row["text"].strip()]
    text = "\n\n".join(lines[:text_lines])
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
    if encoded.shape[1] < token_count:
        raise ValueError(f"dataset produced {encoded.shape[1]} tokens, need {token_count}")
    return encoded[:, :token_count]


def _legacy_kv_pairs(cache: DynamicCache) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Pairwise (K, V) per layer. transformers>=5 removed DynamicCache.to_legacy_cache()."""
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer in cache.layers:
        if layer.keys is None or layer.values is None:
            raise RuntimeError("dense cache layer is missing keys/values")
        pairs.append((layer.keys, layer.values))
    return tuple(pairs)


def _clone_cache(cache: DynamicCache) -> DynamicCache:
    cloned = tuple((key.detach().clone(), value.detach().clone()) for key, value in _legacy_kv_pairs(cache))
    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(cloned)
    return _cache_from_cpu_snapshot(tuple((key, value, str(key.device)) for key, value in cloned))


def _cache_to_cpu_snapshot(cache: DynamicCache) -> tuple[tuple[torch.Tensor, torch.Tensor, str], ...]:
    snapshot = []
    for key, value in _legacy_kv_pairs(cache):
        snapshot.append(
            (
                key.detach().to("cpu", copy=True),
                value.detach().to("cpu", copy=True),
                str(key.device),
            )
        )
    return tuple(snapshot)


def _cache_from_cpu_snapshot(
    snapshot: tuple[tuple[torch.Tensor, torch.Tensor, str], ...],
) -> DynamicCache:
    # Build the cache by assigning each layer's full K/V directly, instead of
    # DynamicCache.from_legacy_cache (which calls layer.update -> torch.cat onto
    # an empty tensor, transiently allocating a second copy of the whole layer).
    # At 256K that extra ~1 GiB/layer copy OOMs the GPU holding that layer.
    cache = DynamicCache()
    for layer_idx, (key, value, device) in enumerate(snapshot):
        target = torch.device(device)
        while len(cache.layers) <= layer_idx:
            cache.layers.append(DynamicLayer())
        layer = cache.layers[layer_idx]
        key_gpu = key.to(target, non_blocking=True)
        value_gpu = value.to(target, non_blocking=True)
        layer.lazy_initialization(key_gpu, value_gpu)
        layer.keys = key_gpu
        layer.values = value_gpu
    return cache


def _accumulate_dict(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if key not in target:
            target[key] = 0
        target[key] += value


def _time_start(device: torch.device) -> float:
    if device.type == "cuda":
        _synchronize_all_cuda()
    return time.perf_counter()


def _time_stop(device: torch.device, start: float) -> float:
    if device.type == "cuda":
        _synchronize_all_cuda()
    return (time.perf_counter() - start) * 1000.0


def _wall_time_start() -> float:
    return time.perf_counter()


def _wall_time_stop(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _synchronize_all_cuda() -> None:
    if not torch.cuda.is_available():
        return
    for idx in range(torch.cuda.device_count()):
        torch.cuda.synchronize(idx)


def _max_memory_gb_any() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return max(torch.cuda.max_memory_allocated(idx) for idx in range(torch.cuda.device_count())) / (
        1024**3
    )


def _reset_peak_memory_all() -> None:
    if not torch.cuda.is_available():
        return
    for idx in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(idx)


def _model_input_device(model: torch.nn.Module, *, fallback: torch.device) -> torch.device:
    embeddings = getattr(getattr(model, "model", None), "embed_tokens", None)
    if embeddings is not None and isinstance(getattr(embeddings, "weight", None), torch.Tensor):
        return embeddings.weight.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback


if __name__ == "__main__":
    main()
