#!/usr/bin/env python3
"""Decode speed of PQ-HSA inside vLLM (same stack, same engine), two clocks.

One engine per process; PQ-HSA is toggled in-process (dense <-> PQ) on the
same prompt, so both arms share weights, KV cache and scheduler. Every timed
generate() is a prefix-cache hit, i.e. the timed window is decode-only.

Stack: vLLM 0.8.5.post1 V1 + FLASH_ATTN, enforce_eager, max_num_seqs=1 (bs=1),
fp16, chunked prefill, prefix caching on, cascade attention off. The optimized
kernel configuration (ADOPTED_ENV below, identical to
``pq_hsa_vllm.config.ADOPTED_KERNEL_ENV``) is applied on top of an all-off
base; the remaining switches (paged KV, flash-kmeans, ...) come from the
environment -- source ``scripts/pqhsa_env.sh`` first.

Clocks (never summed across):
  e2e ms/tok        host wall (cuda.synchronize around generate) / n_out over an
                    ``--e2e-steps`` window (default 512 = 2 x the index-update
                    interval of 256), whole engine.
  attention segment CUDA events on the compute stream (benchmarks.cuda_event_log):
                    dense = dense_fa; PQ = replay_gpu + append + copy_ctx.
                    reshape_and_cache_flash is excluded (both arms pay it).
                    At TP>1 the events are recorded inside each worker and the
                    per-rank mean is reported.

Receipts written to the json:
  * per-rank pq_decode_calls (must be > 0 on every rank)
  * per-rank capture-time ``_cuda_attend_only_hits`` summed over layer adapters
    (the valid receipt that the CUDA attend epilogue ran; the steady-window
    counter stays 0 under CUDA graphs)
  * per-layer index health (graph captured, finite index, empty/max IVF lists)
  * worker-side env check of every applied switch

Quality gate: passkey x8 (16 tokens), dense vs PQ on the SAME prompt token ids:
  main gate  = needle hit in both arms + delta(PQ - dense) >= -0.05 + key-span top-1 >= 0.9
  extra gate = 16-token full-window top-1 >= 0.9 (reported, never decisive)

Example (one GPU, Llama-3.1-8B-Instruct, 128K):
  source scripts/pqhsa_env.sh 0 l8b_128k l8b
  python benchmarks/speed/decode_speed.py --line l8b --tp 1 --ctx 130400 --mem 0.85 --tag l8b_128k
"""
from __future__ import annotations

import argparse, gc, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("PQ_HSA_SIDECAR", "fast")
os.environ.setdefault("PQ_USE_CUDA_GRAPH", "1")
os.environ.setdefault("PQ_CG_SENTINEL_INTERVAL", "0")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

WORKER_CLS = "benchmarks.vllm_backend.pq_hsa_worker.PQHSAWorker"

_BASE = {
    "PQ_HSA_FUSED_DECODE": "0", "PQ_HSA_NATIVE_BACKEND": "0", "PQ_HSA_FP16_LUT": "0",
    "PQ_HSA_HOST_SYNC_FIX": "1", "PQ_HSA_INCREMENTAL_FLUSH": "1", "PQ_HSA_BLOCK_TOPK": "0",
    "PQ_HSA_CUDA_FUSED": "0", "PQ_HSA_FLUSH_NO_RECAPTURE": "0", "PQ_HSA_LEAN_WRAP": "0",
    "PQ_HSA_GRAPH_OUT_DIRECT": "0", "PQ_HSA_GRAPH_APPEND": "0",
    "PQ_HSA_CUDA_ATTEND_ONLY": "0", "PQ_HSA_FUSED_RADIX": "0", "PQ_HSA_FUSED_FULL": "0",
    "PQ_HSA_LEAN_APPEND": "0", "PQ_HSA_LEAN_COPYCTX": "0",
    "PQ_HSA_PARSTREAM": "0", "PQ_HSA_FUSED_LUTPREP": "0", "PQ_HSA_FUSED_BLOCKRED": "0",
    "PQ_HSA_ATTEND_RAWIDX": "0", "PQ_HSA_MASKSKIP": "0", "PQ_HSA_LEAN_REPLAY": "0",
    "PQ_HSA_SCAN_TOPK": "0",
}

# Model lines. ``model`` is a Hugging Face id; pass --model to use a local path.
# All lines use native RoPE (no YaRN / rope override) and no thinking mode.
LINES = {
    "l8b": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "alias": "Meta-Llama-3.1-8B-Instruct",
        "n_layers": 32, "q_heads": 32, "kv_heads": 8, "gqa_group": 4,
        "cuda_instance": "pq_hsa_fused_h20_l_g4 (-DPQ_HSA_KG=4)",
        "note": "native RoPE (llama3 scaling in config), max_position_embeddings=131072.",
    },
    "gradient": {
        "model": "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
        "alias": "Llama-3-8B-Instruct-Gradient-1048k",
        "n_layers": 32, "q_heads": 32, "kv_heads": 8, "gqa_group": 4,
        "cuda_instance": "pq_hsa_fused_h20_l_g4 (-DPQ_HSA_KG=4)",
        "note": ("native long-context RoPE (large rope_theta, rope_scaling=null), "
                 "max_position_embeddings=1048576."),
    },
    "q14": {
        "model": "Qwen/Qwen2.5-14B-Instruct-1M",
        "alias": "Qwen2.5-14B-Instruct-1M",
        "n_layers": 48, "q_heads": 40, "kv_heads": 8, "gqa_group": 5,
        "cuda_instance": "pq_hsa_fused_h20_l_g5 (-DPQ_HSA_KG=5)",
        "note": ("native RoPE: max_position_embeddings=1,010,000, rope_scaling=null, "
                 "rope_theta=1e7.  vLLM 0.8.5.post1 does not implement Dual Chunk "
                 "Attention, so this stack is RoPE-only."),
    },
    "q3_30b": {
        "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "alias": "Qwen3-30B-A3B-Instruct-2507",
        "n_layers": 48, "q_heads": 32, "kv_heads": 4, "gqa_group": 8,
        "cuda_instance": "pq_hsa_fused_h20_l_g8 (-DPQ_HSA_KG=8)",
        "note": ("native RoPE: max_position_embeddings=262144, rope_scaling=null, "
                 "rope_theta=1e7.  MoE 128 experts top-8, 30.5B total / 3.3B active."),
    },
}

NEEDLES = ["42891735", "71928346", "30581624", "86420917",
           "19283746", "65748392", "31415926", "90817263"]

# Optimized CUDA graph-body configuration (19 switches; identical to
# pq_hsa_vllm.config.ADOPTED_KERNEL_ENV and to scripts/pqhsa_env.sh).
ADOPTED_ENV = {
    "PQ_HSA_CUDA_ATTEND_ONLY": "1", "PQ_HSA_CUDA_ATTEND_SORT": "0",
    "PQ_HSA_CUDA_SPLITS": "8", "PQ_HSA_CUDA_PARTWARPS": "8", "PQ_HSA_CUDA_PARTUNROLL": "8",
    "PQ_HSA_FP16_LUT": "1", "PQ_HSA_CUDA_REDWARPS": "4",
    "PQ_HSA_CUDA_BGSPLITS": "8", "PQ_HSA_CUDA_BGWARPS": "8", "PQ_HSA_CUDA_BGUNROLL": "8",
    "PQ_HSA_CUDA_BGMM": "0",
    "PQ_HSA_LEAN_APPEND": "1", "PQ_HSA_LEAN_COPYCTX": "1", "PQ_HSA_PARSTREAM": "1",
    "PQ_HSA_FUSED_LUTPREP": "1", "PQ_HSA_FUSED_BLOCKRED": "1", "PQ_HSA_ATTEND_RAWIDX": "1",
    "PQ_HSA_MASKSKIP": "1", "PQ_HSA_LEAN_REPLAY": "1",
}


def _adopted_env() -> dict:
    assert len(ADOPTED_ENV) == 19, len(ADOPTED_ENV)
    return dict(ADOPTED_ENV)


# ---------------------------------------------------------------- RPC bodies
def _rpc_enable(self, on: bool, flush: int):
    import os as _os
    _os.environ["PQ_HSA_ENABLE"] = "1" if on else "0"
    _os.environ["PQ_HSA_FLUSH_INTERVAL"] = str(int(flush))
    return {"rank": int(getattr(self, "rank", 0)), "enable": _os.environ["PQ_HSA_ENABLE"]}


def _rpc_stats(self):
    from benchmarks.vllm_backend.pq_hsa_decode_runtime import pq_hsa_runtime_stats
    import os as _os
    return {"rank": int(getattr(self, "rank", 0)), "pid": _os.getpid(),
            "pq_hsa_enable": _os.environ.get("PQ_HSA_ENABLE", "0"),
            **pq_hsa_runtime_stats()}


def _rpc_reset_stats(self):
    from benchmarks.vllm_backend.pq_hsa_decode_runtime import reset_pq_hsa_runtime_stats
    reset_pq_hsa_runtime_stats()
    return True


def _rpc_reset_layers(self):
    from benchmarks.vllm_backend.pq_hsa_decode_runtime import reset_pq_hsa_layer_state
    reset_pq_hsa_layer_state()
    return True


def _rpc_events_on(self):
    import os as _os
    from benchmarks.cuda_event_log import enable, reset
    _os.environ["PQ_HSA_STEADY_EVENTS"] = "1"
    enable(); reset()
    return int(getattr(self, "rank", 0))


def _rpc_events_off(self):
    import os as _os
    from benchmarks.cuda_event_log import summarize, disable, reset
    us = dict(summarize())
    disable(); reset()
    _os.environ["PQ_HSA_STEADY_EVENTS"] = "0"
    return {"rank": int(getattr(self, "rank", 0)), "us": us}


def _attend_receipt_body(rank: int) -> dict:
    """capture-time hits + the per-rank sidecar head shape.

    `cuda_attend_only_calls` in STATS is NOT usable: with PQ_USE_CUDA_GRAPH=1 the
    python epilogue only runs during graph capture, so the steady-window counter is
    0 even when the CUDA tail is active.  Per-adapter `_cuda_attend_only_hits`
    (set at capture time) is the real receipt.
    """
    from benchmarks.vllm_backend.pq_hsa_decode_runtime import _LAYER_STATE
    hits, errs, n_layers = 0, {}, 0
    n_graph = 0  # per-layer CUDA-graph receipt (sidecar.adapter._cg is not None)
    shapes = {}
    for stt in _LAYER_STATE.values():
        ad = getattr(stt.get("sidecar"), "adapter", None)
        sc = stt.get("sidecar")
        if ad is None:
            continue
        n_layers += 1
        n_graph += int(getattr(ad, "_cg", None) is not None)
        hits += int(getattr(ad, "_cuda_attend_only_hits", 0) or 0)
        e = getattr(ad, "_cuda_attend_only_error", None)
        if e:
            errs[str(e)] = errs.get(str(e), 0) + 1
        qh = getattr(sc, "num_query_heads", None) or getattr(ad, "num_query_heads", None)
        kh = getattr(sc, "num_kv_heads", None) or getattr(ad, "num_kv_heads", None)
        if qh is not None and kh is not None:
            shapes[f"{int(qh)}/{int(kh)}"] = shapes.get(f"{int(qh)}/{int(kh)}", 0) + 1
    try:
        from pq_hsa.kernels import triton_lut_scan_h20 as _tls
        path_hits = dict(getattr(_tls, "_PAIR_PATH_HITS", {}) or {})
    except Exception as exc:  # pragma: no cover
        path_hits = {"error": f"{type(exc).__name__}: {exc}"}
    return {"rank": rank, "layers_with_sidecar": n_layers,
            "layers_graph_active": n_graph,
            "cuda_attend_only_hits_total": hits,
            "lut_scan_pair_path_hits": path_hits,
            "cuda_attend_only_used": hits > 0,
            "cuda_attend_only_errors": errs,
            "sidecar_head_shapes_q_over_kv": shapes,
            "receipt_note": ("hits = sum over layer adapters of capture-time "
                             "_cuda_attend_only_hits; 0 means the CUDA tail never "
                             "ran and PQ used the all-aten graph body")}


def _index_health_body() -> dict:
    """Read-only receipt: per layer, is the sidecar graph captured, and is the built index
    finite?  Locates non-finite values: index side (coarse centroids / PQ codebooks / value
    centroids / key norms, empty IVF lists) vs graph-body output (index finite but capture fails)."""
    import torch as _t
    from benchmarks.vllm_backend.pq_hsa_decode_runtime import _LAYER_STATE
    rows = []
    for li, stt in enumerate(_LAYER_STATE.values()):
        sc = stt.get("sidecar")
        ad = getattr(sc, "adapter", None)
        if ad is None:
            continue
        bh = getattr(ad, "_batched_heads", None) or {}
        row = {"layer": li, "graph": getattr(ad, "_cg", None) is not None, "N": bh.get("N")}
        for k in ("coarse", "codebooks", "value_centroids", "knorm", "rotation"):
            v = bh.get(k)
            if isinstance(v, _t.Tensor) and v.is_floating_point():
                row[f"nonfinite_{k}"] = int((~_t.isfinite(v)).sum().item())
        io = bh.get("inv_offsets")
        if isinstance(io, _t.Tensor) and io.numel() > 1:
            sz = (io[..., 1:] - io[..., :-1]).to(_t.int64)
            row["empty_lists"] = int((sz == 0).sum().item())
            row["max_list"] = int(sz.max().item())
            row["num_lists"] = int(sz.shape[-1])
        rows.append(row)
    bad = [r["layer"] for r in rows if not r["graph"]]
    nf = [r["layer"] for r in rows if any(v for k, v in r.items() if k.startswith("nonfinite_"))]
    return {"n_layers": len(rows), "layers_no_graph": bad, "layers_index_nonfinite": nf,
            "max_list_graph_layers": max([r.get("max_list") or 0 for r in rows if r["graph"]] or [0]),
            "max_list_nograph_layers": max([r.get("max_list") or 0 for r in rows if not r["graph"]] or [0]),
            "empty_lists_total": sum(int(r.get("empty_lists") or 0) for r in rows),
            "per_layer": rows}


def _rpc_attend_receipt(self):
    return _attend_receipt_body(int(getattr(self, "rank", 0)))


def _rpc_mem(self):
    import torch as _t
    free, total = _t.cuda.mem_get_info()
    return {"rank": int(getattr(self, "rank", 0)),
            "alloc_gib": round(_t.cuda.memory_allocated() / 2**30, 3),
            "max_alloc_gib": round(_t.cuda.max_memory_allocated() / 2**30, 3),
            "reserved_gib": round(_t.cuda.memory_reserved() / 2**30, 3),
            "device_used_gib": round((total - free) / 2**30, 3),
            "device_total_gib": round(total / 2**30, 3)}


def _rpc_env_check(self, keys: list):
    import os as _os
    return {"rank": int(getattr(self, "rank", 0)),
            "env": {k: _os.environ.get(k) for k in keys}}


# ---------------------------------------------------------------- engine facade
class Arm:
    def __init__(self, llm, tp: int):
        self.llm, self.tp = llm, tp

    def _rpc(self, fn, *args):
        return self.llm.collective_rpc(fn, timeout=3600.0, args=args)

    def set_pq(self, on: bool, flush: int = 0) -> None:
        os.environ["PQ_HSA_ENABLE"] = "1" if on else "0"
        os.environ["PQ_HSA_FLUSH_INTERVAL"] = str(int(flush))
        if self.tp > 1:
            self._rpc(_rpc_enable, on, int(flush))

    def stats(self) -> list:
        if self.tp > 1:
            return list(self._rpc(_rpc_stats))
        from benchmarks.vllm_backend.pq_hsa_flashattn_wrap import pq_hsa_runtime_stats
        return [{"rank": 0, "pid": os.getpid(), **pq_hsa_runtime_stats()}]

    def reset_stats(self) -> None:
        if self.tp > 1:
            self._rpc(_rpc_reset_stats)
        else:
            from benchmarks.vllm_backend.pq_hsa_flashattn_wrap import reset_pq_hsa_runtime_stats
            reset_pq_hsa_runtime_stats()

    def reset_layers(self) -> None:
        if self.tp > 1:
            self._rpc(_rpc_reset_layers)
        else:
            from benchmarks.vllm_backend.pq_hsa_decode_runtime import reset_pq_hsa_layer_state
            reset_pq_hsa_layer_state()

    def events_on(self) -> None:
        os.environ["PQ_HSA_STEADY_EVENTS"] = "1"
        if self.tp > 1:
            self._rpc(_rpc_events_on)
        else:
            from benchmarks.cuda_event_log import enable, reset
            enable(); reset()

    def events_off(self) -> list:
        if self.tp > 1:
            rows = list(self._rpc(_rpc_events_off))
        else:
            from benchmarks.cuda_event_log import summarize, disable, reset
            rows = [{"rank": 0, "us": dict(summarize())}]
            disable(); reset()
        os.environ["PQ_HSA_STEADY_EVENTS"] = "0"
        return rows

    def attend_receipt(self) -> list:
        if self.tp > 1:
            return list(self._rpc(_rpc_attend_receipt))
        return [_attend_receipt_body(0)]

    def env_check(self, keys: list) -> list:
        if self.tp > 1:
            return list(self._rpc(_rpc_env_check, list(keys)))
        return [{"rank": 0, "env": {k: os.environ.get(k) for k in keys}}]

    def mem(self) -> list:
        if self.tp > 1:
            return list(self._rpc(_rpc_mem))
        import torch
        free, total = torch.cuda.mem_get_info()
        return [{"rank": 0, "alloc_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
                 "max_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                 "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3),
                 "device_used_gib": round((total - free) / 2**30, 3),
                 "device_total_gib": round(total / 2**30, 3)}]


def _sync() -> None:
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)


def _gen(llm, prompt_ids, n: int) -> dict:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    params = SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True)
    _sync(); t0 = time.perf_counter()
    outs = llm.generate([TokensPrompt(prompt_token_ids=list(prompt_ids))],
                        sampling_params=params, use_tqdm=False)
    _sync(); wall = time.perf_counter() - t0
    comp = outs[0].outputs[0]
    return {"token_ids": list(comp.token_ids), "text": comp.text,
            "wall_s": wall, "n_out": len(comp.token_ids)}


def _ev_per_tok(rows: list, n_out: int) -> dict:
    out = {}
    for r in rows:
        out[str(r.get("rank", 0))] = {k: round(v / 1000.0 / max(1, n_out), 4)
                                      for k, v in (r.get("us") or {}).items()}
    return out


def _mean_phase(per_rank: dict, phase: str):
    vals = [v.get(phase) for v in per_rank.values() if v.get(phase) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _pick(st: dict, keys) -> dict:
    return {k: st.get(k) for k in keys}


_STAT_KEYS = ("rank", "pid", "pq_decode_calls", "layers_built", "cuda_graph_active",
              "batched_heads_active", "cuda_attend_only_calls", "cuda_fused_calls",
              "fused_decode_calls", "dense_fallback_calls", "pq_flush_calls",
              "pq_append_calls", "reject", "last_error",
              # Harness receipts (no numeric effect): index-build / restore wall
              # so every json carries stats.pq_build_s_total.
              "pq_build_calls", "pq_build_s_total", "pq_restore_s_total",
              "pq_flush_s_total", "prefix_persist_hits", "prefix_persist_misses")


def _flash_kmeans_receipt() -> dict:
    """Did the flash-kmeans / batched-build path actually load in this process?"""
    mods = sorted(m for m in sys.modules if m.startswith("flash_kmeans"))
    try:
        from pq_hsa.index import kmeans as _km
        cached = getattr(_km, "_FLASH_KMEANS_MODULE", None) is not None
    except Exception as exc:  # pragma: no cover
        cached = f"{type(exc).__name__}: {exc}"
    return {"flash_kmeans_modules_loaded": mods,
            "kmeans_l2_flash_module_cached": cached,
            "batched_build_kernels_imported": "flash_kmeans.assign_euclid_triton" in sys.modules,
            "env_PQ_HSA_FLASH_KMEANS": os.environ.get("PQ_HSA_FLASH_KMEANS"),
            "env_PQ_HSA_BATCHED_BUILD": os.environ.get("PQ_HSA_BATCHED_BUILD")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", required=True, choices=sorted(LINES))
    ap.add_argument("--model", default=None,
                    help="override the line's model (HF id or local path)")
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--mem", type=float, required=True)
    ap.add_argument("--mml", type=int, default=0)
    ap.add_argument("--mbt", type=int, default=8192)
    ap.add_argument("--e2e-steps", type=int, default=512)
    ap.add_argument("--event-steps", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--flush-interval", type=int, default=256)
    ap.add_argument("--top-frac", type=float, default=0.01)
    ap.add_argument("--quality-tok", type=int, default=16)
    ap.add_argument("--n-needles", type=int, default=8)
    ap.add_argument("--skip-speed", action="store_true")
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default=str(REPO / "results" / "speed"))
    args = ap.parse_args()

    cfg = dict(LINES[args.line])
    if args.model:
        cfg["model"] = args.model
    tp = int(args.tp)
    N_LAYERS = cfg["n_layers"]

    vis = [int(x) for x in (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")
           if x.strip() != ""]
    if not vis:
        raise SystemExit("set CUDA_VISIBLE_DEVICES explicitly")
    if len(vis) != tp:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={vis} does not match --tp {tp}")

    OUT = Path(args.out_dir)
    OUTJ = OUT / f"{args.tag}.json"
    if OUTJ.exists():
        raise SystemExit(f"REFUSE to overwrite existing {OUTJ}")

    os.environ["MODEL_INST"] = cfg["model"]
    adopted = _adopted_env()
    os.environ.update(_BASE)
    os.environ.update(adopted)
    # Lets G=5 / G=8 / G=16 dispatch to the -DPQ_HSA_KG={5,8,16} instances.
    # Instances are compiled lazily on first use.
    os.environ["PQ_HSA_CUDA_GQA_MULTI"] = "1"
    os.environ["PQ_HSA_CUDA_GQA_SET"] = "4,5,8,16"
    os.environ["PQ_HSA_TOP_FRAC"] = str(args.top_frac)
    # Extra opt-in flags whose arrival in every worker is receipted in the json
    # (set by scripts/pqhsa_env.sh).
    _extra_check_keys = [k for k in os.environ.get("PQ_HSA_EXTRA_ENV_CHECK", "").split(",")
                         if k.strip()]
    os.environ["PQ_HSA_ENABLE"] = "0"
    os.environ["PQ_HSA_FLUSH_INTERVAL"] = "0"
    os.environ["PQ_HSA_STEADY_EVENTS"] = "0"

    if tp > 1:
        from benchmarks.vllm_backend.worker_hook import apply_worker_hook_env
        apply_worker_hook_env(OUT)
        os.environ["PQ_HSA_STATS_PREFIX"] = f"{args.tag}_rank"
        os.environ.setdefault("MASTER_PORT", os.environ.get("PQ_MASTER_PORT", "29851"))

    mml = args.mml or (args.ctx + args.e2e_steps + 2 * args.warmup + 16)

    import torch, vllm
    from transformers import AutoTokenizer
    from passkey import make_passkey_prompt

    dev = torch.cuda.get_device_properties(0)
    P = {
        "task": "decode speed, same-stack dense vs PQ-HSA",
        "line": args.line, "model": cfg["model"], "model_alias": cfg["alias"],
        "model_note": cfg["note"],
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tp": tp, "batch_size": 1, "enforce_eager": True, "dtype": "float16",
        "worker_hook": (WORKER_CLS if tp > 1 else "install_pq_hsa_flashattn_backend (in-process)"),
        "device": {"name": dev.name, "sm": dev.multi_processor_count},
        "vllm": vllm.__version__, "torch": torch.__version__,
        "stack": f"vLLM {vllm.__version__} V1 + FLASH_ATTN",
        "ctx": args.ctx, "max_model_len": mml, "gpu_mem_util": args.mem,
        "max_num_batched_tokens": args.mbt,
        "e2e_steps": args.e2e_steps, "event_steps": args.event_steps,
        "warmup": args.warmup, "flush_interval": args.flush_interval,
        "top_frac": args.top_frac,
        "adopted_env": adopted,
        "adopted_env_source": "ADOPTED_ENV in this script (19 items)",
        "extra_env": {"PQ_HSA_CUDA_GQA_MULTI": "1",
                      "PQ_HSA_CUDA_GQA_SET": os.environ.get("PQ_HSA_CUDA_GQA_SET"),
                      "PQ_HSA_TOP_FRAC": str(args.top_frac),
                      **{k: os.environ.get(k) for k in _extra_check_keys}},
        "n_layers": N_LAYERS, "q_heads": cfg["q_heads"], "kv_heads": cfg["kv_heads"],
        "gqa_group": cfg["gqa_group"], "cuda_instance": cfg["cuda_instance"],
        "yarn": False, "rope_scaling_override": None, "hf_overrides": None,
        "no_config_1m_overlay": True, "no_thinking": True,
        "calibers": {
            "e2e_ms_per_tok": "host wall (cuda.synchronize) / n_out, whole engine, bs=1, decode-only",
            "*_events_ms": ("CUDA Event on the compute stream (benchmarks.cuda_event_log); "
                            "at TP>1 recorded inside each worker => PER RANK"),
            "attention_segment": ("dense=dense_fa ; PQ=replay_gpu+append+copy_ctx "
                                  "(reshape_and_cache_flash excluded: both arms pay it)"),
            "ms_per_layer": f"attention segment / {N_LAYERS} layers",
            "never_summed": "attention segment and e2e are different calibers",
            "tp_note": ("at TP>1 each rank owns kv_heads/TP KV heads; the sidecar is built "
                        "from impl.num_kv_heads / impl.num_heads, i.e. the TP-local head "
                        "counts -> the sidecar is sharded by head across ranks, and the GQA "
                        "group is unchanged.  attend_receipt.sidecar_head_shapes_q_over_kv "
                        "is the measured evidence."),
        },
    }
    print(f"[speed] tag={args.tag} line={args.line} ctx={args.ctx} tp={tp} "
          f"mml={mml} mem={args.mem} p={args.top_frac}", flush=True)

    def flush_json():
        OUTJ.parent.mkdir(parents=True, exist_ok=True)
        tmp = OUTJ.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(P, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(OUTJ)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)

    # Opt-in: vLLM 0.8.5 MultiprocExecutor hard-codes a 40 s per-step RPC timeout;
    # the FIRST PQ decode step at long context (paged gather + sidecar build over
    # all layers of a large-model rank) can exceed it.  In-process engine
    # (VLLM_ENABLE_V1_MULTIPROCESSING=0) => patching the module constant is enough.
    _rpc_to = os.environ.get("PQ_HSA_RPC_TIMEOUT_S", "").strip()
    if _rpc_to:
        import vllm.v1.executor.multiproc_executor as _mpe
        _mpe.EXECUTE_MODEL_TIMEOUT_S = float(_rpc_to)
        print(f"[speed] EXECUTE_MODEL_TIMEOUT_S -> {_mpe.EXECUTE_MODEL_TIMEOUT_S}", flush=True)
    from vllm import LLM
    if tp == 1:
        from benchmarks.vllm_backend.pq_hsa_flashattn_wrap import install_pq_hsa_flashattn_backend
        install_pq_hsa_flashattn_backend()
    kw = dict(model=cfg["model"], dtype="float16", max_model_len=mml,
              max_num_batched_tokens=args.mbt, max_num_seqs=1,
              gpu_memory_utilization=args.mem, enable_chunked_prefill=True,
              enforce_eager=True, enable_prefix_caching=True,
              disable_cascade_attn=True, trust_remote_code=True,
              disable_log_stats=True)
    if tp > 1:
        kw.update(tensor_parallel_size=tp, worker_cls=WORKER_CLS)
    t_load = time.perf_counter()
    try:
        llm = LLM(**kw)
    except Exception as exc:
        P["engine_error"] = f"{type(exc).__name__}: {exc}"
        P["oom"] = ("memory" in str(exc).lower()) or ("kv cache" in str(exc).lower())
        P["status"] = "n.m."
        flush_json()
        print(f"[speed] ENGINE FAILED: {P['engine_error']}", flush=True)
        sys.exit(4)
    P["engine_load_s"] = round(time.perf_counter() - t_load, 2)
    arm = Arm(llm, tp)

    try:
        cc = llm.llm_engine.cache_config
        P["kv_cache_tokens"] = int(cc.num_gpu_blocks) * int(cc.block_size or 16)
    except Exception:
        P["kv_cache_tokens"] = None
    P["mem_after_load"] = arm.mem()
    ev = arm.env_check(sorted(adopted) + ["PQ_HSA_CUDA_GQA_MULTI", "PQ_HSA_TOP_FRAC"]
                       + _extra_check_keys)
    P["worker_env_check"] = ev
    P["adopted_env_match_all_ranks"] = all(
        all(r["env"].get(k) == v for k, v in adopted.items())
        and r["env"].get("PQ_HSA_CUDA_GQA_MULTI") == "1"
        and r["env"].get("PQ_HSA_TOP_FRAC") == str(args.top_frac)
        for r in ev)
    print(f"[speed] engine {P['engine_load_s']}s kv={P['kv_cache_tokens']} "
          f"env_match={P['adopted_env_match_all_ranks']}", flush=True)
    flush_json()

    status = "ok"
    try:
        if not args.skip_speed:
            prompt = [1] * args.ctx

            # ---------------- dense ----------------
            arm.set_pq(False, 0)
            t0 = time.perf_counter()
            _gen(llm, prompt, 1)                      # cold prefill -> prefix cache
            P["prefill_s_first_call"] = round(time.perf_counter() - t0, 2)
            _gen(llm, prompt, args.warmup)
            d = _gen(llm, prompt, args.e2e_steps)
            P["dense_e2e"] = {
                "ms_per_tok": round(1000.0 * d["wall_s"] / max(1, d["n_out"]), 4),
                "n_out": d["n_out"], "steps_requested": args.e2e_steps,
                "window_ge_2delta": d["n_out"] >= 2 * args.flush_interval}
            arm.events_on()
            de = _gen(llm, prompt, args.event_steps)
            P["dense_events_ms_by_rank"] = _ev_per_tok(arm.events_off(), de["n_out"])
            P["dense_fa_ms_per_tok"] = _mean_phase(P["dense_events_ms_by_rank"], "dense_fa")
            P["mem_dense"] = arm.mem()
            print(f"[speed] dense e2e={P['dense_e2e']['ms_per_tok']} "
                  f"dense_fa={P['dense_fa_ms_per_tok']} "
                  f"byrank={P['dense_events_ms_by_rank']}", flush=True)
            flush_json()

            # ---------------- PQ, flush 0 (attention-segment caliber) ----------------
            arm.set_pq(True, 0)
            arm.reset_stats()
            t0 = time.perf_counter()
            _gen(llm, prompt, 1)
            P["pq_index_build_first_call_s"] = round(time.perf_counter() - t0, 2)
            _gen(llm, prompt, args.warmup); _gen(llm, prompt, args.warmup)
            # The index is built on the first DECODE step (the 1-token call above
            # is prefill-only), so read the build wall here, before the reset.
            _wst = arm.stats()
            P["pq_warmup_stats_by_rank"] = [_pick(r, _STAT_KEYS) for r in _wst]
            P["stats"] = {
                "pq_build_s_total": round(sum(float(r.get("pq_build_s_total") or 0.0)
                                              for r in _wst), 4),
                "pq_build_calls": sum(int(r.get("pq_build_calls") or 0) for r in _wst),
                "pq_restore_s_total": round(sum(float(r.get("pq_restore_s_total") or 0.0)
                                                for r in _wst), 4),
                "prefix_persist_hits": sum(int(r.get("prefix_persist_hits") or 0) for r in _wst),
                "source": "PQ first call + 2 warmup gens (index build happens on first decode step)"}
            if tp == 1:
                P["flash_kmeans_receipt"] = _flash_kmeans_receipt()
            arm.reset_stats()
            p = _gen(llm, prompt, args.e2e_steps)
            rows = arm.stats()
            P["pq_e2e_flush0"] = {
                "ms_per_tok": round(1000.0 * p["wall_s"] / max(1, p["n_out"]), 4),
                "n_out": p["n_out"], "flush_interval": 0,
                "stats_by_rank": [_pick(r, _STAT_KEYS) for r in rows]}
            P["pq_decode_calls_by_rank"] = {str(r.get("rank")): int(r.get("pq_decode_calls") or 0)
                                            for r in rows}
            P["layers_built_by_rank"] = {str(r.get("rank")): r.get("layers_built") for r in rows}
            P["all_ranks_pq_active"] = (len(rows) == tp and
                                        all(int(r.get("pq_decode_calls") or 0) > 0 for r in rows))
            P["layer_coverage_by_rank"] = {
                str(r.get("rank")): round(float(r.get("pq_decode_calls") or 0)
                                          / max(1, p["n_out"] * N_LAYERS), 4) for r in rows}

            arm.events_on(); arm.reset_stats()
            pe = _gen(llm, prompt, args.event_steps)
            P["pq_events_ms_by_rank"] = _ev_per_tok(arm.events_off(), pe["n_out"])
            P["pq_events_stats_by_rank"] = [_pick(r, _STAT_KEYS) for r in arm.stats()]
            P["attend_receipt_by_rank"] = arm.attend_receipt()
            if tp == 1:
                try:
                    P["index_health"] = _index_health_body()
                except Exception as _ih_exc:  # receipt only; never affects the run
                    P["index_health"] = {"error": f"{type(_ih_exc).__name__}: {_ih_exc}"}
            P["attend_hits_total_all_ranks"] = sum(
                int(r.get("cuda_attend_only_hits_total") or 0)
                for r in P["attend_receipt_by_rank"])
            P["cuda_fast_path_used_all_ranks"] = all(
                bool(r.get("cuda_attend_only_used")) for r in P["attend_receipt_by_rank"]) \
                and len(P["attend_receipt_by_rank"]) == tp
            P["mem_pq"] = arm.mem()
            print(f"[speed] pq(flush0) e2e={P['pq_e2e_flush0']['ms_per_tok']} "
                  f"calls={P['pq_decode_calls_by_rank']} "
                  f"attend={[r.get('cuda_attend_only_hits_total') for r in P['attend_receipt_by_rank']]} "
                  f"shapes={[r.get('sidecar_head_shapes_q_over_kv') for r in P['attend_receipt_by_rank']]}",
                  flush=True)
            flush_json()

            # ---------------- PQ with the deferred flush amortized (Delta=256) --------
            arm.set_pq(True, args.flush_interval)
            arm.reset_stats()
            _gen(llm, prompt, 1); _gen(llm, prompt, args.warmup)
            arm.reset_stats()
            pf = _gen(llm, prompt, args.e2e_steps)
            frows = arm.stats()
            P["pq_e2e_flush"] = {
                "ms_per_tok": round(1000.0 * pf["wall_s"] / max(1, pf["n_out"]), 4),
                "n_out": pf["n_out"], "flush_interval": args.flush_interval,
                "window_ge_2delta": pf["n_out"] >= 2 * args.flush_interval,
                "flush_calls_by_rank": {str(r.get("rank")): r.get("pq_flush_calls")
                                        for r in frows},
                "flush_s_total_by_rank": {str(r.get("rank")):
                                          round(float(r.get("pq_flush_s_total") or 0.0), 4)
                                          for r in frows},
                "stats_by_rank": [_pick(r, _STAT_KEYS) for r in frows]}
            arm.set_pq(True, 0)

            # ---------------- derived ----------------
            fa = float(P.get("dense_fa_ms_per_tok") or 0.0)
            parts = {k: _mean_phase(P["pq_events_ms_by_rank"], k)
                     for k in ("replay_gpu", "append", "copy_ctx")}
            pq_attn = sum(float(v or 0.0) for v in parts.values())
            per_rank_seg = {}
            for rk, ph in P["pq_events_ms_by_rank"].items():
                s = sum(float(ph.get(k) or 0.0) for k in ("replay_gpu", "append", "copy_ctx"))
                dfa = float((P["dense_events_ms_by_rank"].get(rk) or {}).get("dense_fa") or 0.0)
                per_rank_seg[rk] = {
                    "dense_fa_ms_per_tok": round(dfa, 4),
                    "dense_fa_ms_per_layer": round(dfa / N_LAYERS, 5),
                    "pq_attn_ms_per_tok": round(s, 4),
                    "pq_attn_ms_per_layer": round(s / N_LAYERS, 5),
                    "speedup_dense_over_pq": round(dfa / s, 4) if s else None}
            P["attention_segment"] = {
                "caliber": "CUDA Event (dense_fa vs replay_gpu+append+copy_ctx); per-rank mean at TP>1",
                "tp": tp, "per_rank": per_rank_seg,
                "dense_fa_ms_per_tok": round(fa, 4),
                "dense_fa_ms_per_layer": round(fa / N_LAYERS, 5),
                "replay_gpu_ms_per_tok": parts["replay_gpu"],
                "append_ms_per_tok": parts["append"],
                "copy_ctx_ms_per_tok": parts["copy_ctx"],
                "pq_attn_ms_per_tok": round(pq_attn, 4),
                "pq_attn_ms_per_layer": round(pq_attn / N_LAYERS, 5),
                "speedup_dense_over_pq": round(fa / pq_attn, 4) if pq_attn else None,
                "pq_faster": bool(pq_attn and pq_attn < fa)}
            d2 = P["dense_e2e"]["ms_per_tok"]
            P["e2e_summary"] = {
                "dense_ms_per_tok": d2,
                "pq_flush0_ms_per_tok": P["pq_e2e_flush0"]["ms_per_tok"],
                "pq_flush_ms_per_tok": P["pq_e2e_flush"]["ms_per_tok"],
                "speedup_dense_over_pq_flush0":
                    round(d2 / P["pq_e2e_flush0"]["ms_per_tok"], 4),
                "speedup_dense_over_pq_flush":
                    round(d2 / P["pq_e2e_flush"]["ms_per_tok"], 4)}
            P["valid_pq_arm"] = bool(P.get("all_ranks_pq_active"))
            print(f"[speed] ATTN dense={fa:.4f} pq={pq_attn:.4f} "
                  f"speedup={P['attention_segment']['speedup_dense_over_pq']}x | "
                  f"E2E {P['e2e_summary']}", flush=True)
            flush_json()

        # ---------------- passkey x8 ----------------
        if not args.skip_quality:
            arm.set_pq(False, 0)
            recs = []
            for i, needle in enumerate(NEEDLES[:args.n_needles]):
                try:
                    t0 = time.perf_counter()
                    _, ids = make_passkey_prompt(tokenizer, args.ctx, needle=needle)
                    tb = round(time.perf_counter() - t0, 2)
                    arm.set_pq(False, 0)
                    dq = _gen(llm, ids, args.quality_tok)
                    arm.set_pq(True, 0); arm.reset_stats()
                    pq = _gen(llm, ids, args.quality_tok)
                    qrows = arm.stats()
                    m = min(len(dq["token_ids"]), len(pq["token_ids"]))
                    agree = sum(int(a == b) for a, b in
                                zip(dq["token_ids"][:m], pq["token_ids"][:m])) / max(1, m)
                    kspan, ktop1 = None, None
                    for t in range(1, m + 1):
                        if needle in tokenizer.decode(dq["token_ids"][:t]):
                            kspan = t; break
                    if kspan:
                        ktop1 = sum(int(a == b) for a, b in
                                    zip(dq["token_ids"][:kspan], pq["token_ids"][:kspan])) / kspan
                    rec = {"i": i, "needle": needle, "ctx": args.ctx,
                           "prompt_build_s": tb, "n_tok": args.quality_tok,
                           "dense_text": dq["text"], "pq_text": pq["text"],
                           "dense_needle": needle in (dq["text"] or ""),
                           "pq_needle": needle in (pq["text"] or ""),
                           "key_span_tokens": kspan,
                           "key_span_top1": round(float(ktop1), 4) if ktop1 is not None else None,
                           "full16_top1": round(float(agree), 4),
                           "pq_decode_calls_by_rank": {str(r.get("rank")):
                                                       int(r.get("pq_decode_calls") or 0)
                                                       for r in qrows},
                           "last_error": next((r.get("last_error") for r in qrows
                                               if r.get("last_error")), None)}
                except Exception as exc:
                    import traceback; traceback.print_exc()
                    rec = {"i": i, "needle": needle, "ctx": args.ctx,
                           "error": f"{type(exc).__name__}: {exc}"}
                recs.append(rec)
                print(f"[speed] passkey[{i}] {needle} dense={rec.get('dense_needle')} "
                      f"pq={rec.get('pq_needle')} keytop1={rec.get('key_span_top1')} "
                      f"full16={rec.get('full16_top1')} err={rec.get('error')}", flush=True)
                P["passkey_x8"] = recs
                flush_json()
            arm.set_pq(False, 0)
            ok = [r for r in recs if "error" not in r]
            dh = sum(int(bool(r["dense_needle"])) for r in ok)
            ph = sum(int(bool(r["pq_needle"])) for r in ok)
            kt = [r["key_span_top1"] for r in ok if r.get("key_span_top1") is not None]
            fw = [r["full16_top1"] for r in ok]
            n = len(ok)
            P["passkey_summary"] = {
                "n_run": len(recs), "n_ok": n, "needles": NEEDLES[:args.n_needles],
                "dense_needle_hits": dh, "pq_needle_hits": ph,
                "dense_hit_rate": round(dh / n, 4) if n else None,
                "pq_hit_rate": round(ph / n, 4) if n else None,
                "delta_pq_minus_dense": round((ph - dh) / n, 4) if n else None,
                "key_span_top1_mean": round(sum(kt) / len(kt), 4) if kt else None,
                "key_span_top1_min": min(kt) if kt else None,
                "full16_top1_mean": round(sum(fw) / len(fw), 4) if fw else None,
                "full16_top1_min": min(fw) if fw else None,
                "MAIN_GATE": "needle hit both arms + delta >= -0.05 + key-span top1 >= 0.9",
                "MAIN_GATE_pass": bool(n and ph == n and dh == n and kt
                                       and min(kt) >= 0.9 and ((ph - dh) / n) >= -0.05),
                "EXTRA_GATE": "16-token full-window top1 >= 0.9 (never decisive)",
                "EXTRA_GATE_pass": bool(fw and min(fw) >= 0.9)}
            print(f"[speed] PASSKEY {P['passkey_summary']}", flush=True)
    except Exception as exc:
        import traceback; traceback.print_exc()
        P["fatal_error"] = f"{type(exc).__name__}: {exc}"
        P["oom"] = "out of memory" in str(exc).lower()
        status = "n.m."
    finally:
        P["status"] = status
        try:
            P["mem_final"] = arm.mem()
        except Exception:
            P["mem_final"] = None
        flush_json()
        try:
            if tp > 1:
                llm.collective_rpc(_rpc_reset_layers, timeout=900.0)
            core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
            if core is not None and hasattr(core, "shutdown"):
                core.shutdown()
            del llm
        except Exception:
            pass
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    print(f"[speed] wrote {OUTJ} status={status}", flush=True)
    if status != "ok":
        sys.exit(5)


if __name__ == "__main__":
    main()
