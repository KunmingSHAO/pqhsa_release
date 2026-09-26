#!/usr/bin/env python3
"""Decode speed of PQ-HSA loaded through the vLLM plugin entry point.

Works on vLLM 0.8.5.post1 and 0.29.0: the plugin is loaded ONLY through vLLM's
own ``load_general_plugins()`` (``VLLM_PLUGINS=pq_hsa_vllm`` +
``PQ_HSA_VLLM=1``); this script never calls the installer itself. One engine,
fp16, per context:

  dense: e2e ms/tok (host wall / n_out, prefix-cache hit => decode-only) +
            CUDA-event attention segment (dense_fa)
  PQ f0: PQ-HSA with index-update interval 0 in the timed window
            (attention segment = replay_gpu + append + copy_ctx)
  PQ f256: PQ-HSA with index-update interval 256 (amortized-update e2e)
  passkey x N per context, dense vs PQ (16 tokens, needle hit + top-1 agreement)

Numbers from different vLLM versions are separate stacks and should not be
subtracted or ratioed across versions.

Usage (one GPU; on vLLM >= 0.10 point CUDA_HOME at a CUDA toolchain matching
torch, e.g. the nvidia/cu13 wheel directory, and use a fresh extension cache):
  CUDA_VISIBLE_DEVICES=0 PQ_HSA_CUDA_EXT_DIR=/tmp/torch_extensions_pqhsa_v029 \
  python benchmarks/speed/plugin_speed.py --model meta-llama/Llama-3.1-8B-Instruct \
    --ctxs 32768,126720 --steps 256 --warmup 8 --passkey-runs 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
# On vLLM 0.29 (torch 2.13+cu130) CUDA_HOME must point at a CUDA 13 toolchain;
# the launcher sets it. Fallback keeps the 0.8.5 default.
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("PQ_HSA_CUDA_EXT_DIR", "/tmp/torch_extensions_pqhsa")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# The two switches that load the plugin from a cold process:
os.environ.setdefault("VLLM_PLUGINS", "pq_hsa_vllm")
os.environ.setdefault("PQ_HSA_VLLM", "1")
# Sidecar CUDA graph on, sentinel off (same as decode_speed.py).
os.environ.setdefault("PQ_USE_CUDA_GRAPH", "1")
os.environ.setdefault("PQ_CG_SENTINEL_INTERVAL", "0")

N_LAYERS = 32  # overwritten from the model config in main()
MODEL = os.environ.get("MODEL_INST", "meta-llama/Llama-3.1-8B-Instruct")
NEEDLE = "42891735"


def _vllm_version() -> tuple[int, ...]:
    import vllm

    out = []
    for tok in str(vllm.__version__).split("."):
        d = ""
        for ch in tok:
            if ch.isdigit():
                d += ch
            else:
                break
        out.append(int(d) if d else 0)
    return tuple(out)


def _sync() -> None:
    import torch

    torch.cuda.synchronize()


def _set_pq(on: bool) -> None:
    os.environ["PQ_HSA_ENABLE"] = "1" if on else "0"


def _build_llm(max_model_len: int, mem: float, mbt: int | None = None, tp: int = 1):
    from vllm import LLM

    kw = dict(
        model=MODEL,
        dtype="float16",
        max_model_len=max_model_len,
        max_num_batched_tokens=int(mbt) if mbt else max_model_len,  # chunked prefill for >=256K
        max_num_seqs=1,
        tensor_parallel_size=int(tp),
        gpu_memory_utilization=mem,
        enforce_eager=True,
        enable_prefix_caching=True,
        disable_cascade_attn=True,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    if _vllm_version() >= (0, 10):
        # VLLM_ATTENTION_BACKEND env was removed; the backend is an engine arg.
        kw["attention_backend"] = "FLASH_ATTN"
    else:
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    return LLM(**kw)


def _gen(llm, prompt_ids: list[int], n: int) -> dict:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    params = SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True)
    _sync()
    t0 = time.perf_counter()
    outs = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids)], sampling_params=params, use_tqdm=False
    )
    _sync()
    wall = time.perf_counter() - t0
    comp = outs[0].outputs[0]
    return {
        "token_ids": list(comp.token_ids),
        "text": comp.text,
        "wall_s": wall,
        "n_out": len(comp.token_ids),
    }


def _destroy(llm) -> None:
    import gc

    import torch

    try:
        core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        if core is not None and hasattr(core, "shutdown"):
            core.shutdown()
    except Exception:
        pass
    try:
        del llm
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _passkey(tokenizer, ctx: int, seed: int) -> list[int]:
    from passkey import make_passkey_prompt

    _, ids = make_passkey_prompt(tokenizer, ctx, needle=NEEDLE)
    return ids


def _events_ms_per_tok(summary: dict, n_out: int) -> dict:
    return {k: round(v / 1000.0 / max(1, n_out), 4) for k, v in summary.items()}


def run_ctx(llm, tokenizer, ctx: int, args, rt) -> dict:
    from benchmarks.cuda_event_log import disable, enable, reset, summarize

    R: dict = {"ctx": ctx}
    prompt = [1] * ctx

    # ---------------- dense ----------------
    _set_pq(False)
    os.environ["PQ_HSA_FLUSH_INTERVAL"] = "0"
    _gen(llm, prompt, 1)  # cold prefill -> prefix cache
    _gen(llm, prompt, args.warmup)
    d = _gen(llm, prompt, args.steps)
    R["dense_e2e_ms_per_tok"] = round(1000.0 * d["wall_s"] / max(1, d["n_out"]), 4)
    os.environ["PQ_HSA_STEADY_EVENTS"] = "1"
    enable(); reset()
    de = _gen(llm, prompt, args.steps)
    R["dense_events_ms"] = _events_ms_per_tok(summarize(), de["n_out"])
    disable(); reset(); os.environ["PQ_HSA_STEADY_EVENTS"] = "0"
    print(f"[plugin_speed] ctx={ctx} dense e2e={R['dense_e2e_ms_per_tok']} events={R['dense_events_ms']}", flush=True)

    # ---------------- PQ, flush 0 (attention-segment caliber) ----------------
    _set_pq(True)
    os.environ["PQ_HSA_FLUSH_INTERVAL"] = "0"
    rt.reset_pq_hsa_runtime_stats()
    _gen(llm, prompt, 1)  # first PQ step: index build (cold), then restore path
    _gen(llm, prompt, args.warmup)
    _gen(llm, prompt, args.warmup)
    rt.reset_pq_hsa_runtime_stats()
    p = _gen(llm, prompt, args.steps)
    st = rt.pq_hsa_runtime_stats()
    R["pq_f0_e2e_ms_per_tok"] = round(1000.0 * p["wall_s"] / max(1, p["n_out"]), 4)
    R["pq_f0_stats"] = {
        k: st.get(k)
        for k in (
            "installed", "backend", "sidecar", "cuda_graph_active", "batched_heads_active",
            "pq_decode_calls", "dense_fa_calls", "dense_fallback_calls",
            "cuda_attend_only_calls", "fused_decode_calls", "pq_restore_s_total",
            "pq_build_s_total", "last_error", "reject",
        )
    }
    os.environ["PQ_HSA_STEADY_EVENTS"] = "1"
    enable(); reset(); rt.reset_pq_hsa_runtime_stats()
    pe = _gen(llm, prompt, args.steps)
    R["pq_f0_events_ms"] = _events_ms_per_tok(summarize(), pe["n_out"])
    disable(); reset(); os.environ["PQ_HSA_STEADY_EVENTS"] = "0"
    print(f"[plugin_speed] ctx={ctx} pq(f0) e2e={R['pq_f0_e2e_ms_per_tok']} events={R['pq_f0_events_ms']} stats={R['pq_f0_stats']}", flush=True)

    # ---------------- PQ, flush 256 (production e2e) ----------------
    os.environ["PQ_HSA_FLUSH_INTERVAL"] = "256"
    _gen(llm, prompt, args.warmup)
    rt.reset_pq_hsa_runtime_stats()
    pf = _gen(llm, prompt, args.steps)
    stf = rt.pq_hsa_runtime_stats()
    R["pq_f256_e2e_ms_per_tok"] = round(1000.0 * pf["wall_s"] / max(1, pf["n_out"]), 4)
    R["pq_f256_stats"] = {k: stf.get(k) for k in ("pq_decode_calls", "dense_fallback_calls", "pq_flush_calls", "last_error")}
    os.environ["PQ_HSA_FLUSH_INTERVAL"] = "0"
    print(f"[plugin_speed] ctx={ctx} pq(f256) e2e={R['pq_f256_e2e_ms_per_tok']} stats={R['pq_f256_stats']}", flush=True)

    fa = float(R["dense_events_ms"].get("dense_fa") or 0.0)
    pq_attn = sum(float(R["pq_f0_events_ms"].get(k) or 0.0) for k in ("replay_gpu", "append", "copy_ctx"))
    R["attention_segment"] = {
        "dense_fa_ms_per_tok": round(fa, 4),
        "pq_attn_ms_per_tok": round(pq_attn, 4),
        "dense_fa_us_per_layer": round(1000.0 * fa / N_LAYERS, 2),
        "pq_attn_us_per_layer": round(1000.0 * pq_attn / N_LAYERS, 2),
        "speedup_dense_over_pq": round(fa / pq_attn, 4) if pq_attn else None,
        "measured": bool(pq_attn > 0 and fa > 0),
    }
    R["e2e_ratio_dense_over_pq_f0"] = round(R["dense_e2e_ms_per_tok"] / R["pq_f0_e2e_ms_per_tok"], 4) if R["pq_f0_e2e_ms_per_tok"] else None
    R["e2e_ratio_dense_over_pq_f256"] = round(R["dense_e2e_ms_per_tok"] / R["pq_f256_e2e_ms_per_tok"], 4) if R["pq_f256_e2e_ms_per_tok"] else None
    print(f"[plugin_speed] ctx={ctx} ATTN dense={fa:.3f} pq={pq_attn:.3f} speedup={R['attention_segment']['speedup_dense_over_pq']}x | e2e dense/pq f0={R['e2e_ratio_dense_over_pq_f0']} f256={R['e2e_ratio_dense_over_pq_f256']}", flush=True)

    # ---------------- passkey x N ----------------
    R["passkey_runs"] = []
    os.environ["PQ_HSA_FLUSH_INTERVAL"] = "256"
    for i in range(args.passkey_runs):
        try:
            ids = _passkey(tokenizer, ctx, i)
            _set_pq(False)
            dq = _gen(llm, ids, args.quality_tok)
            _set_pq(True)
            rt.reset_pq_hsa_runtime_stats()
            pq_q = _gen(llm, ids, args.quality_tok)
            qst = rt.pq_hsa_runtime_stats()
            m = min(len(dq["token_ids"]), len(pq_q["token_ids"]))
            agree = sum(int(a == b) for a, b in zip(dq["token_ids"][:m], pq_q["token_ids"][:m])) / max(1, m)
            rec = {
                "run": i,
                "dense_needle": NEEDLE in (dq["text"] or ""),
                "pq_needle": NEEDLE in (pq_q["text"] or ""),
                "top1_agreement": round(float(agree), 3),
                "pq_decode_calls": qst.get("pq_decode_calls"),
                "last_error": qst.get("last_error"),
                # keep the generated ids so paged vs non-paged runs can be diffed token-by-token
                "dense_ids": list(dq["token_ids"][:32]),
                "pq_ids": list(pq_q["token_ids"][:32]),
                "pq_restore_s_total": qst.get("pq_restore_s_total"),
                "pq_build_s_total": qst.get("pq_build_s_total"),
            }
        except Exception as exc:  # keep going; record
            rec = {"run": i, "error": f"{type(exc).__name__}: {exc}"}
        R["passkey_runs"].append(rec)
        print(f"[plugin_speed] ctx={ctx} passkey#{i} {rec}", flush=True)
    os.environ["PQ_HSA_FLUSH_INTERVAL"] = "0"
    R["passkey_summary"] = {
        "dense_hits": sum(int(bool(r.get("dense_needle"))) for r in R["passkey_runs"]),
        "pq_hits": sum(int(bool(r.get("pq_needle"))) for r in R["passkey_runs"]),
        "n": len(R["passkey_runs"]),
        "mean_top1_agreement": round(
            sum(float(r.get("top1_agreement") or 0.0) for r in R["passkey_runs"]) / max(1, len(R["passkey_runs"])), 3
        ),
    }
    return R


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctxs", default="32768,126720")
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--mem", type=float, default=0.60)
    ap.add_argument("--max-model-len", type=int, default=131072)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--mbt", type=int, default=0, help="max_num_batched_tokens (0 = max_model_len)")
    ap.add_argument("--quality-tok", type=int, default=16)
    ap.add_argument("--passkey-runs", type=int, default=8)
    ap.add_argument("--tag", default="plugin_speed")
    ap.add_argument("--model", default=None, help="HF id or local path (default: $MODEL_INST or Llama-3.1-8B-Instruct)")
    ap.add_argument("--out-dir", default=str(REPO / "results" / "speed"))
    args = ap.parse_args()

    global MODEL, N_LAYERS
    if args.model:
        MODEL = args.model
    OUT = Path(args.out_dir) / f"{args.tag}.json"
    OUT.parent.mkdir(parents=True, exist_ok=True)

    import torch
    import vllm
    from transformers import AutoTokenizer

    from pq_hsa_vllm.plugin import enabled as plugin_enabled

    assert plugin_enabled(), "PQ_HSA_VLLM must be 1 for this script"
    dev = torch.cuda.get_device_properties(0)
    P: dict = {
        "task": "decode speed via the vLLM plugin entry point",
        "model": MODEL,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": {"name": dev.name, "sm": dev.multi_processor_count},
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version.split()[0],
        "ctxs": [int(x) for x in args.ctxs.split(",") if x],
        "steps": args.steps,
        "warmup": args.warmup,
        "launch_env": {
            k: os.environ.get(k)
            for k in (
                "VLLM_PLUGINS", "PQ_HSA_VLLM", "VLLM_ENABLE_V1_MULTIPROCESSING", "CUDA_HOME",
                "PQ_HSA_CUDA_EXT_DIR", "PQ_USE_CUDA_GRAPH", "PQ_HSA_CUDA_GQA_MULTI",
                "PQ_HSA_PAGED_ATTEND", "PQ_HSA_PAGED_RESTORE", "PQ_HSA_CUDA_ATTEND_ONLY",
            )
        },
        "calibers": {
            "e2e_ms_per_tok": "host wall (cuda.synchronize around generate) / n_out; prefix-cache hit => decode-only window",
            "events": "CUDA Event on the compute stream (benchmarks.cuda_event_log)",
            "attention_segment": "dense = dense_fa ; PQ(flush 0) = replay_gpu + append + copy_ctx (reshape_and_cache_flash excluded)",
            "cross_version_rule": "numbers from different vLLM versions are separate stacks; never subtracted or ratioed across versions",
        },
    }
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    try:
        from transformers import AutoConfig

        _cfg = AutoConfig.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
        _tc = getattr(_cfg, "get_text_config", lambda: _cfg)()
        N_LAYERS = int(getattr(_tc, "num_hidden_layers", N_LAYERS))
    except Exception as exc:  # keep the default and record why
        P["n_layers_note"] = f"AutoConfig failed ({type(exc).__name__}); assumed {N_LAYERS}"
    P["n_layers"] = N_LAYERS
    t_build = time.perf_counter()
    llm = _build_llm(args.max_model_len, args.mem, args.mbt or None, args.tp)
    P["engine_build_s"] = round(time.perf_counter() - t_build, 1)

    import benchmarks.vllm_backend.pq_hsa_decode_runtime as _RT

    inst = _RT.pq_hsa_runtime_stats()
    P["plugin_install_stats"] = {k: inst.get(k) for k in ("installed", "backend", "sidecar")}
    P["plugin_installed_via_general_plugins"] = bool(inst.get("installed"))
    print(f"[plugin_speed] plugin install stats={P['plugin_install_stats']}", flush=True)
    if not inst.get("installed"):
        raise RuntimeError("pq_hsa_vllm general_plugins entry point did not install the backend")
    from pq_hsa_vllm.config import PQHSAConfig

    P["pqhsa_config"] = PQHSAConfig.from_env().asdict()
    P["results"] = []
    for ctx in P["ctxs"]:
        try:
            R = run_ctx(llm, tokenizer, ctx, args, _RT)
        except Exception as exc:
            R = {"ctx": ctx, "error": f"{type(exc).__name__}: {exc}"}
            import traceback

            traceback.print_exc()
        P["results"].append(R)
        OUT.write_text(json.dumps(P, indent=2, default=str), encoding="utf-8")

    go = []
    for R in P["results"]:
        if "error" in R:
            go.append({"ctx": R["ctx"], "go": False, "why": R["error"]})
            continue
        ps = R["passkey_summary"]
        ok_pass = ps["n"] > 0 and ps["pq_hits"] >= max(1, ps["n"] - 1) and ps["dense_hits"] >= max(1, ps["n"] - 1)
        ok_receipt = int(R["pq_f0_stats"].get("pq_decode_calls") or 0) > 0 and int(R["pq_f0_stats"].get("dense_fallback_calls") or 0) == 0
        go.append({"ctx": R["ctx"], "go": bool(ok_pass and ok_receipt), "passkey_ok": ok_pass, "receipt_ok": ok_receipt})
    P["go_no_go"] = {"per_ctx": go, "overall_go": all(g["go"] for g in go) if go else False}
    _destroy(llm)
    OUT.write_text(json.dumps(P, indent=2, default=str), encoding="utf-8")
    print(f"[plugin_speed] wrote {OUT}", flush=True)
    print(f"[plugin_speed] go_no_go={P['go_no_go']}", flush=True)


if __name__ == "__main__":
    main()
