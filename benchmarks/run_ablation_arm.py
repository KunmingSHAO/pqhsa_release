#!/usr/bin/env python3
"""Run ONE (variant, budget) cell of the same-selector background ablation and
write its RULER jsonl/json via the ``eval_task_utility.py`` harness. Reuses the
production code paths only; every variant is expressed either as (a) stock CLI
flags accepted by eval_task_utility.py / e2e_pq_param_sweep.py, or (b) a
runtime monkeypatch applied from this script only, for the duration of this
process, on top of the unmodified library code.

Variants (same selector = PQ-HSA's own IVF-PQ retrieval unless noted; only the
treatment of UNSELECTED tokens differs):

  A  truncation with the IVF-probed selector: exact top-k only, unselected
     tokens dropped. Implemented as SparseAttentionConfig.mode="sparse"
     (equivalent to a -inf background: sparse mode normalizes the softmax only
     over sink+local+retrieved top-k). mode is fixed to "hybrid" in
     e2e_pq_param_sweep.py::_sweep_configs and not exposed as a CLI flag, so
     eval_task_utility._paper_configs is monkeypatched to override it. The
     sparse mode selects candidates through an nprobe-limited IVF probe
     (``--arm-a-nprobe`` lists, no candidate cap) rather than the all-PQ
     scan used by B.

  B  PQ-HSA default: exact top-k + PQ-score background with per-list value
     means (hybrid_topk_source=all_pq, hybrid_denominator_source=all_pq,
     hybrid_value_mode=centroid). No patch.

  C  exact top-k + a background with the SAME TOTAL MASS as B but a single
     global value mean instead of per-list means. Implemented by patching the
     one data source every forward branch reads,
     pq_hsa.attention.kv_cache.SparseKVCache._value_centroids_from_summaries():
     every list's value centroid becomes the token-count-weighted global mean
     of all retrieval-region values (accumulated in fp32). By linearity the
     background output becomes total_background_mass * global_mean_value,
     independent of how the mass is distributed across lists; selection and
     total background mass are unchanged.

  C2 / C2b  C plus a PQ-free background mass: after selection, the background
     logits are replaced by a per-row constant (C2: mean PQ score of the
     selected set; C2b: minimum exact logit of the selected set), so the
     background mass is count(unselected) * exp(anchor - row_max).
     Enabled through PQ_HSA_E21_C2_MASS / PQ_HSA_E21_C2_ANCHOR, read by the
     batched-heads path in e2e_pq_param_sweep.py.

  T  fixed-selector truncation (PQ_HSA_E21_TRUNC_ALLPQ=1): everything of B
     (same flags, same all-PQ top-k at the same budget, same sink/local, same
     exact rerank, same eager batched-heads path), then the output is replaced
     by the exact-only softmax over E = sink+local+top-k (background mass and
     value zero, fp32). Unlike A, the selector is unchanged, so T isolates
     "drop the background" alone. Implemented in
     e2e_pq_param_sweep._e21_trunc_epilogue; any other decode path (CUDA
     graph, candidate-pruned, per-head fallback) raises under T.

  Bx unselected-member list-mean background value
     (PQ_HSA_E21_BG_UNSELECTED_MEAN=1). B verbatim (selection, per-list
     background mass, shared normalisation) except the background VALUE of
     list j is (sum_{R_j} v - sum_{T cap R_j} v) / |B cap R_j| per query row
     (0 if no unselected member) instead of the all-member list mean.
     Implemented in e2e_pq_param_sweep._e21_bx_epilogue; eager batched-heads
     only; exclusive with T.

  D  near-pure PQ approximation: retrieval_top_fraction=1e-6 (one exact
     token); only sink+local stay exact.

  dense  full attention.

Usage:
  python benchmarks/run_ablation_arm.py --arm B --p 0.01 --ctx 131072 \
    --tasks niah_multiquery,niah_multivalue,cwe,fwe,qa --nsamples 50 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --jsonl-output results/quality/full_B_p1.jsonl --json-output results/quality/full_B_p1.json
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from pathlib import Path

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import eval_task_utility as etu  # noqa: E402


def _install_trunc_patch():
    """Arm A: force mode='sparse' (== -inf background == truncation)."""
    orig = etu._paper_configs

    def patched(args):
        index_cfg, attn_cfg = orig(args)
        attn_cfg = dataclasses.replace(attn_cfg, mode="sparse")
        return index_cfg, attn_cfg

    etu._paper_configs = patched
    return orig


def _restore_trunc_patch(orig):
    etu._paper_configs = orig


def _install_uniform_bg_patch():
    """Arm C: patch the single source of `retrieval_value_centroids` so every
    list's "centroid" is the same token-count-weighted global mean of all
    retrieval-region values. See the module docstring (arm C) for why this
    data-level patch is used instead of patching a specific forward branch."""
    import pq_hsa.attention.kv_cache as kvc

    orig = kvc.SparseKVCache._value_centroids_from_summaries

    def uniform_value_centroids(self):
        assert self._retrieval_value_sums is not None
        assert self._retrieval_value_counts is not None
        # Numerical note: a naive version summed
        # self._retrieval_value_sums (fp16, one row per IVF list, each row
        # already a sum over that list's ~len/num_lists tokens) across ALL
        # num_lists rows in one shot to get the global sum. Each per-list
        # row individually stays within fp16 range (that's how the
        # unpatched code always used it -- summed once per list), but
        # summing hundreds of those rows together routinely overflows fp16
        # (max ~65504) given known "massive activation" outlier dimensions
        # in transformer value vectors. The resulting +-inf/NaN centroid
        # propagated through background_output into every attention output,
        # producing degenerate generations. Fix: accumulate the global sum in
        # float32, then cast back to the working dtype only at the end.
        sums_f32 = self._retrieval_value_sums.to(torch.float32)
        counts_f32 = self._retrieval_value_counts.to(torch.float32)
        total_sum = sums_f32.sum(dim=0)
        total_count = counts_f32.sum()
        if float(total_count) > 0:
            global_mean = (total_sum / total_count).to(self._retrieval_value_sums.dtype)
        else:
            global_mean = torch.zeros(
                self._retrieval_value_sums.shape[1],
                device=self._retrieval_value_sums.device,
                dtype=self._retrieval_value_sums.dtype,
            )
        if not torch.isfinite(global_mean).all():
            raise RuntimeError(
                "variant C: uniform background centroid is non-finite even in "
                "float32 -- investigate further before trusting this variant's scores"
            )
        num_lists = self._retrieval_value_sums.shape[0]
        uniform_value_centroids.hits = getattr(uniform_value_centroids, "hits", 0) + 1
        if uniform_value_centroids.hits == 1:
            # First-hit evidence line in the run log: proves the switch fired.
            print(
                f"[ablation-C] uniform background centroids active (num_lists={num_lists})",
                flush=True,
            )
        return global_mean.unsqueeze(0).expand(num_lists, -1).clone()

    kvc.SparseKVCache._value_centroids_from_summaries = uniform_value_centroids
    return kvc, orig


def _restore_uniform_bg_patch(kvc, orig):
    kvc.SparseKVCache._value_centroids_from_summaries = orig


def _install_pqfree_mass_patch():
    """C2: same as C (uniform VALUE, via the patch above) PLUS a PQ-free
    background TOTAL MASS. C's mass is still computed from real per-token PQ
    logits (same denominator as B), so C only tests "uniform value" while
    leaving PQ in control of how much mass the background gets overall; C2
    additionally neutralizes that. (Reference path for the per-head
    implementation; the batched-heads path reads PQ_HSA_E21_C2_MASS.)

    Selection (topk) for hybrid_topk_source="all_pq" happens in
    _forward_many_hybrid_all_pq_shared_kv_fast via
    `torch.topk(approx_logits_all, k=topk)` on the REAL PQ scores, and is
    already finished (passed in as exact_local_tensor/exact_global_tensor)
    by the time _forward_many_hybrid_shared_kv_vectorized -- the function
    that actually builds the softmax denominator and background mass -- is
    called. This patch wraps exactly that function (all its args are
    keyword-only, so kwargs interception is exact and total) and replaces
    its `approx_logits_all_tensor` input (used only for the softmax
    denominator/background-mass math from that point on, NOT for
    selection, which already happened) with a per-row CONSTANT: the mean
    of `approx_exact_logits_tensor` (the PQ scores of the already-selected
    exact set).

    Net effect (worked out from the untouched downstream math): background
    total mass becomes count(unselected) * exp(mean_exact_pq_score -
    row_max) / denom -- i.e. proportional to how many tokens are
    unselected and to one aggregate scale statistic of the exact set, with
    NO per-token or per-list PQ score influencing the split. Combined with
    the uniform-value patch, background_output collapses to
    total_mass * global_mean_value, i.e.
    "count(unselected) x mean(exact exp-score), uniformly distributed".
    """
    import pq_hsa.attention.sparse_attention as sa

    orig = sa.IVFPQSparseAttention._forward_many_hybrid_shared_kv_vectorized

    def neutralized(self, **kwargs):
        alat = kwargs.get("approx_logits_all_tensor")
        aet = kwargs.get("approx_exact_logits_tensor")
        if alat is not None:
            if aet is not None and aet.numel() > 0 and aet.shape[1] > 0:
                anchor = aet.mean(dim=1, keepdim=True)
            else:
                anchor = alat.mean(dim=1, keepdim=True)
            kwargs["approx_logits_all_tensor"] = anchor.expand_as(alat).clone()
        return orig(self, **kwargs)

    sa.IVFPQSparseAttention._forward_many_hybrid_shared_kv_vectorized = neutralized
    return sa, orig


def _restore_pqfree_mass_patch(sa, orig):
    sa.IVFPQSparseAttention._forward_many_hybrid_shared_kv_vectorized = orig


COMMON_SPARSE_FLAGS = [
    "--sparse-warmup-steps", "4",
    "--cpu-cache-snapshot", "--prebuild-sparse-cache",
    "--num-lists", "512", "--subspaces", "8", "--bits", "4",
    "--coarse-iter", "1", "--pq-iter", "2",
    "--sink", "4", "--local-window", "128",
    "--hybrid-value-mode", "centroid",
    "--hybrid-topk-source", "all_pq", "--hybrid-denominator-source", "all_pq",
    "--share-gqa-kv-cache", "--index-update-strategy", "deferred",
    "--index-update-interval", "256",
    "--kernel-backend", "h20", "--topk-block-size", "1024", "--direction-normalize",
]


def _apply_iter_override(argv: list[str]) -> list[str]:
    """Opt-in: PQ_HSA_E21_COARSE_ITER / PQ_HSA_E21_PQ_ITER replace the values of
    --coarse-iter / --pq-iter (COMMON_SPARSE_FLAGS: 1 / 2). Unset -> argv unchanged."""
    out = list(argv)
    for flag, env in (("--coarse-iter", "PQ_HSA_E21_COARSE_ITER"), ("--pq-iter", "PQ_HSA_E21_PQ_ITER")):
        val = os.environ.get(env, "").strip()
        if val and flag in out:
            i = out.index(flag)
            print(f"[build-audit] {flag}: {out[i + 1]} -> {int(val)} (from {env})", flush=True)
            out[i + 1] = str(int(val))
    return out


_AUDIT = {"builds": 0, "build_s": 0.0, "bh_calls": 0, "bh_none": 0, "first_cfg": None}


def _install_build_audit():
    """Opt-in (PQ_HSA_E21_BUILD_LOG=1 or an iteration override): time every
    IVFPQIndex.build (cuda-synchronised wall; no numeric effect), log the config the
    index was actually built with, and count eager batched-heads calls (first-hit line
    shows the index format the production path read). Diagnosis only."""
    from pq_hsa.index.ivfpq import IVFPQIndex
    from pq_hsa.index import kmeans as _km
    e2e = etu.e2e
    orig_build = IVFPQIndex.build

    def timed_build(self, keys, **kw):
        if keys.is_cuda:
            torch.cuda.synchronize(keys.device)
        t0 = time.perf_counter()
        res = orig_build(self, keys, **kw)
        if keys.is_cuda:
            torch.cuda.synchronize(keys.device)
        _AUDIT["builds"] += 1
        _AUDIT["build_s"] += time.perf_counter() - t0
        if _AUDIT["first_cfg"] is None:
            c = self.config
            _AUDIT["first_cfg"] = dict(num_lists=c.num_lists, num_subspaces=c.num_subspaces, num_bits=c.num_bits,
                                     coarse_max_iter=c.coarse_max_iter, pq_max_iter=c.pq_max_iter,
                                     pack_codes=c.pack_codes, kernel_backend=c.kernel_backend,
                                     direction_normalize=c.direction_normalize,
                                     stores_packed_codes=bool(self.stores_packed_codes),
                                     flash_kmeans=bool(_km._flash_kmeans_enabled()), n_keys=int(keys.shape[0]))
            print(f"[build-audit] first index build: {_AUDIT['first_cfg']} ({time.perf_counter() - t0:.3f}s)", flush=True)
        return res

    IVFPQIndex.build = timed_build
    cls = e2e.GQAIVFPQDecodeAttentionAdapter
    orig_bh = cls._forward_many_batched_heads

    def counted(self, queries):
        out = orig_bh(self, queries)
        _AUDIT["bh_calls"] += 1
        if out is None:
            _AUDIT["bh_none"] += 1
        elif _AUDIT["bh_calls"] == 1:
            idx = self._streams[0].cache.index
            print(f"[build-audit] eager batched-heads path hit; index read by it: coarse_max_iter="
                  f"{idx.config.coarse_max_iter} pq_max_iter={idx.config.pq_max_iter} num_bits={idx.config.num_bits} "
                  f"stores_packed_codes={idx.stores_packed_codes} packed_snapshot={self._batched_heads is not None and self._batched_heads.get('packed') is not None}",
                  flush=True)
        return out

    cls._forward_many_batched_heads = counted
    orig_run = etu.run_one_sample

    def timed_run(model, tokenizer, args_, method, input_ids, gen_steps):
        t0 = time.perf_counter()
        b0 = _AUDIT["builds"]; s0 = _AUDIT["build_s"]
        out = orig_run(model, tokenizer, args_, method, input_ids, gen_steps)
        print(f"[build-audit] sample wall={time.perf_counter() - t0:.1f}s gen_steps={gen_steps} "
              f"n_input={int(input_ids.shape[1])} builds={_AUDIT['builds'] - b0} build_s={_AUDIT['build_s'] - s0:.2f}",
              flush=True)
        return out

    etu.run_one_sample = timed_run


def _install_probe_tagging():
    """--probe-out: tag every adapter with (sample ordinal, layer index) so the
    epilogue probe records are addressable. Diagnosis only; no numeric effect."""
    e2e = etu.e2e
    orig = e2e._new_sparse_adapter_for_kv
    state = {"sample": -1}

    def tagged(attn, key_states, **kw):
        adapter = orig(attn, key_states, **kw)
        layer = int(getattr(attn, "_e2e_pq_layer_idx", -1))
        if layer == 0:
            state["sample"] += 1
        adapter._e21_layer_idx = layer
        adapter._e21_sample = state["sample"]
        return adapter

    e2e._new_sparse_adapter_for_kv = tagged
    return orig


def build_argv(args) -> list[str]:
    argv = [
        "eval_task_utility.py",
        "--suite", "ruler",
        "--ruler-tasks", args.tasks,
        "--ruler-samples", str(args.nsamples),
        "--ruler-tokens", str(args.ctx),
        "--model", args.model,
        "--prefill-chunk-size", "8192",
        "--device", "cuda", "--device-map", "single", "--dtype", "float16",
        "--attn-implementation", "sdpa", "--local-files-only",
        "--jsonl-output", args.jsonl_output,
        "--json-output", args.json_output,
        "--seed", "0",
    ]
    if args.arm == "dense":
        argv += ["--method", "dense"]
        return argv

    argv += ["--method", "pqhsa"]
    argv += COMMON_SPARSE_FLAGS

    # D = "near-pure PQ approximation". SparseAttentionConfig validates
    # retrieval_top_fraction into (0, 1] (0.0 itself raises ValueError), and
    # ceil(p * retrieval_len) can never round down to 0 for any p>0 -- so a
    # literal 0-exact-token config is not reachable through this flag. Use
    # the smallest practically-zero fraction (1e-6): for any realistic
    # context length this resolves to topk=1 (ceil(1e-6 * len) == 1), i.e.
    # one exact token out of tens of thousands -- effectively 0% and
    # documented as such, not a true mathematical 0.
    p = 1e-6 if args.arm == "D" else args.p
    argv += ["--retrieval-top-fraction", str(p)]

    if args.arm == "A":
        # Selector-purity fix (see module docstring): probe (most/all) lists,
        # no candidate_budget cap, so the nprobe-limited IVF search
        # approximates a full-corpus scan over PQ-approx scores -- same
        # candidate universe as arm B/D's hybrid_topk_source="all_pq".
        # --arm-a-nprobe defaults to num_lists (512, exhaustive) but is
        # overridable: at 128K exhaustive probing of a per-query, per-KV-head
        # Python loop (mode="sparse" never takes the batched-heads fast path)
        # is very slow, so the 128K runs pass --arm-a-nprobe 128 as a
        # time-budget compromise -- still above the default nprobe=64 used
        # elsewhere.
        argv += ["--nprobe", str(args.arm_a_nprobe)]
        argv += ["--candidate-budget", str(max(args.ctx * 2, 65536))]
    else:
        argv += ["--nprobe", "64"]
        argv += ["--candidate-budget", "4096"]

    # All arms use the production default fast path. Arm C's patch acts on
    # the retrieval_value_centroids DATA (see module docstring), so it is
    # correct regardless of which forward branch reads that tensor -- no
    # need to force the slower --collect-attention-details path just for C.
    argv += ["--no-collect-attention-details"]
    return argv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=("A", "B", "C", "C2", "C2b", "D", "T", "Bx", "dense"))
    ap.add_argument("--p", type=float, default=0.01)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--tasks", default="niah_multiquery,qa")
    ap.add_argument("--nsamples", type=int, default=10)
    ap.add_argument("--arm-a-nprobe", type=int, default=512)
    ap.add_argument("--model", required=True)
    ap.add_argument("--jsonl-output", required=True)
    ap.add_argument("--json-output", required=True)
    ap.add_argument("--probe-out", default=None,
                    help="save first-step selection ids + in-call T-vs-hybrid "
                         "diffs (torch.save) to this path; diagnosis only")
    args = ap.parse_args()
    # Variants T / Bx: set BOTH the env (audit trail) and the module attribute the
    # eager batched-heads path actually reads (the module was imported above,
    # so an env set now would not reach its import-time constant). Any other
    # arm forces the attribute off, so an inherited env cannot turn B into T.
    _e2e = etu.e2e
    if args.arm == "Bx":
        os.environ["PQ_HSA_E21_BG_UNSELECTED_MEAN"] = "1"
        _e2e._E21_BG_UNSELECTED_MEAN = True
    else:
        if os.environ.get("PQ_HSA_E21_BG_UNSELECTED_MEAN", "0") == "1":
            print(f"[run_ablation_arm] WARNING: PQ_HSA_E21_BG_UNSELECTED_MEAN=1 inherited but arm={args.arm}; "
                  "forcing it OFF", flush=True)
        os.environ.pop("PQ_HSA_E21_BG_UNSELECTED_MEAN", None)
        _e2e._E21_BG_UNSELECTED_MEAN = False
    if args.arm == "T":
        os.environ["PQ_HSA_E21_TRUNC_ALLPQ"] = "1"
        _e2e._E21_TRUNC_ALLPQ = True
    else:
        if os.environ.get("PQ_HSA_E21_TRUNC_ALLPQ", "0") == "1":
            print(f"[run_ablation_arm] WARNING: PQ_HSA_E21_TRUNC_ALLPQ=1 inherited but arm={args.arm}; "
                  "forcing it OFF", flush=True)
        os.environ.pop("PQ_HSA_E21_TRUNC_ALLPQ", None)
        _e2e._E21_TRUNC_ALLPQ = False
    if args.probe_out:
        _e2e._E21_PROBE = True
        _install_probe_tagging()
    print(f"[run_ablation_arm] module switches: _E21_TRUNC_ALLPQ={_e2e._E21_TRUNC_ALLPQ} "
          f"_E21_BG_UNSELECTED_MEAN={_e2e._E21_BG_UNSELECTED_MEAN} "
          f"_E21_PROBE={_e2e._E21_PROBE}", flush=True)
    if args.arm in ("C2", "C2b"):
        # C2 on the production batched-heads path: the sparse_attention
        # monkeypatch below is never reached when
        # _batched_heads_supported() is True, so the harness reads this env
        # (see e2e_pq_param_sweep._forward_many_batched_heads).
        os.environ["PQ_HSA_E21_C2_MASS"] = "1"
    if args.arm == "C2b":
        # C2b = C2 with the tighter, genuinely PQ-free anchor (min exact rerank
        # logit of the selected set) instead of the mean PQ score of that set.
        os.environ["PQ_HSA_E21_C2_ANCHOR"] = "boundary"
    elif args.arm == "C2":
        # Explicit, so that an inherited PQ_HSA_E21_C2_ANCHOR from the shell
        # cannot silently turn a run labelled C2 into C2b. Pass
        # --arm C2b (or PQ_HSA_E21_C2_ANCHOR_ALLOW_ENV=1) to override.
        if os.environ.get("PQ_HSA_E21_C2_ANCHOR_ALLOW_ENV", "0") != "1":
            os.environ["PQ_HSA_E21_C2_ANCHOR"] = "mean"
    # Prove the opt-in switches actually reached the harness. This script calls
    # eval_task_utility.main() IN-PROCESS (no subprocess, no env rebuild), so
    # anything exported by the driver shell survives -- this line is the audit
    # trail that says so.
    print("[run_ablation_arm] PQ_HSA_E21_* env: "
          + repr({k: v for k, v in sorted(os.environ.items())
                  if k.startswith("PQ_HSA_E21_")}), flush=True)

    sys.argv = _apply_iter_override(build_argv(args)) if args.arm != "dense" else build_argv(args)
    _audit_on = (os.environ.get("PQ_HSA_E21_BUILD_LOG", "0") == "1"
               or bool(os.environ.get("PQ_HSA_E21_COARSE_ITER", "").strip())
               or bool(os.environ.get("PQ_HSA_E21_PQ_ITER", "").strip()))
    if _audit_on and args.arm != "dense":
        _install_build_audit()
    print(f"[run_ablation_arm] arm={args.arm} p={args.p} argv={sys.argv}", flush=True)

    trunc_orig = None
    uniform_sa = uniform_orig = None
    mass_sa = mass_orig = None
    if args.arm == "A":
        trunc_orig = _install_trunc_patch()
    if args.arm in ("C", "C2", "C2b"):
        uniform_sa, uniform_orig = _install_uniform_bg_patch()
    if args.arm in ("C2", "C2b"):
        mass_sa, mass_orig = _install_pqfree_mass_patch()
    try:
        etu.main()
        if _audit_on and args.arm != "dense":
            print(f"[build-audit] summary builds={_AUDIT['builds']} build_s_total={_AUDIT['build_s']:.2f} "
                  f"mean_s_per_build={_AUDIT['build_s'] / max(1, _AUDIT['builds']):.4f} "
                  f"eager_batched_calls={_AUDIT['bh_calls']} eager_batched_none={_AUDIT['bh_none']} "
                  f"first_cfg={_AUDIT['first_cfg']}", flush=True)
        if args.arm == "T" or args.probe_out:
            print(f"[ablation-T] summary arm={args.arm} eager_batched_T_hits="
                  f"{_e2e._E21_T_STATS['hits']} probe_records={len(_e2e._E21_T_STATS['probe'])}",
                  flush=True)
        if args.arm == "Bx":
            print(f"[ablation-Bx] summary arm=Bx eager_batched_Bx_hits="
                  f"{_e2e._E21_T_STATS.get('bx_hits', 0)}", flush=True)
        if args.probe_out:
            torch.save(_e2e._E21_T_STATS["probe"], args.probe_out)
            print(f"[ablation-T] probe records saved to {args.probe_out}", flush=True)
    finally:
        if trunc_orig is not None:
            _restore_trunc_patch(trunc_orig)
        if uniform_sa is not None:
            _restore_uniform_bg_patch(uniform_sa, uniform_orig)
        if mass_sa is not None:
            _restore_pqfree_mass_patch(mass_sa, mass_orig)


if __name__ == "__main__":
    main()
