#!/usr/bin/env bash
# PQ-HSA optimized environment for vLLM 0.8.5.post1, TP=1 speed runs.
# Source it; do not execute it.
#
#   source scripts/pqhsa_env.sh <GPU> <tag> [l8b|q14|q3_30b|gradient]
#   $PQHSA_PY benchmarks/speed/decode_speed.py --line l8b --tp 1 --ctx 130400 --mem 0.85 --tag <tag>
#
# Every PQ_HSA_* switch below defaults to 0 in code; forgetting one silently
# falls back to a slower (but numerically equivalent or near-equivalent) path.
# The vLLM plugin (PQ_HSA_VLLM=1) applies the same kernel configuration with
# setdefault semantics, see pq_hsa_vllm/src/pq_hsa_vllm/config.py.
#
# Multi-request runs: source this, then `export PQ_HSA_BATCH=1 PQ_HSA_PAGED_RESTORE=0`
# (the batch path together with PAGED_RESTORE is not supported).
#
# Optional overrides (set before sourcing):
#   PQHSA_PY              python of the environment that has vLLM installed (default: python)
#   PQ_HSA_CUDA_EXT_DIR   JIT extension cache (default: /tmp/torch_extensions_<tag>_gpu<GPU>);
#                         use a separate directory per vLLM/torch version
#   PQHSA_FLASH_KMEANS=1  opt in to the batched flash-kmeans index build (with FP32 norms and per-head re-seeding)
#                         (needs the flash_kmeans package; OFF by default)

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "pqhsa_env.sh must be sourced, not executed" >&2; exit 2
fi
if [ $# -lt 2 ]; then
  echo "usage: source scripts/pqhsa_env.sh <GPU> <tag> [l8b|q14|q3_30b|gradient]" >&2; return 2
fi

_PQHSA_GPU="$1"; _PQHSA_TAG="$2"; _PQHSA_LINE="${3:-}"
PQHSA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PQHSA_ROOT
export PQHSA_PY="${PQHSA_PY:-python}"
export PYTHONPATH="${PQHSA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Device, extension cache (one per tag and GPU), engine.
export CUDA_VISIBLE_DEVICES=$_PQHSA_GPU
export PQ_HSA_CUDA_EXT_DIR="${PQ_HSA_CUDA_EXT_DIR:-/tmp/torch_extensions_${_PQHSA_TAG}_gpu${_PQHSA_GPU}}"
mkdir -p "$PQ_HSA_CUDA_EXT_DIR"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN   # FlashInfer prefill is not used
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PQ_HSA_SIDECAR=fast PQ_USE_CUDA_GRAPH=1 PQ_CG_SENTINEL_INTERVAL=0

# Optimized CUDA graph-body configuration (19 switches; identical to
# pq_hsa_vllm.config.ADOPTED_KERNEL_ENV and benchmarks/speed/decode_speed.py:ADOPTED_ENV).
#   CUDA_ATTEND_ONLY            CUDA exact-attend epilogue after the Triton LUT scan
#   CUDA_SPLITS/PARTWARPS/PARTUNROLL/REDWARPS, CUDA_BG{SPLITS,WARPS,UNROLL,MM}, CUDA_ATTEND_SORT
#                               split-K partial / background / reduce kernel shapes
#   FP16_LUT                    fp16 lookup tables and list GEMM
#   FUSED_LUTPREP, FUSED_BLOCKRED, ATTEND_RAWIDX, MASKSKIP
#                               fused launches (LUT prep, block statistics, index casts, dead-slot skip)
#   LEAN_APPEND, LEAN_COPYCTX, LEAN_REPLAY, PARSTREAM
#                               host-side trimming (single-launch append, branchless context copy,
#                               cached replay checks, list-mass on a side stream)
export PQ_HSA_CUDA_ATTEND_ONLY=1 PQ_HSA_CUDA_ATTEND_SORT=0 \
       PQ_HSA_CUDA_SPLITS=8 PQ_HSA_CUDA_PARTWARPS=8 PQ_HSA_CUDA_PARTUNROLL=8 \
       PQ_HSA_FP16_LUT=1 PQ_HSA_CUDA_REDWARPS=4 \
       PQ_HSA_CUDA_BGSPLITS=8 PQ_HSA_CUDA_BGWARPS=8 PQ_HSA_CUDA_BGUNROLL=8 PQ_HSA_CUDA_BGMM=0 \
       PQ_HSA_LEAN_APPEND=1 PQ_HSA_LEAN_COPYCTX=1 PQ_HSA_PARSTREAM=1 \
       PQ_HSA_FUSED_LUTPREP=1 PQ_HSA_FUSED_BLOCKRED=1 PQ_HSA_ATTEND_RAWIDX=1 \
       PQ_HSA_MASKSKIP=1 PQ_HSA_LEAN_REPLAY=1

# GQA kernel instances for group sizes 4/5/8/16 (without CUDA_GQA_MULTI, G != 4 silently
# uses the aten fallback), paged-KV exact attend reading vLLM's KV pages in place, and
# ring-aware prefix-index restore (PAGED_RESTORE must be on together with PAGED_ATTEND
# for valid end-to-end numbers).
export PQ_HSA_CUDA_GQA_MULTI=1 PQ_HSA_CUDA_GQA_SET=4,5,8,16
export PQ_HSA_PAGED_ATTEND=1 PQ_HSA_PAGED_RESTORE=1 PQ_HSA_PAGED_KV=0

# Batched (all heads x all subspaces) flash-kmeans index build: experimental, OFF by
# default. The default torch k-means build is the reference configuration.
if [ "${PQHSA_FLASH_KMEANS:-0}" = "1" ]; then
  export PQ_HSA_BATCHED_BUILD=1 PQ_HSA_FLASH_KMEANS=1 PQ_HSA_BATCHED_RESEED=1
else
  export PQ_HSA_BATCHED_BUILD=0 PQ_HSA_FLASH_KMEANS=0
fi

# G=5 pair padding (Qwen2.5-14B): on for the q14 line only.
if [ "$_PQHSA_LINE" = "q14" ]; then
  export PQ_HSA_PAIR_PAD_GROUPS=1 PQ_HSA_PAIR_PADG_BUILD=1
else
  unset PQ_HSA_PAIR_PAD_GROUPS PQ_HSA_PAIR_PADG_BUILD
fi

# Switches that must stay off for speed measurements (slower alternatives,
# approximations outside the method definition, or diagnostic variants).
export PQ_HSA_FUSED_DECODE=0 PQ_HSA_CUDA_FUSED=0 PQ_HSA_SCAN_TOPK=0 PQ_HSA_FUSED_RADIX=0 \
       PQ_HSA_FUSED_FULL=0 PQ_HSA_BLOCK_TOPK=0 PQ_HSA_LEAN_WRAP=0 PQ_HSA_GRAPH_OUT_DIRECT=0 \
       PQ_HSA_GRAPH_APPEND=0 PQ_HSA_FLUSH_NO_RECAPTURE=0 PQ_HSA_NATIVE_BACKEND=0 \
       PQ_HSA_CUDA_BGFUSE=0 PQ_HSA_TOPK_RADIX=0 PQ_HSA_GROUP_SHARED_RETRIEVAL=0 PQ_HSA_GSR_LEAN=0 \
       PQ_HSA_REUSE_TOPK=0 PQ_HSA_OFFLOAD_FAST=0 PQ_HSA_PAIR_SCALAR_G=0 PQ_HSA_PAIR_VEC_ANYG=0 \
       PQ_HSA_HOST_SYNC_FIX=1 PQ_HSA_INCREMENTAL_FLUSH=1 PQ_HSA_BATCH=0 PQ_HSA_FALLBACK=0
# PQ_T12_EVENTS=1 disables per-layer CUDA graphs (per-stage timing diagnosis only).
unset PQ_T12_EVENTS PQ_HSA_E21_C2_MASS PQ_HSA_E21_C2_ANCHOR PQ_HSA_E21_BGMASS_LOG

# Switches whose arrival in every worker the speed script records in its json.
export PQ_HSA_EXTRA_ENV_CHECK=PQ_HSA_PAGED_ATTEND,PQ_HSA_PAGED_RESTORE,PQ_HSA_BATCHED_BUILD,PQ_HSA_FLASH_KMEANS,PQ_HSA_CUDA_GQA_MULTI${PQ_HSA_PAIR_PAD_GROUPS:+,PQ_HSA_PAIR_PAD_GROUPS}

# Sanity checks before any GPU work.
command -v "$PQHSA_PY" >/dev/null 2>&1 || [ -x "$PQHSA_PY" ] || { echo "missing interpreter $PQHSA_PY" >&2; return 3; }
if [ "$PQ_HSA_FLASH_KMEANS" = "1" ]; then
  CUDA_VISIBLE_DEVICES="" "$PQHSA_PY" -c "import importlib.util as u, sys; sys.exit(0 if u.find_spec('flash_kmeans') else 1)" \
    || { echo "PQHSA_FLASH_KMEANS=1 but flash_kmeans is not importable from $PQHSA_PY" >&2; return 4; }
fi

echo "[pqhsa_env] gpu=$_PQHSA_GPU tag=$_PQHSA_TAG line=${_PQHSA_LINE:-any} py=$PQHSA_PY"
env | grep -E '^PQ_HSA_(CUDA_ATTEND_ONLY|FP16_LUT|CUDA_GQA_MULTI|PAGED_ATTEND|PAGED_RESTORE|BATCHED_BUILD|FLASH_KMEANS|PAIR_PAD_GROUPS|BATCH)=' | sort | tr '\n' ' '; echo
