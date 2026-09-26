#!/usr/bin/env bash
# One PQ-HSA quality run (HF transformers stack) with the default configuration.
#
# Usage:
#   scripts/eval_pqhsa.sh <gpu_id> <suite> <retrieval_fraction> <out_prefix> [extra eval_task_utility args]
#     suite: ruler | infinitebench | longbench
#     e.g. scripts/eval_pqhsa.sh 0 ruler 0.01 results/quality/ruler_pqhsa_p1 \
#            --ruler-tasks niah_multiquery,niah_multivalue,cwe,fwe,qa --ruler-samples 50 --ruler-tokens 131072
#
# Replace "--method pqhsa" by passing `--method dense` (full attention) or a baseline
# (quest, snapkv, retrieval_attention, pariskv, sink_local) as an extra argument;
# the last --method on the command line wins.
#
# Environment overrides: MODEL (default meta-llama/Llama-3.1-8B-Instruct), PY (default python).
set -euo pipefail

GPU="${1:?gpu}"; SUITE="${2:?suite}"; P="${3:?retrieval fraction}"; PREFIX="${4:?output prefix}"
shift 4
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PY="${PY:-python}"
mkdir -p "$(dirname "${PREFIX}")"

export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PQHSA_FLAGS=(
  # IVF-PQ index: 512 lists, 8 subspaces x 4 bits, direction/norm split, per-head k-means
  --num-lists 512 --subspaces 8 --bits 4 --coarse-iter 1 --pq-iter 2 --direction-normalize
  # exact sink + local window, retrieval fraction p, deferred index update every 256 steps
  --sink 4 --local-window 128 --retrieval-top-fraction "${P}"
  --index-update-strategy deferred --index-update-interval 256
  # all indexed keys scored by PQ; top-k attended exactly; the rest as per-list background
  --hybrid-topk-source all_pq --hybrid-denominator-source all_pq --hybrid-value-mode centroid
  --nprobe 64 --candidate-budget 4096
  --share-gqa-kv-cache --kernel-backend h20 --topk-block-size 1024
  --cpu-cache-snapshot --prebuild-sparse-cache --sparse-warmup-steps 4
  --no-collect-attention-details
)

"${PY}" -u "${ROOT}/benchmarks/eval_task_utility.py" \
  --method pqhsa --suite "${SUITE}" \
  --model "${MODEL}" --local-files-only \
  --device cuda --device-map single --dtype float16 --attn-implementation sdpa \
  --prefill-chunk-size 8192 --seed 0 \
  "${PQHSA_FLAGS[@]}" \
  --jsonl-output "${PREFIX}.jsonl" --json-output "${PREFIX}.json" \
  "$@"
