#!/usr/bin/env bash
# RULER background-ablation runs on one GPU (HF transformers stack), followed by
# the paired-bootstrap CI script.
#
# Usage:
#   scripts/run_ruler_ablation.sh <gpu_id> [variant ...]
#
# Variants (see benchmarks/run_ablation_arm.py): dense A B C C2 C2b T Bx D.
# Default: dense A B C C2. Each non-dense variant is run at p=1% and p=2%.
#
# Environment overrides:
#   MODEL     HF id or local path   (default: meta-llama/Llama-3.1-8B-Instruct)
#   CTX       prompt length          (default: 131072)
#   TASKS     RULER tasks            (default: niah_multiquery,niah_multivalue,cwe,fwe,qa)
#   NSAMPLES  prompts per task       (default: 50)
#   OUT       output directory       (default: results/quality)
#   PY        python interpreter     (default: python)
set -euo pipefail

GPU="${1:?usage: run_ruler_ablation.sh <gpu_id> [variant ...]}"
shift || true
VARIANTS=("$@")
if [[ ${#VARIANTS[@]} -eq 0 ]]; then
  VARIANTS=(dense A B C C2)
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
CTX="${CTX:-131072}"
TASKS="${TASKS:-niah_multiquery,niah_multivalue,cwe,fwe,qa}"
NSAMPLES="${NSAMPLES:-50}"
OUT="${OUT:-${ROOT}/results/quality}"
PY="${PY:-python}"
mkdir -p "${OUT}/logs"

export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

run_one() {
  local arm="$1" p="$2" tag="$3"
  local extra=()
  if [[ "${arm}" == "A" ]]; then
    extra=(--arm-a-nprobe 128)   # IVF-probed selector, 128 of 512 lists
  fi
  echo "=== [$(date '+%F %T')] START arm=${arm} p=${p} tag=${tag}"
  "${PY}" -u "${ROOT}/benchmarks/run_ablation_arm.py" --arm "${arm}" --p "${p}" --ctx "${CTX}" \
    --tasks "${TASKS}" --nsamples "${NSAMPLES}" --model "${MODEL}" \
    --jsonl-output "${OUT}/full_${tag}.jsonl" --json-output "${OUT}/full_${tag}.json" \
    ${extra[@]+"${extra[@]}"} >> "${OUT}/logs/full_${tag}.log" 2>&1
  echo "=== [$(date '+%F %T')] END   arm=${arm} tag=${tag}"
}

for arm in "${VARIANTS[@]}"; do
  if [[ "${arm}" == "dense" ]]; then
    run_one dense 0.01 dense
  else
    run_one "${arm}" 0.01 "${arm}_p1"
    run_one "${arm}" 0.02 "${arm}_p2"
  fi
done

"${PY}" "${ROOT}/benchmarks/stats/paired_bootstrap_ci.py" --input-dir "${OUT}" \
  --output "${OUT}/paired_bootstrap_ci.json"
