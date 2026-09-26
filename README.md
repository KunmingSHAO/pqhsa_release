# PQ-HSA

PQ-HSA is a decode-time hybrid sparse/approximate attention operator for long-context LLM inference.
After prefill, it builds an IVF-PQ index over the cached keys of every KV head with per-head k-means: keys are split into a unit direction and a norm, directions are assigned to one of 512 coarse lists, and the residual to the list centroid is product-quantized (8 subspaces, 4 bits each).
At each decode step every indexed key is scored with PQ lookup tables; the top-k keys (a fixed fraction of the context) are attended exactly together with a sink and a local window, and the unselected keys contribute their PQ logits, aggregated per IVF list and paired with each list's mean value, to the same softmax.
The operator ships as CUDA/Triton kernels for NVIDIA H20 (sm_90) and as a vLLM plugin that works on vLLM 0.8.5.post1 and 0.29.0 and reads vLLM's paged KV cache in place.

Results are reported in the paper; this repository contains the code needed to reproduce them.

## Repository layout

```
pq_hsa/                        core package
  index/                       IVF-PQ index: coarse k-means, product quantizer, 4-bit packing,
                               incremental index update, optional batched build
  attention/                   hybrid attention operator (exact top-k + per-list PQ background),
                               sparse KV cache, decode adapters
  kernels/                     Triton LUT-scan / top-k kernels (generic and H20-tuned)
    cuda/                      CUDA kernels (pq_fused_h20.cu) + JIT build script (compile.py)
pq_hsa_vllm/                   installable vLLM plugin (entry point vllm.general_plugins)
  src/pq_hsa_vllm/             plugin.py (entry point), config.py (PQHSAConfig), worker.py
  tests/                       CPU self-test of the plugin
benchmarks/
  vllm_backend/                decode runtime installed by the plugin (FlashAttention wrap,
                               GQA sidecar, paged-KV readers, vLLM 0.29 compatibility shims)
  e2e_pq_param_sweep.py        HF-transformers integration of the operator (batched-heads,
                               CUDA-graph decode path) used by the quality harness and by vLLM
  eval_task_utility.py         quality harness: RULER-style synthetic tasks, InfiniteBench, LongBench
  baselines_*.py               harness re-implementations of Quest, SnapKV, RetrievalAttention
                               and ParisKV used for comparisons
  run_ablation_arm.py          background-ablation variants (A, B, C, C2, C2b, T, Bx, D, dense)
  stats/paired_bootstrap_ci.py paired differences with task-stratified bootstrap 95% CIs
  speed/decode_speed.py        decode speed inside vLLM 0.8.5 (attention-segment and wall-clock clocks)
  speed/plugin_speed.py        decode speed through the plugin entry point (vLLM 0.8.5 and 0.29)
  speed/passkey.py             passkey prompt used by the speed scripts' quality gate
  cuda_event_log.py            CUDA-event timer for the attention-segment clock
scripts/
  pqhsa_env.sh                 optimized flag set for speed runs (source it)
  eval_pqhsa.sh                one quality run with the default configuration
  run_ruler_ablation.sh        background-ablation runs + paired bootstrap CIs
tests/                         CPU unit tests of the core package
```

## Installation

Tested with Python 3.12, CUDA 12.4 (nvcc 12.4), PyTorch 2.6.0 (cu124), Triton 3.2.0, on NVIDIA H20 GPUs.
The CUDA kernels are JIT-compiled with `torch.utils.cpp_extension` on first use (`TORCH_CUDA_ARCH_LIST=9.0`, `CUDA_HOME=/usr/local/cuda` by default); the build cache directory is set with `PQ_HSA_CUDA_EXT_DIR` (default `/tmp/torch_extensions`).

Core package and HF-transformers harness:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```

vLLM integration. Use a separate environment per vLLM version and install the plugin into it from the repository root:

```bash
# vLLM 0.8.5.post1 (torch 2.6.0+cu124)
pip install vllm==0.8.5.post1
pip install -e ./pq_hsa_vllm

# vLLM 0.29.0 (torch 2.13.0+cu130); the JIT kernels need a CUDA 13 toolchain, e.g. the
# nvidia-cuda-nvcc / crt / nvvm wheels, with CUDA_HOME=<site-packages>/nvidia/cu13
pip install vllm==0.29.0
pip install -e ./pq_hsa_vllm
```

The plugin is an editable install: at import time it puts the repository root on `sys.path` so that `pq_hsa` and `benchmarks.vllm_backend` are importable (override with `PQ_HSA_REPO_ROOT`).

Optional: the batched index build (`PQ_HSA_BATCHED_BUILD=1 PQ_HSA_FLASH_KMEANS=1 PQ_HSA_BATCHED_RESEED=1`) trains all heads of a layer in one flash-kmeans call and requires the `flash_kmeans` package. `PQ_HSA_BATCHED_RESEED=1` computes key norms in FP32 and re-seeds empty clusters with the per-head rule, so the batched build assigns keys to the same lists as the per-head build. It is off by default; the default build is the per-head torch k-means.

## Enabling the vLLM plugin

The plugin is registered under `vllm.general_plugins` and is a no-op unless `PQ_HSA_VLLM=1` is set:

```bash
PQ_HSA_VLLM=1 VLLM_PLUGINS=pq_hsa_vllm \
VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN \
python your_script.py          # any script that creates vllm.LLM(...)
```

Alternatively pass `worker_cls="pq_hsa_vllm.PQHSAWorker"` to `LLM(...)` (choosing the worker class is the opt-in).
With the plugin enabled, PQ-HSA replaces FlashAttention on single-token decode steps; prefill and multi-token steps stay on the dense FlashAttention path.
`PQ_HSA_ENABLE=0` keeps the backend registered but decodes densely (in-process A/B).

`pq_hsa_vllm.config.PQHSAConfig` documents the knobs and writes defaults with `setdefault` semantics (an explicitly set environment variable always wins):

| Knob | Environment variable | Default |
|---|---|---|
| retrieval fraction p | `PQ_HSA_TOP_FRAC` | `0.01` |
| sink tokens / local window | `PQ_HSA_SINK` / `PQ_HSA_LOCAL` | `4` / `128` |
| IVF lists / PQ subspaces / bits | `PQ_HSA_NUM_LISTS` / `PQ_HSA_SUBSPACES` / `PQ_HSA_BITS` | `512` / `8` / `4` |
| index update interval | `PQ_HSA_FLUSH_INTERVAL` (and `PQ_HSA_UPD_INTERVAL`) | `256` |
| GQA kernel instances (G in 4/5/8/16) | `PQ_HSA_CUDA_GQA_MULTI` | `1` |
| paged-KV exact attend / prefix restore | `PQ_HSA_PAGED_ATTEND` / `PQ_HSA_PAGED_RESTORE` | `1` / `1` |
| optimized kernel configuration (19 switches) | `PQ_HSA_ADOPT_KERNEL_STACK` | `1` |
| multi-request decode (one replay per request) | `PQ_HSA_BATCH` | `0` |
| dense fallback on error | `PQ_HSA_FALLBACK` | `0` |

`VLLM_ENABLE_V1_MULTIPROCESSING` and the attention backend must be set by the launcher, not by the plugin, because vLLM reads them before plugins are loaded.

### vLLM 0.29.0

The same plugin runs unchanged on vLLM 0.29.0. The 0.29 runs use:

| Setting | Kind | Purpose |
|---|---|---|
| `VLLM_USE_V2_MODEL_RUNNER=0` | environment variable | selects the earlier (V1) GPU model runner; 0.29 defaults to the V2 runner, whose request-state API the prefix-restore path does not read |
| `VLLM_USE_FLASHINFER_SAMPLER=0` | environment variable | uses the torch-native sampler instead of the JIT-compiled FlashInfer top-k/top-p sampler (greedy decoding does not need it) |
| `attention_backend="FLASH_ATTN"` | `LLM(...)` engine argument | replaces the `VLLM_ATTENTION_BACKEND` environment variable, which 0.29 removed (`benchmarks/speed/plugin_speed.py` sets it automatically for vLLM >= 0.10) |
| `CUDA_HOME=<site-packages>/nvidia/cu13` | environment variable | CUDA 13 toolchain for the JIT kernels (torch 2.13+cu130) |
| `PQ_HSA_CUDA_EXT_DIR=<fresh dir>` | environment variable | keeps the 0.29 kernel build separate from the 0.8.5 build |

`VLLM_ENABLE_V1_MULTIPROCESSING=0`, `VLLM_PLUGINS=pq_hsa_vllm` and `PQ_HSA_VLLM=1` are set as on 0.8.5.
On 0.29 vLLM writes the decode step's KV itself and stores K and V in one 4-D page `[num_blocks, kv_heads, block_size, 2*D]`; `benchmarks/vllm_backend/compat_v029.py` and `paged_kv_fa.py` handle both page layouts, and the paged-KV CUDA epilogue reads either layout in place.
Numbers measured on different vLLM versions are separate stacks and are not compared with each other.

## Quality evaluation

Models are loaded with `--local-files-only`; download them to the Hugging Face cache first (or pass local paths).
RULER-style tasks are generated by the harness itself (`niah_s`, `niah_mk`, `niah_multiquery`, `niah_multivalue`, `cwe`, `fwe`, `qa`, `vt`; `qa` is a harness-native fact task, see `RULER_TASK_META` in `eval_task_utility.py`).
For InfiniteBench place the official task files as `benchmarks/data/infinitebench/<task>.jsonl`; for LongBench place the official `data/<task>.jsonl` files under `benchmarks/data/longbench/data/` and `dataset2prompt.json` under `benchmarks/data/longbench/`.

One RULER run of PQ-HSA (Llama-3.1-8B-Instruct, 128K, p = 1%):

```bash
scripts/eval_pqhsa.sh 0 ruler 0.01 results/quality/ruler_pqhsa_p1 \
  --ruler-tasks niah_multiquery,niah_multivalue,cwe,fwe,qa --ruler-samples 50 --ruler-tokens 131072
```

The same wrapper runs `--suite infinitebench` (six tasks at 128K by default, `--infinitebench-tasks`, `--infinitebench-samples`) and `--suite longbench` (`--tasks`, `--max-input-tokens`).
Passing `--method dense` evaluates full attention; `--method quest|snapkv|retrieval_attention|pariskv|sink_local` evaluates the harness baselines, whose exact-token budget is set with `--token-budget` or derived from a retrieval fraction with `--quest-budget-p` / `--pariskv-budget-p` (see `--help`). Each run writes one jsonl row per prompt (`suite`, `task`, `_id`, `score`, generation) and a json summary.

Background ablation and paired confidence intervals (same selector, only the treatment of unselected tokens differs):

```bash
scripts/run_ruler_ablation.sh 0 dense A B C C2          # writes results/quality/full_<variant>_p{1,2}.jsonl
python benchmarks/stats/paired_bootstrap_ci.py --input-dir results/quality
```

`paired_bootstrap_ci.py` pairs prompts strictly on `(task, _id)`, macro-averages per task, and resamples prompts within each task (10,000 replicates, percentile 95% CI, fixed seeds). Further variants are added with `--method NAME=PATH` and compared with `--pair A,B`.

| Variant | Meaning | Switch |
|---|---|---|
| A | truncation with an IVF-probed selector (128 of 512 lists), no background | `--arm A` |
| B | PQ-HSA default | `--arm B` |
| C | same background mass as B, one global value mean instead of list means | `--arm C` |
| C2 / C2b | C with a PQ-free background logit (mean selected PQ score / lowest exact logit) | `PQ_HSA_E21_C2_MASS=1`, `PQ_HSA_E21_C2_ANCHOR=mean\|boundary` |
| T | fixed-selector truncation: B's selection, background removed | `PQ_HSA_E21_TRUNC_ALLPQ=1` |
| Bx | B with list means taken over unselected members only | `PQ_HSA_E21_BG_UNSELECTED_MEAN=1` |
| D | near-pure PQ approximation (one exact retrieved token) | `--arm D` |

The variant switches are opt-in diagnostics read by the eager batched-heads path; `run_ablation_arm.py` sets them per variant and they are off in every other run.

## Speed measurement

Two clocks are reported and never summed:

* **attention segment**: CUDA events on the compute stream (`benchmarks/cuda_event_log.py`); dense = the FlashAttention decode kernel, PQ-HSA = sidecar graph replay + index append + context copy (the KV-cache write is excluded because both arms pay it);
* **decode wall clock**: host wall time around `generate()` (with `cuda.synchronize`) divided by the number of generated tokens, on a prefix-cache hit so the timed window is decode-only (512 tokens = two index-update intervals in `decode_speed.py`).

Both arms run in one engine with PQ-HSA toggled in-process on the same prompt, batch size 1, fp16, `enforce_eager`, chunked prefill and prefix caching on.

vLLM 0.8.5, one GPU:

```bash
source scripts/pqhsa_env.sh 0 l8b_128k l8b          # GPU, tag, model line
$PQHSA_PY benchmarks/speed/decode_speed.py --line l8b --tp 1 --ctx 130400 --mem 0.85 --tag l8b_128k
```

Model lines: `l8b` (Llama-3.1-8B-Instruct), `gradient` (Llama-3-8B-Instruct-Gradient-1048k), `q14` (Qwen2.5-14B-Instruct-1M), `q3_30b` (Qwen3-30B-A3B-Instruct-2507); `--model` overrides the checkpoint path. The json in `results/speed/` records both clocks, the per-layer receipts that the CUDA path and CUDA graphs were active, index health, the worker-side flag values, and an 8-needle passkey check (dense vs PQ-HSA on the same token ids).

Through the plugin entry point (vLLM 0.8.5 or 0.29):

```bash
CUDA_VISIBLE_DEVICES=0 PQ_HSA_CUDA_EXT_DIR=/tmp/pqhsa_ext_v029 \
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
python benchmarks/speed/plugin_speed.py --model meta-llama/Llama-3.1-8B-Instruct \
  --ctxs 32768,126720 --steps 256 --warmup 8 --passkey-runs 8
```

(the two `VLLM_USE_*` variables are needed on 0.29 only; on 0.29 also set `CUDA_HOME` as above).

### Optimized flag set (`scripts/pqhsa_env.sh`)

Every `PQ_HSA_*` switch defaults to `0` in code; the script sets the configuration used for the speed runs:

| Group | Switches | Effect |
|---|---|---|
| CUDA attend epilogue | `CUDA_ATTEND_ONLY=1` | exact gather + hybrid softmax + list background in CUDA after the Triton LUT scan |
| kernel shapes | `CUDA_SPLITS=8 CUDA_PARTWARPS=8 CUDA_PARTUNROLL=8 CUDA_REDWARPS=4 CUDA_BGSPLITS=8 CUDA_BGWARPS=8 CUDA_BGUNROLL=8 CUDA_BGMM=0 CUDA_ATTEND_SORT=0` | split-K partial, background and reduce kernel geometry |
| LUT precision | `FP16_LUT=1` | fp16 lookup tables and list GEMM |
| fused launches | `FUSED_LUTPREP=1 FUSED_BLOCKRED=1 ATTEND_RAWIDX=1 MASKSKIP=1` | one-launch LUT preparation, block statistics, index casts, dead-slot skipping |
| host-side trimming | `LEAN_APPEND=1 LEAN_COPYCTX=1 LEAN_REPLAY=1 PARSTREAM=1` | single-launch append, branchless context copy, cached replay checks, list mass on a side stream |
| GQA instances | `CUDA_GQA_MULTI=1 CUDA_GQA_SET=4,5,8,16` | kernel instances for group sizes 4/5/8/16 (otherwise G != 4 uses the aten fallback) |
| paged KV | `PAGED_ATTEND=1 PAGED_RESTORE=1` | exact gather reads vLLM's KV pages in place; ring-aware prefix-index restore (use both together) |
| G=5 padding | `PAIR_PAD_GROUPS=1 PAIR_PADG_BUILD=1` | Qwen2.5-14B line only |
| batched index build | `BATCHED_BUILD=0 FLASH_KMEANS=0` | optional flash-kmeans build; **off by default** (`PQHSA_FLASH_KMEANS=1` before sourcing turns it on) |

The script also pins alternative or diagnostic paths off (for example `PQ_HSA_TOPK_RADIX`, `PQ_HSA_GROUP_SHARED_RETRIEVAL`, `PQ_HSA_REUSE_TOPK`, `PQ_HSA_OFFLOAD_FAST`) and unsets `PQ_T12_EVENTS` (per-stage event timing, which disables the per-layer CUDA graphs). For multi-request runs add `PQ_HSA_BATCH=1 PQ_HSA_PAGED_RESTORE=0` after sourcing.

## Default hyperparameters

| Parameter | Value |
|---|---|
| IVF lists (n_list) | 512 |
| PQ subspaces (M) | 8 |
| bits per code (b) | 4 |
| key representation | unit direction + norm, residual PQ |
| index build | per-head k-means after prefill (1 coarse iteration, 2 PQ iterations; `PQ_HSA_COARSE_ITER` / `PQ_HSA_PQ_ITER`) |
| sink tokens | 4 |
| local window | 128 |
| index update interval | 256 decode steps (deferred, incremental) |
| retrieval fraction p | 1% (also 2%) |
| scoring | every indexed key scored by PQ; top-k attended exactly; unselected keys as per-list background under one softmax |

## Tests

```bash
python -m pytest tests                                   # CPU; CUDA/Triton kernel tests are skipped without a GPU
python -m pytest pq_hsa_vllm/tests                       # in a vLLM environment with the plugin installed
```

## License

Apache License 2.0, see `LICENSE`.
