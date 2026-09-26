"""Auto-load PQ-HSA worker hook in spawned vLLM TP processes.

This directory is prepended to PYTHONPATH by ``apply_worker_hook_env``.
Workers inherit that env. TP=1 in-process runs do not add this path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# hook_sitecustomize/sitecustomize.py -> repository root (override: PQ_HSA_REPO_ROOT).
_REPO = Path(os.environ.get("PQ_HSA_REPO_ROOT") or Path(__file__).resolve().parents[3])
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

if os.environ.get("PQ_HSA_WORKER_HOOK", "0") == "1":
    try:
        from benchmarks.vllm_backend.worker_hook import bootstrap

        bootstrap()
    except Exception as exc:  # noqa: BLE001 — never break interpreter startup
        print(f"[pq_hsa] sitecustomize bootstrap failed: {exc}", flush=True)
