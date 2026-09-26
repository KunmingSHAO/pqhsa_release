"""Install PQ-HSA inside every vLLM TP worker.

Driver-side ``install_pq_hsa_flashattn_backend()`` does not reach spawned TP
workers. This module is loaded in each rank via:

* ``VLLM_PLUGINS`` / ``vllm.general_plugins`` entry
  (``hook_sitecustomize/*.dist-info`` on PYTHONPATH — no site-packages edit)
* sitecustomize in ``benchmarks/vllm_backend/hook_sitecustomize/``
* wrapping ``Worker.init_device`` / ``execute_model`` after those modules import
* optional ``worker_cls=benchmarks.vllm_backend.pq_hsa_worker.PQHSAWorker``

TP=1 runs without the hook env are unchanged: the wrap stays in-process.
The installable plugin (``pq_hsa_vllm``) supersedes the sitecustomize route.
"""

from __future__ import annotations

import atexit
import builtins
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

# benchmarks/vllm_backend/worker_hook.py -> repository root (override: PQ_HSA_REPO_ROOT).
REPO = Path(os.environ.get("PQ_HSA_REPO_ROOT") or Path(__file__).resolve().parents[2])
HOOK_SITE = REPO / "benchmarks" / "vllm_backend" / "hook_sitecustomize"
DEFAULT_STATS_DIR = REPO / "results" / "worker_stats"
STATS_NAME_ENV = "PQ_HSA_STATS_PREFIX"
DEFAULT_STATS_PREFIX = "pqhsa_rank"

_BOOTSTRAPPED = False
_IMPORT_WRAPPED = False
_ORIG_IMPORT = builtins.__import__
_ATEXIT_REGISTERED = False


def hook_enabled() -> bool:
    return os.environ.get("PQ_HSA_WORKER_HOOK", "0") == "1"


def apply_worker_hook_env(stats_dir: Optional[os.PathLike[str] | str] = None) -> None:
    """Set PYTHONPATH / plugin env so *child* TP workers load this hook.

    Call before ``LLM()``. Nothing changes unless the caller calls this.
    """
    extra = [str(HOOK_SITE), str(REPO)]
    old = os.environ.get("PYTHONPATH", "")
    parts = extra + ([old] if old else [])
    # Keep a single copy of each prefix, first occurrence wins.
    seen: set[str] = set()
    ordered: list[str] = []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            ordered.append(p)
    os.environ["PYTHONPATH"] = os.pathsep.join(ordered)
    os.environ["PQ_HSA_WORKER_HOOK"] = "1"
    if stats_dir is not None:
        os.environ["PQ_HSA_STATS_DIR"] = str(Path(stats_dir))
    else:
        os.environ.setdefault("PQ_HSA_STATS_DIR", str(DEFAULT_STATS_DIR))
    os.environ.setdefault("PQ_HSA_STATS_PREFIX", DEFAULT_STATS_PREFIX)
    # Only this plugin; no other vLLM general plugins are needed.
    os.environ.setdefault("VLLM_PLUGINS", "pq_hsa_worker")
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def stats_dir() -> Path:
    return Path(os.environ.get("PQ_HSA_STATS_DIR", str(DEFAULT_STATS_DIR)))


def stats_prefix() -> str:
    return os.environ.get(STATS_NAME_ENV, DEFAULT_STATS_PREFIX)


def rank_index(explicit: Optional[int] = None) -> int:
    if explicit is not None:
        return int(explicit)
    for key in ("PQ_HSA_RANK", "VLLM_RPC_RANK", "RANK", "LOCAL_RANK"):
        raw = os.environ.get(key)
        if raw is not None and raw != "":
            try:
                return int(raw)
            except ValueError:
                pass
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return 0


def rank_stats_path(rank: Optional[int] = None) -> Path:
    r = rank_index(rank)
    return stats_dir() / f"{stats_prefix()}{r}.json"


def dump_rank_stats(
    rank: Optional[int] = None,
    reason: str = "dump",
    extra: Optional[dict[str, Any]] = None,
) -> Optional[Path]:
    """Write this process's sidecar STATS. Driver STATS must not be trusted at TP>1."""
    is_worker = os.environ.get("PQ_HSA_IS_WORKER", "0") == "1"
    if rank is None and not is_worker:
        return None
    try:
        from benchmarks.vllm_backend.pq_hsa_decode_runtime import pq_hsa_runtime_stats

        r = rank_index(rank)
        os.environ["PQ_HSA_RANK"] = str(r)
        path = rank_stats_path(r)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "rank": r,
            "pid": os.getpid(),
            "reason": reason,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pq_hsa_enable": os.environ.get("PQ_HSA_ENABLE", "0"),
            "hook_enabled": hook_enabled(),
            "stats": pq_hsa_runtime_stats(),
        }
        if extra:
            payload.update(extra)
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        tmp.replace(path)
        return path
    except Exception:
        return None


def install_in_process(rank: Optional[int] = None) -> dict[str, Any]:
    """Call wrap or native installer in *this* process. Idempotent."""
    if rank is not None:
        os.environ["PQ_HSA_RANK"] = str(int(rank))
        os.environ["PQ_HSA_IS_WORKER"] = "1"
    from benchmarks.vllm_backend.pq_hsa_flashattn_wrap import (
        install_pq_hsa_flashattn_backend,
        pq_hsa_runtime_stats,
    )

    install_pq_hsa_flashattn_backend()
    st = pq_hsa_runtime_stats()
    r = rank_index(rank)
    print(
        f"[pq_hsa] worker hook install rank={r} pid={os.getpid()} "
        f"installed={st.get('installed')} backend={st.get('backend')} "
        f"sidecar={st.get('sidecar')}",
        flush=True,
    )
    if os.environ.get("PQ_HSA_IS_WORKER", "0") == "1":
        dump_rank_stats(r, reason="install")
    return {"rank": r, "pid": os.getpid(), **st}


def vllm_general_plugin() -> None:
    """``vllm.general_plugins`` entry. Runs in each worker during init_worker."""
    install_in_process()
    _patch_imported_workers()


def _patch_worker_class(cls) -> None:
    if getattr(cls.init_device, "_pq_hsa_hooked", False):
        return
    orig_init = cls.init_device
    orig_exec = getattr(cls, "execute_model", None)

    def init_device(self, *args, **kwargs):
        install_in_process(rank=getattr(self, "rank", None))
        return orig_init(self, *args, **kwargs)

    init_device._pq_hsa_hooked = True  # type: ignore[attr-defined]
    cls.init_device = init_device

    if orig_exec is not None and not getattr(orig_exec, "_pq_hsa_hooked", False):

        def execute_model(self, *args, **kwargs):
            try:
                return orig_exec(self, *args, **kwargs)
            finally:
                try:
                    dump_rank_stats(getattr(self, "rank", None), reason="execute_model")
                except Exception:
                    pass

        execute_model._pq_hsa_hooked = True  # type: ignore[attr-defined]
        cls.execute_model = execute_model


def _patch_plugins_module(mod) -> None:
    orig = getattr(mod, "load_general_plugins", None)
    if orig is None or getattr(orig, "_pq_hsa_hooked", False):
        return

    def load_general_plugins(*args, **kwargs):
        out = orig(*args, **kwargs)
        # Plugin may have been filtered out; still install.
        install_in_process()
        _patch_imported_workers()
        return out

    load_general_plugins._pq_hsa_hooked = True  # type: ignore[attr-defined]
    mod.load_general_plugins = load_general_plugins


def _patch_imported_workers() -> None:
    gw = sys.modules.get("vllm.v1.worker.gpu_worker")
    if gw is not None and hasattr(gw, "Worker"):
        _patch_worker_class(gw.Worker)
    vw = sys.modules.get("vllm.worker.worker")
    if vw is not None and hasattr(vw, "Worker"):
        _patch_worker_class(vw.Worker)
    plug = sys.modules.get("vllm.plugins")
    if plug is not None:
        _patch_plugins_module(plug)


def _import_and_patch(name, globals=None, locals=None, fromlist=(), level=0):
    mod = _ORIG_IMPORT(name, globals, locals, fromlist, level)
    if isinstance(name, str) and name.startswith("vllm"):
        _patch_imported_workers()
    return mod


def bootstrap() -> None:
    """Idempotent process-start hook. Safe to call from sitecustomize."""
    global _BOOTSTRAPPED, _IMPORT_WRAPPED, _ATEXIT_REGISTERED
    if not hook_enabled():
        return
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    if not _IMPORT_WRAPPED:
        builtins.__import__ = _import_and_patch
        _IMPORT_WRAPPED = True
    _patch_imported_workers()
    if not _ATEXIT_REGISTERED:
        def _atexit_dump() -> None:
            try:
                if os.environ.get("PQ_HSA_IS_WORKER", "0") == "1":
                    dump_rank_stats(reason="atexit")
            except Exception:
                pass

        atexit.register(_atexit_dump)
        _ATEXIT_REGISTERED = True
    _BOOTSTRAPPED = True


def read_rank_stats_files(n_ranks: int, directory: Optional[Path] = None) -> dict[int, dict]:
    d = Path(directory) if directory is not None else stats_dir()
    prefix = stats_prefix()
    out: dict[int, dict] = {}
    for i in range(n_ranks):
        p = d / f"{prefix}{i}.json"
        if p.is_file():
            try:
                out[i] = json.loads(p.read_text())
            except json.JSONDecodeError:
                out[i] = {"rank": i, "error": "invalid_json", "path": str(p)}
    return out


__all__ = [
    "HOOK_SITE",
    "REPO",
    "apply_worker_hook_env",
    "bootstrap",
    "dump_rank_stats",
    "hook_enabled",
    "install_in_process",
    "rank_index",
    "rank_stats_path",
    "read_rank_stats_files",
    "vllm_general_plugin",
]
