"""VLLM >= 0.10 compatibility shims for the PQ-HSA sidecar (opt-in by
construction -- every entry point here is a no-op on 0.8.5.post1).

What changed between 0.8.5.post1 and 0.29.0 that the sidecar touches:

* ``FlashAttentionMetadataBuilder`` is built from
  ``(kv_cache_spec, layer_names, vllm_config, device)`` and no longer carries
  ``self.runner``. The sidecar needs the runner only to read the current
  request's ``prompt_token_ids`` for the prefix-persist digest
  (``pq_hsa_decode_runtime.prefix_cache_key``). We recover it by recording the
  ``GPUModelRunner`` instance on entry to its ``execute_model`` -- with
  ``max_num_seqs=1`` / TP=1 there is exactly one runner per process.
* ``FlashAttentionImpl.forward`` gained ``output_scale`` /
  ``output_block_scale`` keyword arguments (handled in
  ``pq_hsa_flashattn_wrap`` by forwarding ``**extra``).
* ``VLLM_ATTENTION_BACKEND`` env is gone; the backend is an engine arg
  (``attention_backend="FLASH_ATTN"``). Handled by the launcher scripts.

Nothing here edits vLLM sources.
"""

from __future__ import annotations

from typing import Any, Optional

_CURRENT_RUNNER: Optional[Any] = None
_INSTALLED = False


def current_runner() -> Optional[Any]:
    return _CURRENT_RUNNER


def vllm_version_tuple() -> tuple[int, ...]:
    try:
        import vllm

        parts = []
        for tok in str(vllm.__version__).split("."):
            digits = ""
            for ch in tok:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            parts.append(int(digits) if digits else 0)
        return tuple(parts)
    except Exception:
        return (0,)


def needs_runner_capture() -> bool:
    """True when the FA metadata builder cannot hand us the runner itself."""
    try:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder
    except Exception:
        return False
    import inspect

    try:
        params = inspect.signature(FlashAttentionMetadataBuilder.__init__).parameters
    except (TypeError, ValueError):
        return False
    return "runner" not in params


def install_runner_capture() -> bool:
    """Wrap ``GPUModelRunner.execute_model`` to remember the runner instance.

    Idempotent. Returns True when the wrap is active in this process.
    """
    global _INSTALLED
    if _INSTALLED:
        return True
    if not needs_runner_capture():
        return False
    classes = []
    for modname in ("vllm.v1.worker.gpu_model_runner", "vllm.v1.worker.gpu.model_runner"):
        # 0.29 ships two runners: the V1 `gpu_model_runner.GPUModelRunner`
        # (`requests` + `input_batch`, what prefix_cache_key reads) and the V2
        # `gpu/model_runner.GPUModelRunner` (`req_states: RequestState`, default
        # unless VLLM_USE_V2_MODEL_RUNNER=0). Capture both; the V2 request-state
        # accessor is not ported yet, so launchers pin the V1 runner.
        try:
            mod = __import__(modname, fromlist=["GPUModelRunner"])
            cls = getattr(mod, "GPUModelRunner", None)
            if cls is not None:
                classes.append((modname, cls))
        except Exception:
            continue
    if not classes:
        return False
    patched = []
    for modname, cls in classes:
        orig = cls.__dict__.get("execute_model")
        if orig is None or getattr(orig, "_pq_hsa_runner_capture", False):
            continue

        def _make(orig_fn):
            def execute_model(self, *args, **kwargs):
                global _CURRENT_RUNNER
                _CURRENT_RUNNER = self
                return orig_fn(self, *args, **kwargs)

            execute_model._pq_hsa_runner_capture = True  # type: ignore[attr-defined]
            return execute_model

        cls.execute_model = _make(orig)
        patched.append(modname)
    _INSTALLED = True
    print(f"[pq_hsa] vLLM>=0.10 compat: runner-capture installed on {patched}", flush=True)
    return True


__all__ = [
    "current_runner",
    "install_runner_capture",
    "needs_runner_capture",
    "vllm_version_tuple",
]
