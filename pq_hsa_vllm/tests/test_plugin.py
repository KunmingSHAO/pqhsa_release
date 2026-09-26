"""CPU self-test for the vLLM plugin (no GPU / no LLM() call).

Covers:
  * the plugin is discoverable via importlib.metadata (vllm's own
    ``load_plugins_by_group`` uses exactly this API -- see
    site-packages/vllm/plugins/__init__.py).
  * with no switch set, vLLM behavior is unchanged: the entry point function
    is a no-op (no monkey-patch installed, no torch/vllm-heavy import even
    triggered).
  * with PQ_HSA_VLLM=1, the plugin installs the wrap backend -- this only
    monkey-patches a Python method (FlashAttentionImpl.forward) and does not
    need a live/free GPU, so it is also covered here on CPU.

Run inside the vLLM environment the package was installed into
(``pip install -e ./pq_hsa_vllm``), from the repository root:
  python -m pytest pq_hsa_vllm/tests -q
"""
from __future__ import annotations

import importlib.metadata
import os
import sys
from pathlib import Path

import pytest

REPO = str(Path(__file__).resolve().parents[2])
if REPO not in sys.path:
    sys.path.insert(0, REPO)


@pytest.fixture(autouse=True)
def _clean_env():
    saved = os.environ.get("PQ_HSA_VLLM")
    os.environ.pop("PQ_HSA_VLLM", None)
    yield
    if saved is None:
        os.environ.pop("PQ_HSA_VLLM", None)
    else:
        os.environ["PQ_HSA_VLLM"] = saved


def test_entry_point_discoverable():
    eps = importlib.metadata.entry_points(group="vllm.general_plugins")
    names = {ep.name: ep for ep in eps}
    assert "pq_hsa_vllm" in names, f"pq_hsa_vllm not in discovered plugins: {sorted(names)}"
    ep = names["pq_hsa_vllm"]
    assert ep.value == "pq_hsa_vllm.plugin:vllm_general_plugin"
    func = ep.load()
    from pq_hsa_vllm.plugin import vllm_general_plugin

    assert func is vllm_general_plugin


def test_import_alone_is_cheap():
    """Importing the package must not import torch/vllm/benchmarks."""
    for mod in ("torch", "vllm", "benchmarks.vllm_backend.pq_hsa_decode_runtime"):
        sys.modules.pop(mod, None)
    import pq_hsa_vllm  # noqa: F401
    import pq_hsa_vllm.plugin  # noqa: F401

    assert "vllm" not in sys.modules
    assert "benchmarks.vllm_backend.pq_hsa_decode_runtime" not in sys.modules


def test_noop_when_switch_unset():
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    from benchmarks.vllm_backend.pq_hsa_decode_runtime import STATS
    from pq_hsa_vllm.plugin import enabled, vllm_general_plugin

    assert not enabled()
    assert not getattr(FlashAttentionImpl.forward, "_pq_hsa_wrapped", False)
    assert STATS["installed"] is False

    vllm_general_plugin()  # must be a true no-op

    assert not getattr(FlashAttentionImpl.forward, "_pq_hsa_wrapped", False)
    assert STATS["installed"] is False


def test_enable_installs_wrap_backend():
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    from benchmarks.vllm_backend.pq_hsa_decode_runtime import STATS
    from pq_hsa_vllm.plugin import vllm_general_plugin

    os.environ["PQ_HSA_VLLM"] = "1"
    try:
        vllm_general_plugin()
    finally:
        pass  # env restored by the _clean_env fixture

    assert getattr(FlashAttentionImpl.forward, "_pq_hsa_wrapped", False) is True
    assert STATS["installed"] is True
    assert STATS["backend"] == "wrap"
    # Defaults applied: GQA multi / paged attend / paged restore.
    assert os.environ["PQ_HSA_CUDA_GQA_MULTI"] == "1"
    assert os.environ["PQ_HSA_PAGED_ATTEND"] == "1"
    assert os.environ["PQ_HSA_PAGED_RESTORE"] == "1"
    assert os.environ["PQ_HSA_ENABLE"] == "1"


def test_config_defaults_roundtrip():
    from pq_hsa_vllm.config import PQHSAConfig

    cfg = PQHSAConfig()
    env = cfg.as_env()
    assert env["PQ_HSA_TOP_FRAC"] == "0.01"
    assert env["PQ_HSA_FLUSH_INTERVAL"] == "256"
    assert env["PQ_HSA_CUDA_GQA_MULTI"] == "1"
    assert env["PQ_HSA_PAGED_ATTEND"] == "1"
    assert env["PQ_HSA_PAGED_RESTORE"] == "1"
    # The optimized kernel configuration is included by default.
    assert env["PQ_HSA_CUDA_ATTEND_ONLY"] == "1"
    assert env["PQ_HSA_LEAN_REPLAY"] == "1"


def test_config_from_env_respects_explicit_values(monkeypatch):
    from pq_hsa_vllm.config import PQHSAConfig

    monkeypatch.setenv("PQ_HSA_TOP_FRAC", "0.02")
    monkeypatch.setenv("PQ_HSA_CUDA_GQA_MULTI", "0")
    cfg = PQHSAConfig.from_env()
    assert cfg.top_fraction == 0.02
    assert cfg.gqa_multi is False

    monkeypatch.delenv("PQ_HSA_TOP_FRAC", raising=False)
    monkeypatch.delenv("PQ_HSA_CUDA_GQA_MULTI", raising=False)
    cfg2 = PQHSAConfig.from_env()
    written = cfg2.apply_env_defaults()
    assert os.environ["PQ_HSA_TOP_FRAC"] == "0.01"
    assert "PQ_HSA_TOP_FRAC" in written

    # setdefault never overwrites an explicit value.
    monkeypatch.setenv("PQ_HSA_TOP_FRAC", "0.05")
    written2 = PQHSAConfig.from_env().apply_env_defaults()
    assert os.environ["PQ_HSA_TOP_FRAC"] == "0.05"
    assert "PQ_HSA_TOP_FRAC" not in written2


def test_batch_mode_keeps_paged_attend_default(monkeypatch):
    """pq_hsa_batch_decode.py::run_pq_decode_one calls set_paged_kv_context()
    on both call sites (mirrors the single-request path), so batch=1 must NOT
    get a different paged_attend/paged_restore default."""
    from pq_hsa_vllm.config import PQHSAConfig

    monkeypatch.delenv("PQ_HSA_PAGED_ATTEND", raising=False)
    monkeypatch.delenv("PQ_HSA_PAGED_RESTORE", raising=False)

    monkeypatch.setenv("PQ_HSA_BATCH", "1")
    cfg = PQHSAConfig.from_env()
    assert cfg.batch is True
    assert cfg.paged_attend is True
    assert cfg.paged_restore is True

    monkeypatch.delenv("PQ_HSA_BATCH", raising=False)
    cfg2 = PQHSAConfig.from_env()
    assert cfg2.batch is False
    assert cfg2.paged_attend is True
    assert cfg2.paged_restore is True

    # explicit env always wins, batch or not.
    monkeypatch.setenv("PQ_HSA_BATCH", "1")
    monkeypatch.setenv("PQ_HSA_PAGED_ATTEND", "0")
    cfg3 = PQHSAConfig.from_env()
    assert cfg3.paged_attend is False


def test_batch_decode_module_has_paged_kv_fix():
    """Guard against the batch-mode paged-KV context refresh disappearing
    underneath this package (would silently make the assumption above wrong)."""
    import inspect

    from benchmarks.vllm_backend import pq_hsa_batch_decode

    src = inspect.getsource(pq_hsa_batch_decode)
    assert src.count("set_paged_kv_context") >= 2
