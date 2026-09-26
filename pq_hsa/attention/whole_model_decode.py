"""Whole-model CUDA-graph decode for Llama + PQ-HSA.

Captures embed + 32 decoder layers (RMSNorm / QKV / RoPE / PQ attention /
o_proj / MLP) + final norm + lm_head as a single CUDA graph, replacing the
HF Python per-layer loop, to which profiling attributed ~47 ms/tok of host
residual.

Attention uses the existing static-shape batched-heads path
(``_cg_forward_static``) so local-window growth is absorbed by padded
full-region buffers + an additive -inf mask. KV appends write through
tensor ``index_copy_`` slots that the host updates before each replay.

Numerical contract: same weights, same RoPE, same hybrid PQ math as the
eager patched HF path. Capture is self-validated against a fresh eager
``static_forward`` on the same static buffers.
"""
from __future__ import annotations

import json
import os
from typing import Any

import torch
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb as llama_apply_rotary_pos_emb,
)


def _env_on(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) == "1"


def collect_pq_adapters(model: torch.nn.Module) -> list[tuple[torch.nn.Module, Any]]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return []
    out: list[tuple[torch.nn.Module, Any]] = []
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        adapter = getattr(attn, "_e2e_pq_adapter", None)
        if adapter is None:
            continue
        out.append((attn, adapter))
    return out


class WholeModelDecodeGraph:
    """One CUDA graph over a full Llama decode step with PQ-HSA attention."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.pairs = collect_pq_adapters(model)
        if not self.pairs:
            raise RuntimeError("no PQ-HSA adapters on model")
        self.n_layers = len(self.pairs)
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        hidden = int(model.config.hidden_size)
        self.input_ids = torch.zeros(1, 1, device=self.device, dtype=torch.long)
        self.position_ids = torch.zeros(1, 1, device=self.device, dtype=torch.long)
        self.logits = torch.zeros(1, 1, int(model.config.vocab_size), device=self.device, dtype=self.dtype)
        self._hidden_dummy = torch.zeros(1, 1, hidden, device=self.device, dtype=self.dtype)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.capture_stream: torch.cuda.Stream | None = None
        self.enabled = False
        self.fail_reason: str | None = None
        self._steps = 0

    def eligible(self) -> bool:
        if getattr(self.model.config, "model_type", "") != "llama":
            return False
        if len(self.pairs) != len(self.model.model.layers):
            return False
        for _attn, adapter in self.pairs:
            prepare = getattr(adapter, "_wmg_prepare_buffers", None)
            if prepare is None or not prepare():
                return False
        return True

    def _static_forward(self) -> torch.Tensor:
        hidden = self.model.model.embed_tokens(self.input_ids)
        cos, sin = self.model.model.rotary_emb(hidden, self.position_ids)
        for layer, (_attn, adapter) in zip(self.model.model.layers, self.pairs, strict=True):
            attn = layer.self_attn
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            bsz, qlen, _ = hidden.shape
            head_dim = attn.head_dim
            q = attn.q_proj(hidden).view(bsz, qlen, -1, head_dim).transpose(1, 2)
            k = attn.k_proj(hidden).view(bsz, qlen, -1, head_dim).transpose(1, 2)
            v = attn.v_proj(hidden).view(bsz, qlen, -1, head_dim).transpose(1, 2)
            q, k = llama_apply_rotary_pos_emb(q, k, cos, sin)
            ctx = adapter._wmg_attend(q, k, v)
            hidden = residual + attn.o_proj(ctx.reshape(bsz, qlen, -1).contiguous())
            residual = hidden
            hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
        hidden = self.model.model.norm(hidden)
        return self.model.lm_head(hidden)

    def _host_set_slots(self) -> None:
        for _attn, adapter in self.pairs:
            adapter._wmg_set_write_slots()

    def _host_notify(self) -> None:
        for _attn, adapter in self.pairs:
            adapter._wmg_notify_after_replay()

    def capture(self, sample_input_ids: torch.Tensor, position: int) -> bool:
        """Warm up and capture. Does not increment cache length."""
        if not self.eligible():
            self.fail_reason = "not eligible (missing shared-base batched heads)"
            return False
        self.input_ids.copy_(sample_input_ids.to(device=self.device, dtype=torch.long))
        self.position_ids.fill_(int(position))
        self._host_set_slots()
        try:
            with torch.cuda.device(self.device):
                torch.cuda.synchronize(self.device)
                with torch.no_grad():
                    for _ in range(5):
                        self._host_set_slots()
                        _ = self._static_forward()
                torch.cuda.synchronize(self.device)
                graph = torch.cuda.CUDAGraph()
                capture_stream = torch.cuda.Stream(device=self.device)
                with torch.cuda.graph(
                    graph,
                    pool=None,
                    stream=capture_stream,
                    capture_error_mode="thread_local",
                ):
                    out = self._static_forward()
                    self.logits.copy_(out)
            with torch.cuda.device(self.device):
                self.logits.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize(self.device)
                self._host_set_slots()
                reference = self._static_forward()
            if not torch.isfinite(self.logits).all() or not torch.allclose(
                self.logits.float(), reference.float(), rtol=5e-2, atol=5e-2
            ):
                self.fail_reason = (
                    f"capture validation failed finite={bool(torch.isfinite(self.logits).all())}"
                )
                self.graph = None
                return False
            self.graph = graph
            self.capture_stream = capture_stream
            self.enabled = True
            self.fail_reason = None
            return True
        except Exception as exc:  # noqa: BLE001
            self.fail_reason = f"capture raised: {exc!r}"
            self.graph = None
            self.enabled = False
            if os.environ.get("PQ_CG_DEBUG"):
                import traceback

                traceback.print_exc()
            return False

    def step(self, input_ids: torch.Tensor, position: int) -> torch.Tensor:
        """Replay one decode token. Updates adapter CPU bookkeeping after replay."""
        if self.graph is None or not self.enabled:
            raise RuntimeError("whole-model graph is not captured")
        self.input_ids.copy_(input_ids.to(device=self.device, dtype=torch.long))
        self.position_ids.fill_(int(position))
        self._host_set_slots()
        with torch.cuda.device(self.device):
            self.graph.replay()
        self._host_notify()
        self._steps += 1
        return self.logits

    def invalidate(self) -> None:
        self.graph = None
        self.enabled = False
        for _attn, adapter in self.pairs:
            invalidate = getattr(adapter, "_wmg_invalidate", None)
            if invalidate is not None:
                invalidate()


def try_build_whole_model_graph(
    model: torch.nn.Module,
    sample_input_ids: torch.Tensor,
    position: int,
) -> WholeModelDecodeGraph | None:
    """Build and capture, or return None (caller keeps the HF loop)."""
    if not _env_on("PQ_WHOLE_MODEL_GRAPH", "1"):
        return None
    if not collect_pq_adapters(model):
        return None
    last_reason = None
    for attempt in range(3):
        try:
            runner = WholeModelDecodeGraph(model)
            if runner.capture(sample_input_ids, position):
                print(
                    json.dumps(
                        {
                            "event": "wmg_capture_ok",
                            "layers": runner.n_layers,
                            "attempt": attempt,
                        }
                    ),
                    flush=True,
                )
                return runner
            last_reason = runner.fail_reason
        except Exception as exc:  # noqa: BLE001
            last_reason = f"capture raised: {exc!r}"
            if os.environ.get("PQ_CG_DEBUG"):
                import traceback

                traceback.print_exc()
    print(json.dumps({"event": "wmg_capture_failed", "reason": last_reason}), flush=True)
    return None
