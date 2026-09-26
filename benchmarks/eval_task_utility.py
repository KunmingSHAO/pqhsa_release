#!/usr/bin/env python3
"""LongBench English + RULER 128K subset for dense / sink+local / PQ-HSA / Quest pages / SnapKV."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import string
import sys
from collections import Counter
from pathlib import Path
from types import MethodType
from typing import Any, Callable

import torch
from rouge import Rouge
from fuzzywuzzy import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_pq_param_sweep as e2e
import baselines_snapkv as snapkv
import baselines_quest_calibrated as quest_cal
import baselines_retrieval_attention as ra
import baselines_sts as sts
import baselines_pariskv as pariskv


def _install_uint16_cuda_gather_shim() -> None:
    """ivf_candidates gathers compact list_ids on CUDA.

    nlist=512 packs list_ids as uint16; PyTorch has no index_cuda for that
    dtype. Promote to int64 before gather — same values. Patch both the
    defining module and sparse_attention's ``from ... import gather_ids``
    alias. Harness-only; pq_hsa/ source is not edited.
    """
    from pq_hsa.attention import sparse_attention as _sa
    from pq_hsa.index import ivfpq as _ivf

    if getattr(_ivf.gather_ids, "_t11_5_uint16_shim", False):
        return

    def gather_ids(ids: torch.Tensor, indices: torch.Tensor | None = None) -> torch.Tensor:
        if indices is None:
            return ids
        if ids.dtype in (torch.uint16, torch.uint32):
            return _ivf.as_index(ids)[indices]
        return ids[indices]

    gather_ids._t11_5_uint16_shim = True  # type: ignore[attr-defined]
    _ivf.gather_ids = gather_ids
    _sa.gather_ids = gather_ids


_install_uint16_cuda_gather_shim()

ROOT = Path(__file__).resolve().parent
LONGBENCH_DIR = ROOT / "data" / "longbench"
EN_TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]
DATASET2MAXLEN = {
    "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64,
    "hotpotqa": 32, "2wikimqa": 32, "musique": 32,
    "gov_report": 512, "qmsum": 512, "multi_news": 512,
    "trec": 64, "triviaqa": 32, "samsum": 128,
    "passage_count": 32, "passage_retrieval_en": 32,
    "lcc": 64, "repobench-p": 64,
}


def _normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(remove_articles("".join(ch for ch in s.lower() if ch not in string.punctuation)).split())


def qa_f1_score(prediction: str, ground_truth: str, **kwargs: Any) -> float:
    pred = _normalize_answer(prediction).split()
    gold = _normalize_answer(ground_truth).split()
    common = Counter(pred) & Counter(gold)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / max(1, len(pred))
    recall = num_same / max(1, len(gold))
    return 2 * precision * recall / (precision + recall)


def rouge_score(prediction: str, ground_truth: str, **kwargs: Any) -> float:
    try:
        return float(Rouge().get_scores([prediction], [ground_truth], avg=True)["rouge-l"]["f"])
    except Exception:
        return 0.0


def classification_score(prediction: str, ground_truth: str, **kwargs: Any) -> float:
    all_classes = kwargs.get("all_classes") or []
    em_match = [c for c in all_classes if c in prediction]
    for match in list(em_match):
        if match in ground_truth and match != ground_truth:
            em_match.remove(match)
    if ground_truth in em_match:
        return 1.0 / len(em_match)
    return 0.0


def count_score(prediction: str, ground_truth: str, **kwargs: Any) -> float:
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(1.0 for n in numbers if str(n) == str(ground_truth)) / len(numbers)


def retrieval_score(prediction: str, ground_truth: str, **kwargs: Any) -> float:
    matches = re.findall(r"Paragraph (\d+)", str(ground_truth))
    if not matches:
        return 0.0
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(1.0 for n in numbers if str(n) == matches[0]) / len(numbers)


def code_sim_score(prediction: str, ground_truth: str, **kwargs: Any) -> float:
    pred_line = ""
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            pred_line = line
            break
    return fuzz.ratio(pred_line, ground_truth) / 100.0


METRIC: dict[str, Callable[..., float]] = {
    "narrativeqa": qa_f1_score, "qasper": qa_f1_score, "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score, "2wikimqa": qa_f1_score, "musique": qa_f1_score,
    "triviaqa": qa_f1_score,
    "gov_report": rouge_score, "qmsum": rouge_score, "multi_news": rouge_score, "samsum": rouge_score,
    "trec": classification_score,
    "passage_count": count_score,
    "passage_retrieval_en": retrieval_score,
    "lcc": code_sim_score, "repobench-p": code_sim_score,
}


def scorer(task: str, prediction: str, answers: list[str], all_classes: list[str] | None) -> float:
    fn = METRIC[task]
    return max(fn(prediction, ans, all_classes=all_classes) for ans in answers)


def load_longbench(task: str) -> list[dict[str, Any]]:
    path = LONGBENCH_DIR / "data" / f"{task}.jsonl"
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def middle_truncate(tokenizer: Any, prompt: str, max_tokens: int) -> torch.Tensor:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > max_tokens:
        half = max_tokens // 2
        ids = ids[:half] + ids[-max_tokens + half :]
    return torch.tensor([ids], dtype=torch.long)


def _thinking_chat_enabled(tokenizer: Any) -> bool:
    """Qwen3 instruct templates expose enable_thinking; Base/Llama do not."""
    tmpl = getattr(tokenizer, "chat_template", None) or ""
    return "enable_thinking" in tmpl


def _wrap_direct_answer(tokenizer: Any, user_text: str) -> str:
    """Native chat template with thinking off → same direct-answer style as Llama."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def encode_eval_prompt(tokenizer: Any, prompt: str, max_tokens: int) -> torch.Tensor:
    """Encode an eval prompt. Thinking models keep the native wrapper intact."""
    if not _thinking_chat_enabled(tokenizer):
        return middle_truncate(tokenizer, prompt, max_tokens)
    empty_wrapped = _wrap_direct_answer(tokenizer, "")
    overhead = len(tokenizer.encode(empty_wrapped, add_special_tokens=False))
    budget = max(1, max_tokens - overhead)
    raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(raw_ids) > budget:
        half = budget // 2
        raw_ids = raw_ids[:half] + raw_ids[-budget + half :]
    user_text = tokenizer.decode(raw_ids, skip_special_tokens=True)
    wrapped = _wrap_direct_answer(tokenizer, user_text)
    ids = tokenizer.encode(wrapped, add_special_tokens=False)
    if len(ids) > max_tokens:
        keep_tail = max(overhead, max_tokens // 4)
        ids = ids[: max_tokens - keep_tail] + ids[-keep_tail:]
    return torch.tensor([ids], dtype=torch.long)


def _strip_think_blocks(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.lstrip()


def _generate(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    cache: Any,
    gen_steps: int,
) -> str:
    text = e2e._greedy_generate_text(
        model,
        tokenizer,
        input_ids,
        cache,
        context_tokens=int(input_ids.shape[1]),
        gen_steps=gen_steps,
    )
    return _strip_think_blocks(text)


def _paper_configs(args: argparse.Namespace) -> tuple[Any, Any]:
    configs = e2e._sweep_configs(args)
    cfg = configs[0]
    return e2e.IVFPQConfig(**cfg["index"]), e2e.SparseAttentionConfig(**cfg["attention"])


# ---------------------------------------------------------------------------
# Quest pages (ICML 2024 algorithm in this harness; not official CUDA kernel)
# Matches third_party/Quest/evaluation/quest_attention.py + v2 paper notes:
# chunk/page size 16, first two layers dense, decode-only sparse, dense prefill.
# ---------------------------------------------------------------------------

QUEST_MIN_PAGES = 3  # official local_heavy_hitter_mask lower bound


def quest_aligned_token_budget(p: float, length: int, sink: int = 4, local: int = 128) -> int:
    """Aligned-budget formula: 16*ceil((sink+local+ceil(p*max(0,L-sink-local)))/16)."""
    inner = sink + local + math.ceil(p * max(0, length - sink - local))
    return 16 * math.ceil(inner / 16)


def quest_min_feasible_budget(chunk_size: int = 16) -> int:
    return int(chunk_size) * QUEST_MIN_PAGES


def _install_quest_pages_patch(
    model: torch.nn.Module,
    *,
    token_budget: int,
    chunk_size: int,
    use_denom_channel: bool = False,
    bg_mode: str = "page_mean_key",
    bg_alpha: float = 1.0,
    bg_delta: float = 0.0,
    bg_margin: float = 0.0,
) -> list[torch.nn.Module]:
    model_type = getattr(model.config, "model_type", "")
    if model_type not in {"llama", "qwen2", "mistral", "qwen3", "qwen3_moe"}:
        raise ValueError(f"unsupported model_type for quest pages patch: {model_type}")
    if chunk_size <= 0:
        raise ValueError(f"quest chunk_size must be positive, got {chunk_size}")
    if token_budget <= 0:
        raise ValueError(f"quest token_budget must be positive, got {token_budget}")

    # Qwen3 and Qwen3-MoE share the same attention block (QK-norm + qwen3 RoPE);
    # only the MLP differs, which this patch does not touch.
    forward = (
        _qwen3_quest_pages_forward
        if model_type in {"qwen3", "qwen3_moe"}
        else _llama_quest_pages_forward
    )
    patched = []
    for layer_idx, decoder_layer in enumerate(model.model.layers):
        attn = decoder_layer.self_attn
        if hasattr(attn, "_quest_original_forward"):
            raise RuntimeError("quest pages patch already installed")
        attn._quest_original_forward = attn.forward
        attn._quest_token_budget = int(token_budget)
        attn._quest_chunk_size = int(chunk_size)
        attn._quest_layer_idx = layer_idx
        attn._quest_use_denom_channel = bool(use_denom_channel)
        attn._quest_bg_mode = str(bg_mode)
        attn._quest_bg_alpha = float(bg_alpha)
        attn._quest_bg_delta = float(bg_delta)
        attn._quest_bg_margin = float(bg_margin)
        attn._quest_scan_buf = []
        attn.forward = MethodType(forward, attn)
        patched.append(attn)
    return patched


def _restore_quest_pages_patch(patched: list[torch.nn.Module]) -> None:
    for attn in patched:
        original = getattr(attn, "_quest_original_forward", None)
        if original is not None:
            attn.forward = original
        for name in (
            "_quest_original_forward",
            "_quest_token_budget",
            "_quest_chunk_size",
            "_quest_layer_idx",
            "_quest_use_denom_channel",
            "_quest_bg_mode",
            "_quest_bg_alpha",
            "_quest_bg_delta",
            "_quest_bg_margin",
            "_quest_scan_buf",
        ):
            if hasattr(attn, name):
                delattr(attn, name)


def _llama_quest_pages_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_values: Any = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    del attention_mask, kwargs
    input_shape = hidden_states.shape[:-1]
    layer_idx = int(getattr(self, "layer_idx", getattr(self, "_quest_layer_idx", 0)))
    # Prefill (q_len>1) and first two layers stay dense — official Quest eval recipe.
    if len(input_shape) != 2 or input_shape[1] != 1 or past_key_values is None or layer_idx < 2:
        return self._quest_original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=None,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )

    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = e2e.llama_apply_rotary_pos_emb(query_states, key_states, cos, sin)
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = past_key_values.update(
        key_states, value_states, self.layer_idx, cache_kwargs,
    )
    key_states = e2e.llama_repeat_kv(key_states, self.num_key_value_groups)
    value_states = e2e.llama_repeat_kv(value_states, self.num_key_value_groups)
    context = _quest_pages_attention(
        query_states,
        key_states,
        value_states,
        token_budget=int(self._quest_token_budget),
        chunk_size=int(self._quest_chunk_size),
        use_denom_channel=bool(getattr(self, "_quest_use_denom_channel", False)),
        bg_mode=str(getattr(self, "_quest_bg_mode", "page_mean_key")),
        bg_alpha=float(getattr(self, "_quest_bg_alpha", 1.0)),
        bg_delta=float(getattr(self, "_quest_bg_delta", 0.0)),
        bg_margin=float(getattr(self, "_quest_bg_margin", 0.0)),
        scan_buf=getattr(self, "_quest_scan_buf", None),
    )
    attn_output = context.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def _qwen3_quest_pages_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_values: Any = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Quest pages decode path for Qwen3: QK-norm + qwen3 RoPE, same page algorithm."""
    del attention_mask, kwargs
    input_shape = hidden_states.shape[:-1]
    layer_idx = int(getattr(self, "layer_idx", getattr(self, "_quest_layer_idx", 0)))
    if len(input_shape) != 2 or input_shape[1] != 1 or past_key_values is None or layer_idx < 2:
        return self._quest_original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=None,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )

    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = e2e.qwen3_apply_rotary_pos_emb(query_states, key_states, cos, sin)
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = past_key_values.update(
        key_states, value_states, self.layer_idx, cache_kwargs,
    )
    key_states = e2e.qwen3_repeat_kv(key_states, self.num_key_value_groups)
    value_states = e2e.qwen3_repeat_kv(value_states, self.num_key_value_groups)
    context = _quest_pages_attention(
        query_states,
        key_states,
        value_states,
        token_budget=int(self._quest_token_budget),
        chunk_size=int(self._quest_chunk_size),
        use_denom_channel=bool(getattr(self, "_quest_use_denom_channel", False)),
        bg_mode=str(getattr(self, "_quest_bg_mode", "page_mean_key")),
        bg_alpha=float(getattr(self, "_quest_bg_alpha", 1.0)),
        bg_delta=float(getattr(self, "_quest_bg_delta", 0.0)),
        bg_margin=float(getattr(self, "_quest_bg_margin", 0.0)),
        scan_buf=getattr(self, "_quest_scan_buf", None),
    )
    attn_output = context.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def _record_quest_scan_stat(
    scan_buf: list | None,
    dense_out: torch.Tensor,
    mixed_out: torch.Tensor,
    sel_mass_method: torch.Tensor,
    sel_mass_dense: torch.Tensor,
) -> None:
    """earlier-style rel_l2 / cosine of attn output vs dense, plus selected softmax mass."""
    if scan_buf is None:
        return
    num = (mixed_out.float() - dense_out.float()).norm().item()
    den = dense_out.float().norm().item()
    cos = torch.nn.functional.cosine_similarity(
        mixed_out.float().reshape(-1), dense_out.float().reshape(-1), dim=0
    ).item()
    scan_buf.append(
        {
            "rel_l2": float(num / max(den, 1e-12)),
            "cosine": float(cos),
            "sel_mass_method": float(sel_mass_method.mean().item()),
            "sel_mass_dense": float(sel_mass_dense.mean().item()),
        }
    )


def _collect_quest_scan_stats(patched: list) -> dict[str, float]:
    rel, cos, sm, sd = [], [], [], []
    n = 0
    for attn in patched:
        for row in getattr(attn, "_quest_scan_buf", None) or []:
            n += 1
            if row.get("rel_l2") is not None:
                rel.append(row["rel_l2"])
            if row.get("cosine") is not None:
                cos.append(row["cosine"])
            sm_v = row.get("sel_mass_method")
            if sm_v is not None and math.isfinite(float(sm_v)):
                sm.append(float(sm_v))
            sd_v = row.get("sel_mass_dense")
            if sd_v is not None and math.isfinite(float(sd_v)):
                sd.append(float(sd_v))
        if getattr(attn, "_quest_scan_buf", None) is not None:
            attn._quest_scan_buf = []
    if n == 0:
        return {}
    out: dict[str, float] = {"n_scan_records": n}
    if rel:
        out["attn_rel_l2_vs_dense"] = float(sum(rel) / len(rel))
    if cos:
        out["attn_cosine_vs_dense"] = float(sum(cos) / len(cos))
    if sm:
        out["sel_mass_method"] = float(sum(sm) / len(sm))
    if sd:
        out["sel_mass_dense"] = float(sum(sd) / len(sd))
    return out


def _quest_pages_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    token_budget: int,
    chunk_size: int,
    use_denom_channel: bool = False,
    bg_mode: str = "page_mean_key",
    bg_alpha: float = 1.0,
    bg_delta: float = 0.0,
    bg_margin: float = 0.0,
    scan_buf: list | None = None,
) -> torch.Tensor:
    """Quest page selector + optional host-native denom channel.

    Channel-off is the published truncated Quest path (bit-identical).
    bg_mode (channel-on only, no PQ):
      upper_bound     — page-max-key scores as logits (unsafe; overestimate)
      ub_margin       — page-max-key minus a constant margin (host-native, not oracle)
      page_mean_key   — q · page-mean K (conservative, SnapKV-like)
      ub_calibrated   — page-max-key minus (ub − true_max) gap on selected pages
      alpha_true      — bg = m + α(true−m) + δ (mean-centered shrink + level shift)
      trunc_true      — unselected tokens −∞ (truncation control; not α=0)
    """
    bsz, n_heads, q_len, head_dim = query_states.shape
    del q_len
    seq_length = key_states.shape[-2]
    scale = 1.0 / math.sqrt(head_dim)

    # Page-max key estimate (sign-aligned), official quest_attention.py.
    sign = torch.where(query_states > 0, query_states.new_ones(()), query_states.new_full((), -1.0))
    positive_query = query_states * sign
    max_key = key_states * sign
    pad = chunk_size - ((seq_length - 1) % chunk_size + 1)
    if pad:
        fill = torch.finfo(max_key.dtype).min
        max_key = torch.nn.functional.pad(max_key, (0, 0, 0, pad), value=fill)
    n_pages = max_key.shape[-2] // chunk_size
    page_max = max_key.view(bsz, n_heads, n_pages, chunk_size, head_dim).amax(dim=-2)
    page_scores = torch.matmul(positive_query.float(), page_max.transpose(2, 3).float())
    k_sel = min(max(QUEST_MIN_PAGES, token_budget // chunk_size), n_pages)
    _, topk = page_scores.topk(k=k_sel, dim=-1)
    token_idx = topk.unsqueeze(-1) * chunk_size + torch.arange(
        chunk_size, device=topk.device, dtype=topk.dtype
    )
    token_idx = token_idx.view(bsz, n_heads, -1)  # [B, H, T]
    valid = token_idx < seq_length
    token_idx = token_idx.clamp(max=seq_length - 1)
    gather_idx = token_idx.unsqueeze(-1).expand(bsz, n_heads, token_idx.shape[-1], head_dim)
    k_sel_t = torch.gather(key_states, 2, gather_idx)
    v_sel_t = torch.gather(value_states, 2, gather_idx)
    sel_logits = torch.matmul(query_states, k_sel_t.transpose(2, 3)) * scale
    sel_logits = sel_logits.masked_fill(~valid.unsqueeze(2), torch.finfo(sel_logits.dtype).min)
    if not use_denom_channel:
        attn_probs = torch.softmax(sel_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
        return torch.matmul(attn_probs, v_sel_t)

    topk_pages = topk.reshape(bsz, n_heads, k_sel)
    page_ub = page_scores.squeeze(2).float() * scale  # [B, H, n_pages]

    if pad:
        key_pad = torch.nn.functional.pad(key_states, (0, 0, 0, pad), value=0.0)
        val_pad = torch.nn.functional.pad(value_states, (0, 0, 0, pad), value=0.0)
    else:
        key_pad = key_states
        val_pad = value_states
    key_paged = key_pad.view(bsz, n_heads, n_pages, chunk_size, head_dim)
    val_paged = val_pad.view(bsz, n_heads, n_pages, chunk_size, head_dim)
    tok_ids = torch.arange(n_pages * chunk_size, device=query_states.device)
    page_tok_valid = (tok_ids < seq_length).view(1, 1, n_pages, chunk_size)
    page_count = page_tok_valid.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype=torch.float32)
    valid_f = page_tok_valid.unsqueeze(-1).to(dtype=torch.float32)
    page_mean_k = (key_paged.float() * valid_f).sum(dim=-2) / page_count
    page_mean_v = (val_paged.float() * valid_f).sum(dim=-2) / page_count
    page_values = page_mean_v.to(query_states.dtype)

    if bg_mode in {"alpha_true", "trunc_true"}:
        all_logits = torch.matmul(
            query_states.float(), key_states.float().transpose(2, 3)
        ) * scale
        sel_mask = torch.zeros(bsz, n_heads, seq_length, dtype=torch.bool, device=query_states.device)
        sel_pos = token_idx.clamp(max=seq_length - 1)
        sel_mask.scatter_(2, sel_pos, valid)
        bg_mask = ~sel_mask
        if bg_mode == "trunc_true":
            mixed = all_logits.masked_fill(bg_mask.unsqueeze(2), torch.finfo(all_logits.dtype).min)
        else:
            # Orthogonal axes: shrink toward background mean m, then shift by δ.
            # α=1, δ=0 → oracle (true logits). α=0, δ=0 → all bg tokens get m.
            # Truncation is NOT this α=0 — it is trunc_true (−∞).
            w = bg_mask.unsqueeze(2).to(dtype=all_logits.dtype)
            m = (all_logits * w).sum(dim=-1, keepdim=True) / w.sum(dim=-1, keepdim=True).clamp_min(1.0)
            bg = m + float(bg_alpha) * (all_logits - m) + float(bg_delta)
            mixed = torch.where(sel_mask.unsqueeze(2), all_logits, bg)
        dense_w = torch.softmax(all_logits, dim=-1, dtype=torch.float32)
        mixed_w = torch.softmax(mixed, dim=-1, dtype=torch.float32)
        v_f = value_states.float()
        dense_out = torch.matmul(dense_w, v_f)
        mixed_out = torch.matmul(mixed_w, v_f)
        sel_f = sel_mask.unsqueeze(2).to(dtype=dense_w.dtype)
        _record_quest_scan_stat(
            scan_buf,
            dense_out,
            mixed_out,
            (mixed_w * sel_f).sum(dim=-1),
            (dense_w * sel_f).sum(dim=-1),
        )
        return mixed_out.to(query_states.dtype)

    if bg_mode == "page_mean_key":
        q2 = query_states.float().squeeze(2)
        page_logits = torch.einsum("bhd,bhpd->bhp", q2, page_mean_k) * scale
    elif bg_mode == "ub_calibrated":
        sel_paged = sel_logits.float().view(bsz, n_heads, k_sel, chunk_size)
        valid_paged = valid.view(bsz, n_heads, k_sel, chunk_size)
        true_max_sel = sel_paged.masked_fill(~valid_paged, torch.finfo(sel_paged.dtype).min).max(dim=-1).values
        ub_sel = torch.gather(page_ub, 2, topk_pages)
        gap = (ub_sel - true_max_sel).mean(dim=-1, keepdim=True)
        page_logits = page_ub - gap
    elif bg_mode == "ub_margin":
        # Constant subtract from the page-max-key bound. No fp16 K read.
        # margin=0 ≡ upper_bound. Large margin is the safe side (underestimate).
        page_logits = page_ub - float(bg_margin)
    elif bg_mode == "upper_bound":
        page_logits = page_ub
    else:
        raise ValueError(f"unknown quest bg_mode: {bg_mode}")

    sel_max = sel_logits.float().max(dim=-1).values
    page_max_logit = page_logits.max(dim=-1).values
    row_max = torch.maximum(sel_max, page_max_logit.unsqueeze(-1))

    sel_exp = torch.exp(sel_logits.float() - row_max.unsqueeze(-1))
    sel_exp = sel_exp.masked_fill(~valid.unsqueeze(2), 0.0)
    # sel_exp is [B, H, q_len=1, T]. Squeeze q_len so denom stays [B, H, 1],
    # not [B, H, 1, 1] (which re-breaks broadcasting) and not [B, H]
    # (which at bsz=1 becomes [B, H, H]).
    sel_exp_sum = sel_exp.squeeze(2).sum(dim=-1, keepdim=True)

    page_exp = torch.exp(page_logits - row_max)
    selected_page_exp = torch.gather(page_exp, 2, topk_pages).sum(dim=2, keepdim=True)
    denom = (page_exp.sum(dim=-1, keepdim=True) - selected_page_exp + sel_exp_sum).clamp_min(
        torch.finfo(page_exp.dtype).tiny
    )

    exact_output = torch.matmul((sel_exp / denom.unsqueeze(-1)).to(query_states.dtype), v_sel_t)
    page_mask = torch.zeros(bsz, n_heads, n_pages, dtype=torch.bool, device=query_states.device)
    page_mask.scatter_(2, topk_pages, True)
    bg_weight = (page_exp.masked_fill(page_mask, 0.0) / denom).to(query_states.dtype)
    bg_output = torch.einsum("bhp,bhpd->bhd", bg_weight, page_values).unsqueeze(2)
    if scan_buf is not None:
        # Diagnostic only — does not change the attention output.
        # The live (sel_exp_sum/denom).mean() is NaN on real 8B when fp16 QK
        # overflows (exp(inf−inf)). Report a finite selected-share from the
        # same logits with a clamped float32 log-sum-exp.
        scan_buf.append(
            {
                "rel_l2": None,
                "cosine": None,
                "sel_mass_method": _finite_page_sel_mass(sel_logits, valid, page_logits, topk_pages),
                "sel_mass_dense": None,
            }
        )
    return (exact_output + bg_output).to(query_states.dtype)


def _finite_page_sel_mass(
    sel_logits: torch.Tensor,
    valid: torch.Tensor,
    page_logits: torch.Tensor,
    topk_pages: torch.Tensor,
) -> float:
    """Selected softmax share in [0, 1]. Recording-only; not used in the forward."""
    sel_f = torch.nan_to_num(sel_logits.detach().float(), nan=-1.0e4, posinf=80.0, neginf=-80.0)
    page_f = torch.nan_to_num(page_logits.detach().float(), nan=-1.0e4, posinf=80.0, neginf=-80.0)
    sel_f = sel_f.masked_fill(~valid.unsqueeze(2), -1.0e4)
    sel_max = sel_f.amax(dim=-1)
    page_max = page_f.amax(dim=-1, keepdim=True)
    row = torch.maximum(sel_max, page_max)
    sel_exp = torch.exp((sel_f - row.unsqueeze(-1)).clamp(-80.0, 0.0))
    sel_exp = sel_exp.masked_fill(~valid.unsqueeze(2), 0.0)
    sel_sum = sel_exp.squeeze(2).sum(dim=-1, keepdim=True)
    page_exp = torch.exp((page_f - row).clamp(-80.0, 0.0))
    page_mask = torch.zeros(
        page_f.shape[0], page_f.shape[1], page_f.shape[-1],
        dtype=torch.bool, device=page_f.device,
    )
    page_mask.scatter_(2, topk_pages, True)
    bg_sum = page_exp.masked_fill(page_mask, 0.0).sum(dim=-1, keepdim=True)
    mass = sel_sum / (sel_sum + bg_sum).clamp_min(1e-12)
    val = float(mass.mean().item())
    if not math.isfinite(val):
        return float("nan")
    return min(1.0, max(0.0, val))


def run_one_sample(
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    method: str,
    input_ids: torch.Tensor,
    gen_steps: int,
) -> str:
    input_ids = input_ids.to(e2e._model_input_device(model, fallback=input_ids.device))
    args._last_scan_stats = {}
    if method == "retrieval_attention":
        return _run_retrieval_attention_sample(model, tokenizer, args, input_ids, gen_steps)
    if method == "sts":
        draft_model = getattr(args, "_sts_draft_model", None)
        if draft_model is not None:
            return _run_sts_real_sample(model, draft_model, tokenizer, args, input_ids, gen_steps)
        return _run_sts_sample(model, tokenizer, args, input_ids, gen_steps)
    cache = e2e._run_prefill(model, input_ids, chunk_size=args.prefill_chunk_size)
    try:
        if method == "dense":
            return _generate(model, tokenizer, input_ids, cache, gen_steps)
        if method == "sink_local":
            patched = e2e._install_sink_local_patch(
                model,
                sink_tokens=int(str(args.sink).split(",")[0]),
                local_window=int(str(args.local_window).split(",")[0]),
            )
            try:
                return _generate(model, tokenizer, input_ids, cache, gen_steps)
            finally:
                e2e._restore_sink_local_patch(patched)
        if method == "pariskv":
            patched, final_topk = pariskv.install_pariskv_patch(
                model,
                token_budget=int(args.token_budget),
                sink=int(str(args.sink).split(",")[0]),
                local_window=int(str(args.local_window).split(",")[0]),
                n_subspaces=int(args.pariskv_subspaces),
                rho=float(args.pariskv_rho),
                beta=float(args.pariskv_beta),
                rotation_seed=int(args.pariskv_rotation_seed),
            )
            args._last_pariskv_final_topk = final_topk
            try:
                pred = _generate(model, tokenizer, input_ids, cache, gen_steps)
                args._last_pariskv_stats = pariskv.collect_pariskv_stats(patched)
                return pred
            finally:
                pariskv.restore_pariskv_patch(patched)
        if method in {"quest", "quest_channel"}:
            patched = _install_quest_pages_patch(
                model,
                token_budget=int(args.token_budget),
                chunk_size=int(args.quest_chunk_size),
                use_denom_channel=(method == "quest_channel"),
                bg_mode=str(getattr(args, "quest_bg_mode", "page_mean_key")),
                bg_alpha=float(getattr(args, "quest_bg_alpha", 1.0)),
                bg_delta=float(getattr(args, "quest_bg_delta", 0.0)),
                bg_margin=float(getattr(args, "quest_bg_margin", 0.0)),
            )
            try:
                pred = _generate(model, tokenizer, input_ids, cache, gen_steps)
                if method == "quest_channel":
                    args._last_scan_stats = _collect_quest_scan_stats(patched)
                return pred
            finally:
                _restore_quest_pages_patch(patched)
        if method == "quest_calibrated":
            patched = quest_cal.install_quest_calibrated_patch(
                model,
                token_budget=int(args.token_budget),
                chunk_size=int(args.quest_chunk_size),
            )
            try:
                return _generate(model, tokenizer, input_ids, cache, gen_steps)
            finally:
                quest_cal.restore_quest_calibrated_patch(patched)
        if method == "snapkv":
            patched = snapkv.install_snapkv_patch(
                model,
                token_budget=int(args.token_budget),
                sink_tokens=int(str(args.sink).split(",")[0]),
                local_window=int(str(args.local_window).split(",")[0]),
                obs_window=int(args.snapkv_obs_window),
                pool_kernel=int(args.snapkv_pool_kernel),
            )
            try:
                snapkv.run_observation_selection(model, cache, input_ids)
                snapkv.compress_selected_cache(cache, patched)
            finally:
                snapkv.restore_snapkv_patch(patched)
            return _strip_think_blocks(
                snapkv.generate_after_compression(
                    model,
                    tokenizer,
                    input_ids,
                    cache,
                    gen_steps,
                    prompt_len=int(input_ids.shape[1]),
                )
            )
        if method == "snapkv_channel":
            patched = snapkv.install_snapkv_patch(
                model,
                token_budget=int(args.token_budget),
                sink_tokens=int(str(args.sink).split(",")[0]),
                local_window=int(str(args.local_window).split(",")[0]),
                obs_window=int(args.snapkv_obs_window),
                pool_kernel=int(args.snapkv_pool_kernel),
                use_denom_channel=True,
            )
            try:
                snapkv.run_observation_selection(model, cache, input_ids)
                snapkv.build_and_store_clusters(cache, patched)
                snapkv.compress_selected_cache(cache, patched)
                return _strip_think_blocks(
                    snapkv.generate_after_compression(
                        model,
                        tokenizer,
                        input_ids,
                        cache,
                        gen_steps,
                        prompt_len=int(input_ids.shape[1]),
                    )
                )
            finally:
                snapkv.restore_snapkv_patch(patched)
        index_config, attention_config = _paper_configs(args)
        patched = e2e._install_pq_sparse_patch(
            model,
            index_config,
            attention_config,
            share_gqa_kv_cache=args.share_gqa_kv_cache,
            share_layer_pq_codebook=args.share_layer_pq_codebook,
        )
        try:
            if args.prebuild_sparse_cache:
                e2e._prebuild_sparse_adapters(
                    patched, cache, share_gqa_kv_cache=args.share_gqa_kv_cache
                )
            return _generate(model, tokenizer, input_ids, cache, gen_steps)
        finally:
            e2e._restore_pq_sparse_patch(patched)
    finally:
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_retrieval_attention_sample(
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    input_ids: torch.Tensor,
    gen_steps: int,
) -> str:
    """Prefill under the RA patch (captures queries), then decode with the index."""
    patched = ra.install_retrieval_attention_patch(
        model,
        token_budget=int(args.token_budget),
        prompt_len=int(input_ids.shape[1]),
        sink=int(args.ra_sink),
        local_window=int(args.ra_local),
        n_queries=int(args.ra_n_queries),
        knn=int(args.ra_knn),
        nlist=int(args.ra_nlist),
        nprobe=int(args.ra_nprobe),
        nprobe_q=int(args.ra_nprobe_q),
    )
    cache = None
    try:
        cache = e2e._run_prefill(model, input_ids, chunk_size=args.prefill_chunk_size)
        ra.build_indexes(patched, cache)
        return _generate(model, tokenizer, input_ids, cache, gen_steps)
    finally:
        ra.restore_retrieval_attention_patch(patched)
        if cache is not None:
            del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_sts_sample(
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    input_ids: torch.Tensor,
    gen_steps: int,
) -> str:
    """Prefill under the STS patch (captures calibration queries at the proxy
    draft layer and every target layer), build the offline head map, then
    decode (draft layer dense each step; target layers exact top-k, no bg)."""
    patched = sts.install_sts_patch(
        model,
        token_budget=int(args.token_budget),
        prompt_len=int(input_ids.shape[1]),
        draft_layer=int(args.sts_draft_layer),
        calib_tokens=int(args.sts_calib_tokens),
        calib_topk=int(args.sts_calib_topk),
    )
    cache = None
    try:
        cache = e2e._run_prefill(model, input_ids, chunk_size=args.prefill_chunk_size)
        head_map_stats = sts.build_head_maps(patched, cache)
        args._last_sts_head_map_stats = head_map_stats
        return _generate(model, tokenizer, input_ids, cache, gen_steps)
    finally:
        sts.restore_sts_patch(patched)
        if cache is not None:
            del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_sts_real_sample(
    model: Any,
    draft_model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    input_ids: torch.Tensor,
    gen_steps: int,
) -> str:
    """Real draft: separate Llama-3.2-1B-Instruct model + its own KV
    cache alongside the target. Prefill both on the identical input_ids,
    build the offline (target-layer -> mapped-draft-layer) head map, then
    decode with the draft's full forward completing before the target's each
    step (see baselines_sts.generate_with_real_draft)."""
    handle = sts.install_sts_real(
        model,
        draft_model,
        token_budget=int(args.token_budget),
        prompt_len=int(input_ids.shape[1]),
        calib_tokens=int(args.sts_calib_tokens),
        calib_topk=int(args.sts_calib_topk),
    )
    cache = None
    draft_cache = None
    try:
        cache = e2e._run_prefill(model, input_ids, chunk_size=args.prefill_chunk_size)
        draft_input_ids = input_ids.to(e2e._model_input_device(draft_model, fallback=input_ids.device))
        draft_cache = e2e._run_prefill(draft_model, draft_input_ids, chunk_size=args.prefill_chunk_size)
        head_map_stats = sts.build_head_maps_real(handle, cache, draft_cache)
        args._last_sts_head_map_stats = head_map_stats
        text, gen_stats = sts.generate_with_real_draft(
            model,
            draft_model,
            tokenizer,
            input_ids,
            cache,
            draft_cache,
            context_tokens=int(input_ids.shape[1]),
            gen_steps=gen_steps,
            shared=handle["shared"],
        )
        args._last_sts_gen_stats = gen_stats
        return _strip_think_blocks(text)
    finally:
        sts.restore_sts_real(handle)
        if cache is not None:
            del cache
        if draft_cache is not None:
            del draft_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def already_done(jsonl_path: Path) -> set[tuple[str, str, str]]:
    done: set[tuple[str, str, str]] = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            done.add((row["suite"], row["task"], row["_id"]))
    return done


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_longbench(model: Any, tokenizer: Any, args: argparse.Namespace) -> dict[str, Any]:
    prompts = json.loads((LONGBENCH_DIR / "dataset2prompt.json").read_text(encoding="utf-8"))
    out_path = Path(args.jsonl_output)
    done = already_done(out_path)
    tasks = [t for t in EN_TASKS if t in args.tasks.split(",")]
    task_scores: dict[str, list[float]] = {t: [] for t in tasks}
    # Reload already-scored rows so a resume still has a complete average.
    if out_path.exists():
        with out_path.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row["suite"] == "longbench" and row["task"] in task_scores:
                    task_scores[row["task"]].append(float(row["score"]))
    for task in tasks:
        rows = load_longbench(task)
        if args.max_samples is not None:
            rows = rows[: args.max_samples]
        prompt_fmt = prompts[task]
        max_gen = DATASET2MAXLEN[task]
        max_input = args.max_input_tokens - max_gen
        for row in rows:
            sid = str(row["_id"])
            if ("longbench", task, sid) in done:
                continue
            prompt = prompt_fmt.format(**row)
            input_ids = encode_eval_prompt(tokenizer, prompt, max_input)
            pred = run_one_sample(model, tokenizer, args, args.method, input_ids, max_gen)
            score = scorer(task, pred, list(row["answers"]), row.get("all_classes"))
            rec = {
                "suite": "longbench",
                "task": task,
                "_id": sid,
                "method": args.method,
                "pred": pred,
                "score": score,
                "n_input_tokens": int(input_ids.shape[1]),
            }
            append_jsonl(out_path, rec)
            task_scores[task].append(score)
            print(json.dumps({"event": "longbench_sample", **{k: rec[k] for k in rec if k != "pred"}}, sort_keys=True), flush=True)
        mean = sum(task_scores[task]) / max(1, len(task_scores[task]))
        print(json.dumps({"event": "longbench_task", "task": task, "n": len(task_scores[task]), "score": mean}, sort_keys=True), flush=True)
    means = {t: (sum(v) / len(v) if v else None) for t, v in task_scores.items()}
    present = [v for v in means.values() if v is not None]
    return {"tasks": means, "average": (sum(present) / len(present) if present else None)}


HAYSTACK = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "

# Incremental RULER extensions. Existing niah_s/mk/qa/vt generators are unchanged.
# Official refs: NVIDIA/RULER scripts/synthetic.yaml + scripts/data/synthetic/{niah,common_words_extraction,freq_words_extraction}.py
RULER_TASK_ORDER = [
    "niah_s",
    "niah_mk",
    "niah_multiquery",
    "niah_multivalue",
    "cwe",
    "fwe",
    "qa",
    "vt",
]
RULER_GEN_STEPS = {
    "niah_s": 16,
    "niah_mk": 16,
    "qa": 16,
    "vt": 16,
    "niah_multiquery": 64,
    "niah_multivalue": 64,
    "cwe": 128,
    "fwe": 64,
}
RULER_TASK_META: dict[str, dict[str, Any]] = {
    "niah_s": {
        "kind": "niah",
        "num_needle_k": 1,
        "num_needle_v": 1,
        "num_needle_q": 1,
        "type_haystack": "noise",
        "type_needle_k": "KEY4",
        "type_needle_v": "numbers7",
        "native": True,
    },
    "niah_mk": {
        "kind": "niah",
        "num_needle_k": 4,
        "num_needle_v": 1,
        "num_needle_q": 1,
        "type_haystack": "noise",
        "type_needle_k": "KEY4",
        "type_needle_v": "numbers7",
        "native": True,
    },
    "niah_multiquery": {
        "kind": "niah",
        "official_name": "niah_multiquery",
        "num_needle_k": 4,
        "num_needle_v": 1,
        "num_needle_q": 4,
        "type_haystack": "noise",
        "type_needle_k": "KEY4",
        "type_needle_v": "numbers7",
        "note": "Official yaml sets num_needle_k=1,num_needle_q=4 then code does k=max(k,q)=4. Haystack is harness noise (same as niah_s/mk), not official essay.",
    },
    "niah_multivalue": {
        "kind": "niah",
        "official_name": "niah_multivalue",
        "num_needle_k": 1,
        "num_needle_v": 4,
        "num_needle_q": 1,
        "type_haystack": "noise",
        "type_needle_k": "KEY4",
        "type_needle_v": "numbers7",
        "note": "Official: 1 key × 4 values. Haystack is harness noise (same as niah_s/mk), not official essay.",
    },
    "cwe": {
        "kind": "common_words_extraction",
        "official_name": "cwe",
        "freq_cw": 30,
        "freq_ucw": 3,
        "num_cw": 10,
        "num_fewshot": 1,
        "metric": "string_match_all",
    },
    "fwe": {
        "kind": "freq_words_extraction",
        "official_name": "fwe",
        "alpha": 2.0,
        "coded_wordlen": 6,
        "vocab_size_rule": "max(32, ctx//50)",
        "n_answer": 3,
        "noise_token": "...",
        "metric": "string_match_all",
    },
    "qa": {
        "kind": "synthetic_fact_qa",
        "native": True,
        "note": "Harness-native city-year fact. NOT official RULER qa_1 (SQuAD) or qa_2 (HotpotQA); those need external datasets and are not split here.",
    },
    "vt": {
        "kind": "variable_tracking",
        "num_chains": 1,
        "num_hops": 7,
        "native": True,
    },
}
_CWE_BASE_WORDS = (
    "apple river mountain forest ocean desert island valley canyon meadow "
    "thunder lightning rainbow sunset sunrise cloud storm breeze shadow "
    "hammer lantern candle window garden castle bridge tunnel harbor meadow "
    "silver copper marble granite amber ivory velvet crimson azure golden "
    "rapid silent ancient modern humble noble clever humble quiet bright "
    "wander gather wanderer farmer sailor painter singer dancer teacher "
    "circle square triangle spiral ladder basket bottle ribbon feather "
    "willow cedar maple oak pine birch cherry blossom petal nectar "
    "falcon eagle sparrow robin heron otter beaver badger fox wolf "
    "copper kettle lantern orchard vineyard meadow brook pebble canyon "
    "winter summer autumn spring harvest festival carnival parade lantern "
    "honest gentle patient brave humble loyal sincere modest cheerful "
    "puzzle riddle secret message signal beacon compass anchor lantern "
    "cotton linen wool silk satin denim leather amber coral pearl "
    "north south east west valley plateau prairie tundra glacier "
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet "
    "kettle lantern meadow nectar orchard pebble quartz ribbon saddle "
    "timber umbrella valley willow xenon yellow zephyr amber bronze "
    "candle daisy ember frost grove haven iris jasmine kernel lotus "
    "mango nickel opal poppy quartz raven saffron tulip umber violet "
    "walnut xenial yarn zenith apricot barley celery durian eggplant "
).split()


def _riemann_zeta(alpha: float) -> float:
    if abs(alpha - 2.0) < 1e-9:
        return math.pi ** 2 / 6.0
    return float(sum(k ** (-alpha) for k in range(1, 100000)))


def _unique_labels(rng: random.Random, n: int, factory: Callable[[], str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    while len(out) < n:
        x = factory()
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _cwe_vocab(n: int, rng: random.Random) -> list[str]:
    base = list(dict.fromkeys(_CWE_BASE_WORDS))
    rng.shuffle(base)
    if n <= len(base):
        return base[:n]
    extra: list[str] = []
    seen = set(base)
    while len(base) + len(extra) < n:
        w = "".join(rng.choice(string.ascii_lowercase) for _ in range(6))
        if w not in seen:
            seen.add(w)
            extra.append(w)
    return base + extra


def _scatter_needles(tokenizer: Any, needles: list[str], query: str, ctx: int, rng: random.Random) -> str:
    """Place needles at distinct depths in the noise haystack, then append the query."""
    n = len(needles)
    depths = sorted(rng.uniform(0.08, 0.88) for _ in range(n))
    needle_ids = [tokenizer.encode(nd + " ", add_special_tokens=False) for nd in needles]
    query_s = "\n" + query
    query_ids = tokenizer.encode(query_s, add_special_tokens=False)
    fill_ids = tokenizer.encode(HAYSTACK, add_special_tokens=False)
    used = sum(len(t) for t in needle_ids) + len(query_ids)
    remain = max(0, ctx - used)
    cuts = depths + [1.0]
    segs: list[int] = []
    prev = 0.0
    for cut in cuts:
        segs.append(int(remain * (cut - prev)))
        prev = cut
    segs[-1] += remain - sum(segs)
    body: list[int] = []
    for i, nd in enumerate(needle_ids):
        chunk: list[int] = []
        while len(chunk) < segs[i]:
            chunk.extend(fill_ids)
        body.extend(chunk[: segs[i]])
        body.extend(nd)
    chunk = []
    while len(chunk) < segs[-1]:
        chunk.extend(fill_ids)
    body.extend(chunk[: segs[-1]])
    body.extend(query_ids)
    return tokenizer.decode(body, skip_special_tokens=True)


def _fit_text_to_ctx(tokenizer: Any, build: Callable[[int], str], ctx: int, lo: int, hi: int) -> tuple[str, int]:
    """Largest `size` in [lo, hi] whose tokenized length is <= ctx (official RULER-style search)."""
    best_text = build(lo)
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        text = build(mid)
        n_tok = len(tokenizer.encode(text, add_special_tokens=False))
        if n_tok <= ctx:
            best_text, best = text, mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best_text, best


def _make_cwe_with_answers(tokenizer: Any, ctx: int, rng: random.Random) -> tuple[str, list[str], dict[str, Any]]:
    """CWE with official freq_cw=30, freq_ucw=3, num_cw=10, 1-shot (seq>=4K)."""
    freq_cw, freq_ucw, num_cw = 30, 3, 10
    seed_i = rng.randint(0, 2**31 - 1)
    prefix = (
        "Below is a numbered list of words. In these words, some appear more often "
        "than others. Memorize the ones that appear most often.\n"
    )
    query = (
        "\nQuestion: What are the 10 most common words in the above list?\n"
        "Answer: The top 10 words that appear most often in the list are:"
    )

    def _pack(local: random.Random, n_unique: int) -> tuple[str, list[str], int]:
        fs_words = _cwe_vocab(40, local)
        fs_common, fs_uncommon = fs_words[:num_cw], fs_words[num_cw:]
        fs_list = fs_common * 10 + fs_uncommon * 3
        local.shuffle(fs_list)
        fs_ctx = " ".join(f"{i + 1}. {w}" for i, w in enumerate(fs_list))
        fewshot = prefix + fs_ctx + query + " " + " ".join(f"{i + 1}. {w}" for i, w in enumerate(fs_common)) + "\n"
        words = _cwe_vocab(max(n_unique, num_cw + 1), local)
        common, uncommon = words[:num_cw], words[num_cw:]
        word_list = common * freq_cw + uncommon * freq_ucw
        local.shuffle(word_list)
        context = " ".join(f"{i + 1}. {w}" for i, w in enumerate(word_list))
        return fewshot + prefix + context + query, common, len(word_list)

    def _build(n_unique: int) -> str:
        text, _, _ = _pack(random.Random(seed_i), n_unique)
        return text

    overhead_probe, _, _ = _pack(random.Random(seed_i), num_cw + 8)
    overhead = len(tokenizer.encode(overhead_probe, add_special_tokens=False))
    # crude remain estimate from a small unique count
    small = overhead
    remain = max(64, ctx - max(1, small // 4))
    probe = " ".join(f"{i + 1}. word" for i in range(20))
    per = max(1.0, len(tokenizer.encode(probe, add_special_tokens=False)) / 20.0)
    n_items_est = max(num_cw * freq_cw + freq_ucw, int(remain / per))
    n_uncommon_est = max(1, (n_items_est - num_cw * freq_cw) // freq_ucw)
    n_unique_est = num_cw + n_uncommon_est
    text, n_unique = _fit_text_to_ctx(tokenizer, _build, ctx, num_cw + 1, max(num_cw + 2, n_unique_est))
    text, common, n_items = _pack(random.Random(seed_i), n_unique)
    meta = {
        "freq_cw": freq_cw,
        "freq_ucw": freq_ucw,
        "num_cw": num_cw,
        "num_unique_words": n_unique,
        "num_list_items": n_items,
        "num_fewshot": 1,
        "vocab_base": len(set(_CWE_BASE_WORDS)),
    }
    return text, common, meta


def _make_fwe_with_answers(tokenizer: Any, ctx: int, rng: random.Random) -> tuple[str, list[str], dict[str, Any]]:
    """FWE with official Zipf α=2.0, 6-letter coded words, answers = ranks 2–4 (rank 1 is '...')."""
    alpha, coded_wordlen = 2.0, 6
    vocab_size = max(32, ctx // 50)
    prefix = (
        "Read the following coded text and track the frequency of each coded word. "
        "Find the three most frequently appeared coded words. "
    )
    query = (
        "\nQuestion: Do not provide any explanation. Please ignore the dots '....'. "
        "What are the three most frequently appeared words in the above coded text?\n"
        "Answer: According to the coded text above, the three most frequently appeared words are:"
    )
    seed_i = rng.randint(0, 2**31 - 1)

    def _pack(local: random.Random, num_words: int) -> tuple[str, list[str], list[int]]:
        vocab: list[str] = []
        seen: set[str] = set()
        while len(vocab) < vocab_size:
            w = "".join(local.choice(string.ascii_lowercase) for _ in range(coded_wordlen))
            if w not in seen:
                seen.add(w)
                vocab.append(w)
        vocab = sorted(vocab)
        local.shuffle(vocab)
        vocab[0] = "..."
        z = _riemann_zeta(alpha)
        counts = [max(0, int(num_words * ((i + 1) ** (-alpha)) / z)) for i in range(len(vocab))]
        sampled: list[str] = []
        for w, c in zip(vocab, counts):
            sampled.extend([w] * c)
        local.shuffle(sampled)
        text = prefix + " ".join(sampled) + query
        return text, vocab[1:4], counts

    def _build(num_words: int) -> str:
        text, _, _ = _pack(random.Random(seed_i), num_words)
        return text

    overhead = len(tokenizer.encode(prefix + query, add_special_tokens=False))
    remain = max(64, ctx - overhead)
    probe = " ".join(["abcdef"] * 20)
    per = max(1.0, len(tokenizer.encode(probe, add_special_tokens=False)) / 20.0)
    n_est = max(vocab_size, int(remain / per))
    text, num_words = _fit_text_to_ctx(tokenizer, _build, ctx, max(32, vocab_size), max(vocab_size + 8, n_est))
    text, answers, counts = _pack(random.Random(seed_i), num_words)
    meta = {
        "alpha": alpha,
        "coded_wordlen": coded_wordlen,
        "vocab_size": vocab_size,
        "num_sampled_slots": num_words,
        "n_answer": 3,
        "top_counts": counts[1:4],
        "noise_token": "...",
    }
    return text, answers, meta


def _ruler_score(task: str, pred: str, answers: list[str]) -> float:
    """Existing tasks keep the old rule; multi-answer extensions use official string_match_all."""
    if task in {"niah_multiquery", "niah_multivalue", "cwe", "fwe"}:
        if not answers:
            return 0.0
        pred_l = pred.lower()
        return sum(1.0 if ans.lower() in pred_l else 0.0 for ans in answers) / len(answers)
    score = max(qa_f1_score(pred, ans) for ans in answers)
    return 1.0 if any(ans in pred for ans in answers) else score


def _fill_to_tokens(tokenizer: Any, prefix: str, suffix: str, target: int) -> str:
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    fill_ids = tokenizer.encode(HAYSTACK, add_special_tokens=False)
    need = max(0, target - len(prefix_ids) - len(suffix_ids))
    body: list[int] = []
    while len(body) < need:
        body.extend(fill_ids)
    body = body[:need]
    return tokenizer.decode(prefix_ids + body + suffix_ids, skip_special_tokens=True)


def make_ruler_samples(tokenizer: Any, task: str, n: int, ctx: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    samples = []
    for i in range(n):
        gen_meta: dict[str, Any] | None = None
        if task == "niah_s":
            key = f"KEY{rng.randint(1000, 9999)}"
            value = f"{rng.randint(1000000, 9999999)}"
            needle = f"One of the special magic numbers for {key} is {value}."
            query = f"What is the special magic number for {key}? Answer with the number only.\nAnswer:"
            depth = rng.uniform(0.1, 0.9)
            prefix_target = int(ctx * depth)
            text = _fill_to_tokens(tokenizer, "", needle + " ", prefix_target)
            text = _fill_to_tokens(tokenizer, text, "\n" + query, ctx)
            answers = [value]
        elif task == "niah_mk":
            pairs = [(f"KEY{rng.randint(1000, 9999)}", f"{rng.randint(1000000, 9999999)}") for _ in range(4)]
            ask_k, ask_v = pairs[rng.randrange(len(pairs))]
            needles = " ".join(f"One of the special magic numbers for {k} is {v}." for k, v in pairs)
            query = f"What is the special magic number for {ask_k}? Answer with the number only.\nAnswer:"
            text = _fill_to_tokens(tokenizer, needles + " ", "\n" + query, ctx)
            answers = [ask_v]
        elif task == "qa":
            city = rng.choice(["Lisbon", "Nairobi", "Osaka", "Bergen", "Cusco", "Dakar"])
            year = rng.randint(1400, 1900)
            fact = f"The city of {city} was founded in the year {year}."
            query = f"In what year was {city} founded? Answer with the year only.\nAnswer:"
            text = _fill_to_tokens(tokenizer, fact + " ", "\n" + query, ctx)
            answers = [str(year)]
        elif task == "vt":
            names = [f"VAR{c}" for c in string.ascii_uppercase[:8]]
            assigns = [f"{names[0]} = {rng.randint(1, 9)}"]
            for j in range(1, len(names)):
                assigns.append(f"{names[j]} = {names[j-1]}")
            chain = ". ".join(assigns) + "."
            query = f"What is the value of {names[-1]}? Answer with the number only.\nAnswer:"
            text = _fill_to_tokens(tokenizer, chain + " ", "\n" + query, ctx)
            answers = [assigns[0].split("=")[1].strip()]
        elif task == "niah_multiquery":
            keys = _unique_labels(rng, 4, lambda: f"KEY{rng.randint(1000, 9999)}")
            values = _unique_labels(rng, 4, lambda: f"{rng.randint(1000000, 9999999)}")
            needles = [f"One of the special magic numbers for {k} is {v}." for k, v in zip(keys, values)]
            keys_txt = ", ".join(keys[:-1]) + f", and {keys[-1]}"
            query = (
                f"What are all the special magic numbers for {keys_txt}? "
                "Answer with the numbers only, in any order.\nAnswer:"
            )
            text = _scatter_needles(tokenizer, needles, query, ctx, rng)
            answers = values
            gen_meta = {
                "num_needle_k": 4,
                "num_needle_v": 1,
                "num_needle_q": 4,
                "keys": keys,
                "type_haystack": "noise",
            }
        elif task == "niah_multivalue":
            key = f"KEY{rng.randint(1000, 9999)}"
            values = _unique_labels(rng, 4, lambda: f"{rng.randint(1000000, 9999999)}")
            needles = [f"One of the special magic numbers for {key} is {v}." for v in values]
            query = (
                f"What are all the special magic numbers for {key}? "
                "Answer with the numbers only, in any order.\nAnswer:"
            )
            text = _scatter_needles(tokenizer, needles, query, ctx, rng)
            answers = values
            gen_meta = {
                "num_needle_k": 1,
                "num_needle_v": 4,
                "num_needle_q": 1,
                "key": key,
                "type_haystack": "noise",
            }
        elif task == "cwe":
            text, answers, gen_meta = _make_cwe_with_answers(tokenizer, ctx, rng)
        elif task == "fwe":
            text, answers, gen_meta = _make_fwe_with_answers(tokenizer, ctx, rng)
        else:
            raise ValueError(task)
        rec = {"_id": f"{task}-{i}", "task": task, "prompt": text, "answers": answers}
        if gen_meta is not None:
            rec["gen_meta"] = gen_meta
        samples.append(rec)
    return samples


def run_ruler(model: Any, tokenizer: Any, args: argparse.Namespace) -> dict[str, Any]:
    out_path = Path(args.jsonl_output)
    done = already_done(out_path)
    requested = {t.strip() for t in args.ruler_tasks.split(",") if t.strip()}
    tasks = [t for t in RULER_TASK_ORDER if t in requested]
    task_scores: dict[str, list[float]] = {t: [] for t in tasks}
    if out_path.exists():
        with out_path.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row["suite"] == "ruler" and row["task"] in task_scores:
                    task_scores[row["task"]].append(float(row["score"]))
    for task in tasks:
        samples = make_ruler_samples(tokenizer, task, args.ruler_samples, args.ruler_tokens, args.seed)
        for row in samples:
            sid = row["_id"]
            if ("ruler", task, sid) in done:
                continue
            input_ids = encode_eval_prompt(tokenizer, row["prompt"], args.ruler_tokens)
            gen_steps = int(RULER_GEN_STEPS.get(task, 16))
            pred = run_one_sample(model, tokenizer, args, args.method, input_ids, gen_steps)
            acc = _ruler_score(task, pred, row["answers"])
            rec = {
                "suite": "ruler",
                "task": task,
                "_id": sid,
                "method": args.method,
                "pred": pred,
                "score": acc,
                "n_input_tokens": int(input_ids.shape[1]),
                "gen_steps": gen_steps,
            }
            if args.method in {
                "quest", "quest_channel", "quest_calibrated", "snapkv", "snapkv_channel",
                "sts", "pariskv",
            }:
                rec["token_budget"] = int(args.token_budget)
            if args.method == "sts":
                if getattr(args, "sts_budget_p", None) is not None:
                    rec["budget_p"] = float(args.sts_budget_p)
                rec["sts_head_map_stats"] = getattr(args, "_last_sts_head_map_stats", None)
                rec["sts_draft_real"] = bool(getattr(args, "_sts_draft_model", None) is not None)
                if getattr(args, "_sts_draft_model", None) is not None:
                    rec["sts_gen_stats"] = getattr(args, "_last_sts_gen_stats", None)
            if args.method == "pariskv":
                if getattr(args, "pariskv_budget_p", None) is not None:
                    rec["budget_p"] = float(args.pariskv_budget_p)
                rec["pariskv_final_topk"] = int(getattr(args, "_last_pariskv_final_topk", 0))
                rec["pariskv_stats"] = getattr(args, "_last_pariskv_stats", None)
            if getattr(args, "quest_budget_p", None) is not None and args.method.startswith("quest"):
                rec["budget_p"] = float(args.quest_budget_p)
            if args.method.startswith("quest"):
                rec["quest_chunk_size"] = int(args.quest_chunk_size)
            if getattr(args, "snapkv_budget_p", None) is not None and args.method.startswith("snapkv"):
                rec["budget_p"] = float(args.snapkv_budget_p)
            if args.method == "quest_channel":
                rec["denom_channel"] = True
                rec["bg_mode"] = str(getattr(args, "quest_bg_mode", "page_mean_key"))
                rec["bg_alpha"] = float(getattr(args, "quest_bg_alpha", 1.0))
                rec["bg_delta"] = float(getattr(args, "quest_bg_delta", 0.0))
                rec["bg_margin"] = float(getattr(args, "quest_bg_margin", 0.0))
                rec["bg_estimator"] = rec["bg_mode"]
                rec["bg_value"] = (
                    "exact_token" if rec["bg_mode"] in {"alpha_true", "trunc_true"} else "page_repr"
                )
                rec.update(getattr(args, "_last_scan_stats", {}) or {})
            if args.method == "snapkv_channel":
                rec["denom_channel"] = True
                rec["bg_estimator"] = "snapkv_evicted_page_mean_key"
                rec["bg_value"] = "evicted_page_mean"
            if row.get("gen_meta"):
                rec["gen_meta"] = row["gen_meta"]
            append_jsonl(out_path, rec)
            task_scores[task].append(acc)
            print(json.dumps({"event": "ruler_sample", **{k: rec[k] for k in rec if k != "pred"}}, sort_keys=True), flush=True)
        mean = sum(task_scores[task]) / max(1, len(task_scores[task]))
        print(json.dumps({"event": "ruler_task", "task": task, "n": len(task_scores[task]), "score": mean}, sort_keys=True), flush=True)
    means = {t: (sum(v) / len(v) if v else None) for t, v in task_scores.items()}
    present = [v for v in means.values() if v is not None]
    return {
        "tasks": means,
        "average": (sum(present) / len(present) if present else None),
        "task_meta": {t: RULER_TASK_META[t] for t in tasks if t in RULER_TASK_META},
        "qa_note": RULER_TASK_META["qa"]["note"],
    }



# ---------------------------------------------------------------------------
# InfiniteBench. Official metrics ported from
# (official InfiniteBench src/compute_scores.py)
# (verified against third_party/PQCache/InfLLM/benchmark/infinitebench_eval.py,
# which carries the same upstream attribution). Harness-only; does not touch
# pq_hsa/ or the existing longbench/ruler code paths.
# ---------------------------------------------------------------------------

INFINITEBENCH_DIR = ROOT / "data" / "infinitebench"

INFINITEBENCH_TASK_ORDER = [
    "passkey",
    "number_string",
    "kv_retrieval",
    "longbook_choice_eng",
    "longbook_qa_eng",
    "math_find",
]

INFINITEBENCH_DISPLAY: dict[str, str] = {
    "passkey": "Retrieve.PassKey",
    "number_string": "Retrieve.Number",
    "kv_retrieval": "Retrieve.KV",
    "longbook_choice_eng": "En.MC",
    "longbook_qa_eng": "En.QA",
    "math_find": "Math.Find",
}

# Official dataset2maxlen.json values (OpenBMB/InfiniteBench).
INFINITEBENCH_MAXGEN: dict[str, int] = {
    "passkey": 32,
    "number_string": 32,
    "kv_retrieval": 100,
    "longbook_choice_eng": 32,
    "longbook_qa_eng": 32,
    "math_find": 32,
}

# Official dataset2prompt.json templates (OpenBMB/InfiniteBench), verbatim.
INFINITEBENCH_PROMPTS: dict[str, str] = {
    "passkey": (
        "There is an important info hidden inside a lot of irrelevant text. "
        "Find it and memorize them. I will quiz you about the important "
        "information there.\n\n{context}\n\n{input}"
    ),
    "number_string": (
        "There is an important info hidden inside a lot of irrelevant text. "
        "Find it. I will quiz you about the important information there.\n\n"
        "{context}\n\n{input}"
    ),
    "kv_retrieval": (
        "Extract the value corresponding to the specified key {key} in the "
        "JSON object below.\n\n{context}\n\n{input}"
    ),
    "longbook_qa_eng": (
        "Read the book below and answer a question.\n\n{context}\n\n"
        "Question: {question}\n\nPlease answer as short as possible. The answer is:"
    ),
    "longbook_choice_eng": (
        "Read the book and answer the question.\n\n{context}\n\n"
        "Question: {question}\n\nOnly one of the following options is correct, "
        "tell me the answer using one single letter (A, B, C, or D). Don't say "
        "anything else.\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}"
    ),
    "math_find": "{prefix}\n\n{context}\n\n{input}",
}

_IB_MATH_FIND_RE = re.compile(r"The .+ of")


def load_infinitebench_rows(task: str, n: int) -> list[dict[str, Any]]:
    """Deterministic first-n rows (file order), streamed so we never load the
    full multi-hundred-MB jsonl into memory."""
    path = INFINITEBENCH_DIR / f"{task}.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if len(rows) >= n:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _ib_math_find_prefix(user_input: str) -> str:
    m = _IB_MATH_FIND_RE.findall(user_input)
    if not m:
        raise ValueError(f"cannot find target-number phrase in: {user_input[:80]!r}")
    target_number = m[0].lower()[:-3]
    return f"What is {target_number} in the following list?"


def build_infinitebench_prompt(task: str, row: dict[str, Any]) -> tuple[str, list[Any]]:
    """Return (prompt_text, answers) exactly like InfLLM/pred.py's get_answer +
    dataset2prompt formatting."""
    context = row["context"]
    ans = row["answer"]
    if task in {"passkey", "number_string"}:
        prompt = INFINITEBENCH_PROMPTS[task].format(context=context, input=row["input"])
        answers = ans if isinstance(ans, list) else [ans]
    elif task == "kv_retrieval":
        inp = row["input"]
        if inp[6] != '"' or inp[43] != '"':
            raise ValueError(f"unexpected kv_retrieval input format: {inp[:60]!r}")
        key = inp[6:44]
        prompt = INFINITEBENCH_PROMPTS[task].format(context=context, input=inp, key=key)
        answers = ans if isinstance(ans, list) else [ans]
    elif task == "longbook_qa_eng":
        prompt = INFINITEBENCH_PROMPTS[task].format(context=context, question=row["input"])
        answers = ans if isinstance(ans, list) else [ans]
    elif task == "longbook_choice_eng":
        options = row["options"]
        prompt = INFINITEBENCH_PROMPTS[task].format(
            context=context,
            question=row["input"],
            OPTION_A=options[0], OPTION_B=options[1], OPTION_C=options[2], OPTION_D=options[3],
        )
        ans_list = ans if isinstance(ans, list) else [ans]
        answer_text = ans_list[0]
        letter = "ABCD"[options.index(answer_text)]
        answers = [answer_text, letter]
    elif task == "math_find":
        prefix = _ib_math_find_prefix(row["input"])
        prompt = INFINITEBENCH_PROMPTS[task].format(prefix=prefix, context=context, input=row["input"])
        answers = ans if isinstance(ans, list) else [ans]
    else:
        raise ValueError(f"unknown infinitebench task: {task}")
    return prompt, answers


def _ib_normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _ib_f1(pred_tokens: list[str], gt_tokens: list[str]) -> float:
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def _ib_qa_f1(pred: str, ground_truths: list[str]) -> float:
    best = 0.0
    for gt in ground_truths:
        p_toks = _ib_normalize_answer(pred).split()
        g_toks = _ib_normalize_answer(gt).split()
        best = max(best, _ib_f1(p_toks, g_toks))
    return best


def _ib_first_int(pred: str) -> str:
    for item in re.split("[^0-9]", pred):
        if item != "":
            return item
    return ""


def _ib_score_passkey(pred: str, label: Any) -> float:
    return 1.0 if str(label) == _ib_first_int(pred) else 0.0


def _ib_score_kv(pred: str, label: Any) -> float:
    p = pred
    for c in ["\n", ":", '"', "'", ".", ",", "?", "!", "{", "}"]:
        p = p.replace(c, " ")
    return 1.0 if str(label) in p.split() else 0.0


def _ib_score_math_find(pred: str, label: Any) -> float:
    m = re.search(r"\d+\.\d+|\d+", pred)
    if m is None:
        return 0.0
    val = m.group(0).strip()
    if isinstance(label, float):
        try:
            return 1.0 if float(val) == label else 0.0
        except ValueError:
            return 0.0
    try:
        return 1.0 if int(val) == int(label) else 0.0
    except ValueError:
        return 0.0


def _ib_score_choice(pred: str, label: Any) -> float:
    """Byte-faithful port of official get_score_one_longbook_choice_eng."""
    if pred and pred[0] in "ABCD":
        return 1.0 if pred[0] == label else 0.0
    p = pred
    for c in ["\n", '"', "'", ".", ",", "?", "!", "{", "}"]:
        p = p.replace(c, " ")
    while "  " in p:
        p = p.replace("  ", " ")
    for prefix in ("answer is:", "answer:", "answer is", "option is"):
        idx = p.find(prefix)
        if idx == -1:
            continue
        if len(p) < idx + len(prefix) + 1:
            return 0.0
        after = p[idx + len(prefix) + 1 :]
        for s in label:
            if after.startswith(s):
                return 1.0
        return 0.0
    for word in p.split():
        if word in "ABCD":
            return 1.0 if word == label else 0.0
    return 0.0


def infinitebench_score(task: str, pred: str, answers: list[Any]) -> float:
    if task == "passkey":
        return max(_ib_score_passkey(pred, a) for a in answers)
    if task == "number_string":
        return max(_ib_score_passkey(pred, a) for a in answers)
    if task == "kv_retrieval":
        return max(_ib_score_kv(pred, a) for a in answers)
    if task == "longbook_qa_eng":
        return _ib_qa_f1(pred, answers)
    if task == "longbook_choice_eng":
        return max(_ib_score_choice(pred, a) for a in answers)
    if task == "math_find":
        return max(_ib_score_math_find(pred, a) for a in answers)
    raise ValueError(f"unknown infinitebench task: {task}")


def run_infinitebench(model: Any, tokenizer: Any, args: argparse.Namespace) -> dict[str, Any]:
    """InfiniteBench @128K, six subtasks, five methods (dense/pqhsa/sink_local
    aka truncation/quest/snapkv). Method dispatch reuses run_one_sample verbatim;
    quest/snapkv/sink_local get a per-sample aligned token budget because
    (unlike RULER's fixed-length synthetic prompts) real InfiniteBench
    documents vary a lot in length even within one task.
    """
    out_path = Path(args.jsonl_output)
    done = already_done(out_path)
    requested = {t.strip() for t in args.infinitebench_tasks.split(",") if t.strip()}
    tasks = [t for t in INFINITEBENCH_TASK_ORDER if t in requested]
    task_scores: dict[str, list[float]] = {t: [] for t in tasks}
    if out_path.exists():
        with out_path.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row["suite"] == "infinitebench" and row["task"] in task_scores:
                    task_scores[row["task"]].append(float(row["score"]))

    orig_local_window = args.local_window
    orig_token_budget = args.token_budget
    budget_methods = {"quest", "quest_channel", "snapkv", "snapkv_channel"}

    for task in tasks:
        rows = load_infinitebench_rows(task, args.infinitebench_samples)
        max_gen = INFINITEBENCH_MAXGEN[task]
        max_input = args.infinitebench_max_tokens - max_gen
        for row in rows:
            sid = str(row["id"])
            if ("infinitebench", task, sid) in done:
                continue
            prompt_text, answers = build_infinitebench_prompt(task, row)
            n_raw = len(tokenizer.encode(prompt_text, add_special_tokens=False))
            input_ids = middle_truncate(tokenizer, prompt_text, max_input)
            n_kept = int(input_ids.shape[1])
            truncated = n_raw > n_kept
            trunc_ratio = 0.0 if n_raw <= 0 else max(0.0, 1.0 - (n_kept / n_raw))
            budget = quest_aligned_token_budget(0.01, n_kept)
            error = None
            try:
                if args.method in budget_methods:
                    args.token_budget = int(budget)
                if args.method == "sink_local":
                    sink0 = int(str(args.sink).split(",")[0])
                    args.local_window = str(max(1, budget - sink0))
                pred = run_one_sample(model, tokenizer, args, args.method, input_ids, max_gen)
                score = infinitebench_score(task, pred, answers)
            except Exception as exc:  # long-running unattended job: skip, don't crash
                pred = ""
                score = 0.0
                error = f"{type(exc).__name__}: {exc}"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            finally:
                args.local_window = orig_local_window
                args.token_budget = orig_token_budget
            rec: dict[str, Any] = {
                "suite": "infinitebench",
                "task": task,
                "task_display": INFINITEBENCH_DISPLAY[task],
                "_id": sid,
                "method": args.method,
                "pred": pred,
                "score": float(score),
                "answers": answers,
                "n_input_tokens_raw": int(n_raw),
                "n_input_tokens": int(n_kept),
                "context_truncated": bool(truncated),
                "truncation_ratio": float(trunc_ratio),
                "gen_steps": int(max_gen),
                "error": error,
            }
            if args.method in budget_methods or args.method == "sink_local":
                rec["token_budget"] = int(budget)
            append_jsonl(out_path, rec)
            task_scores[task].append(float(score))
            print(
                json.dumps(
                    {"event": "infinitebench_sample", **{k: rec[k] for k in rec if k not in ("pred", "answers")}},
                    sort_keys=True,
                ),
                flush=True,
            )
        mean = sum(task_scores[task]) / max(1, len(task_scores[task]))
        print(
            json.dumps({"event": "infinitebench_task", "task": task, "n": len(task_scores[task]), "score": mean}, sort_keys=True),
            flush=True,
        )
    means = {t: (sum(v) / len(v) if v else None) for t, v in task_scores.items()}
    present = [v for v in means.values() if v is not None]
    return {
        "tasks": means,
        "tasks_display": {t: INFINITEBENCH_DISPLAY[t] for t in tasks},
        "average": (sum(present) / len(present) if present else None),
    }


def parse_extra() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=(
            "dense",
            "sink_local",
            "pqhsa",
            "quest",
            "quest_channel",
            "quest_calibrated",
            "snapkv",
            "snapkv_channel",
            "retrieval_attention",
            "sts",
            "pariskv",
        ),
    )
    parser.add_argument(
        "--suite", default="longbench", choices=("longbench", "ruler", "both", "infinitebench")
    )
    parser.add_argument("--tasks", default=",".join(EN_TASKS))
    parser.add_argument("--ruler-tasks", default="niah_s,niah_mk,qa,vt")
    parser.add_argument("--ruler-samples", type=int, default=50)
    parser.add_argument(
        "--infinitebench-tasks", default=",".join(INFINITEBENCH_TASK_ORDER)
    )
    parser.add_argument("--infinitebench-samples", type=int, default=50)
    parser.add_argument("--infinitebench-max-tokens", type=int, default=131072)
    parser.add_argument("--ruler-tokens", type=int, default=131072)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=130000)
    parser.add_argument(
        "--token-budget",
        type=int,
        default=512,
        help="Quest/SnapKV exact-token budget (v2 default 512; matched/grid uses the aligned-budget formula).",
    )
    parser.add_argument(
        "--quest-chunk-size",
        type=int,
        default=16,
        help="Quest page size / chunk size (paper + v2: 16).",
    )
    parser.add_argument(
        "--quest-budget-p",
        type=float,
        default=None,
        help="Optional grid fraction p used to derive token-budget; recorded in json.",
    )
    parser.add_argument(
        "--quest-bg-mode",
        default="page_mean_key",
        choices=("upper_bound", "ub_margin", "page_mean_key", "ub_calibrated", "alpha_true", "trunc_true"),
        help="Host-native background estimator when --method quest_channel.",
    )
    parser.add_argument(
        "--quest-bg-alpha",
        type=float,
        default=1.0,
        help="For alpha_true: shrink toward bg mean, bg = m + α(true−m) + δ. α=1 is oracle.",
    )
    parser.add_argument(
        "--quest-bg-delta",
        type=float,
        default=0.0,
        help="For alpha_true: level shift in nats after mean-centered shrink. Orthogonal to α.",
    )
    parser.add_argument(
        "--quest-bg-margin",
        type=float,
        default=0.0,
        help="For ub_margin: subtract this many nats from the page-max-key bound.",
    )
    parser.add_argument(
        "--snapkv-obs-window",
        type=int,
        default=snapkv.SNAPKV_OBS_WINDOW,
        help="SnapKV observation window (paper LongBench default: 32).",
    )
    parser.add_argument(
        "--snapkv-pool-kernel",
        type=int,
        default=snapkv.SNAPKV_POOL_KERNEL,
        help="SnapKV 1D max-pool kernel (paper LongBench default: 7).",
    )
    parser.add_argument(
        "--snapkv-budget-p",
        type=float,
        default=None,
        help="Optional grid fraction p used to derive SnapKV token-budget; recorded in json.",
    )
    parser.add_argument(
        "--ra-budget-p",
        type=float,
        default=None,
        help="Optional grid fraction p used to derive RetrievalAttention token-budget; recorded in json.",
    )
    parser.add_argument("--ra-sink", type=int, default=ra.RA_SINK, help="RA static initial tokens (paper: 128).")
    parser.add_argument("--ra-local", type=int, default=ra.RA_LOCAL, help="RA static local window (paper: 512).")
    parser.add_argument("--ra-n-queries", type=int, default=ra.RA_N_QUERIES)
    parser.add_argument("--ra-knn", type=int, default=ra.RA_KNN)
    parser.add_argument("--ra-nlist", type=int, default=ra.RA_NLIST)
    parser.add_argument("--ra-nprobe", type=int, default=ra.RA_NPROBE)
    parser.add_argument("--ra-nprobe-q", type=int, default=ra.RA_NPROBE_Q)
    parser.add_argument(
        "--sts-budget-p",
        type=float,
        default=None,
        help="Optional grid fraction p used to derive STS-harness token-budget; recorded in json.",
    )
    parser.add_argument(
        "--sts-draft-layer",
        type=int,
        default=sts.STS_DRAFT_LAYER,
        help="Proxy draft layer index (used when --sts-draft-model is not given).",
    )
    parser.add_argument("--sts-calib-tokens", type=int, default=sts.STS_CALIB_TOKENS)
    parser.add_argument("--sts-calib-topk", type=int, default=sts.STS_CALIB_TOPK)
    parser.add_argument(
        "--sts-draft-model",
        default=None,
        help=(
            "Path or HF id of a real Llama-3.2-1B-Instruct checkpoint. When set, "
            "--method sts loads it as a second model + its own KV cache and uses "
            "baselines_sts.install_sts_real (real cross-model draft, every target "
            "layer sparsifiable) instead of the proxy draft (a single shallow layer "
            "of the target itself, --sts-draft-layer). Default unset keeps the "
            "proxy path."
        ),
    )
    parser.add_argument(
        "--pariskv-budget-p",
        type=float,
        default=None,
        help="Optional grid fraction p used to derive ParisKV token-budget; recorded in json.",
    )
    parser.add_argument(
        "--pariskv-subspaces",
        type=int,
        default=pariskv.PARISKV_B,
        help="ParisKV number of rotated subspaces B (paper's D=128 example: B=16, m=8).",
    )
    parser.add_argument(
        "--pariskv-rho",
        type=float,
        default=pariskv.PARISKV_RHO,
        help="ParisKV per-subspace top-rho fraction that can score in Stage-I collision voting (rho>=beta).",
    )
    parser.add_argument(
        "--pariskv-beta",
        type=float,
        default=pariskv.PARISKV_BETA,
        help="ParisKV top-beta fraction of all keys kept as Stage-I candidates (paper: typically 5-10%%).",
    )
    parser.add_argument(
        "--pariskv-rotation-seed",
        type=int,
        default=pariskv.PARISKV_ROTATION_SEED,
        help="Seed for the shared Haar-random rotation (stand-in for SRHT, see baselines_pariskv.py).",
    )
    known, rest = parser.parse_known_args()
    return known, rest


def main() -> None:
    extra, rest = parse_extra()
    sys.argv = [sys.argv[0], *rest]
    args = e2e.parse_args()
    args.method = extra.method
    args.suite = extra.suite
    args.tasks = extra.tasks
    args.ruler_tasks = extra.ruler_tasks
    args.ruler_samples = extra.ruler_samples
    args.ruler_tokens = extra.ruler_tokens
    args.infinitebench_tasks = extra.infinitebench_tasks
    args.infinitebench_samples = extra.infinitebench_samples
    args.infinitebench_max_tokens = extra.infinitebench_max_tokens
    args.max_samples = extra.max_samples
    args.max_input_tokens = extra.max_input_tokens
    args.token_budget = extra.token_budget
    args.quest_chunk_size = extra.quest_chunk_size
    args.quest_budget_p = extra.quest_budget_p
    args.quest_bg_mode = extra.quest_bg_mode
    args.quest_bg_alpha = extra.quest_bg_alpha
    args.quest_bg_delta = extra.quest_bg_delta
    args.quest_bg_margin = extra.quest_bg_margin
    args.snapkv_obs_window = extra.snapkv_obs_window
    args.snapkv_pool_kernel = extra.snapkv_pool_kernel
    args.snapkv_budget_p = extra.snapkv_budget_p
    args.ra_budget_p = extra.ra_budget_p
    args.ra_sink = extra.ra_sink
    args.ra_local = extra.ra_local
    args.ra_n_queries = extra.ra_n_queries
    args.ra_knn = extra.ra_knn
    args.ra_nlist = extra.ra_nlist
    args.ra_nprobe = extra.ra_nprobe
    args.ra_nprobe_q = extra.ra_nprobe_q
    args.sts_budget_p = extra.sts_budget_p
    args.sts_draft_layer = extra.sts_draft_layer
    args.sts_calib_tokens = extra.sts_calib_tokens
    args.sts_calib_topk = extra.sts_calib_topk
    args.sts_draft_model = extra.sts_draft_model
    args.pariskv_budget_p = extra.pariskv_budget_p
    args.pariskv_subspaces = extra.pariskv_subspaces
    args.pariskv_rho = extra.pariskv_rho
    args.pariskv_beta = extra.pariskv_beta
    args.pariskv_rotation_seed = extra.pariskv_rotation_seed
    if args.jsonl_output is None:
        raise ValueError("--jsonl-output is required")
    if args.method in {"quest", "quest_channel", "quest_calibrated"}:
        min_budget = quest_min_feasible_budget(args.quest_chunk_size)
        if args.token_budget < min_budget:
            raise ValueError(
                f"{args.method} token_budget {args.token_budget} < page-granularity min "
                f"{min_budget} ({QUEST_MIN_PAGES} pages × chunk {args.quest_chunk_size})"
            )
    if args.method in {"snapkv", "snapkv_channel"}:
        snapkv_sink = int(str(args.sink).split(",")[0])
        snapkv_local = int(str(args.local_window).split(",")[0])
        min_budget = snapkv.min_feasible_budget(
            sink=snapkv_sink,
            obs_window=int(args.snapkv_obs_window),
            local_window=snapkv_local,
        )
        if args.token_budget < min_budget:
            raise ValueError(
                f"{args.method} token_budget {args.token_budget} < min feasible "
                f"sink+obs+local={min_budget} (sink={snapkv_sink}, "
                f"obs={args.snapkv_obs_window}, local={snapkv_local})"
            )
    if args.method == "retrieval_attention":
        min_budget = ra.min_feasible_budget()
        if args.token_budget < min_budget:
            raise ValueError(
                f"retrieval_attention token_budget {args.token_budget} < min feasible {min_budget}"
            )
    if args.method == "sts":
        min_budget = sts.min_feasible_budget()
        if args.token_budget < min_budget:
            raise ValueError(
                f"sts token_budget {args.token_budget} < min feasible {min_budget}"
            )
    if args.method == "pariskv":
        pariskv_sink = int(str(args.sink).split(",")[0])
        pariskv_local = int(str(args.local_window).split(",")[0])
        min_budget = pariskv.min_feasible_budget(sink=pariskv_sink, local_window=pariskv_local)
        if args.token_budget < min_budget:
            raise ValueError(
                f"pariskv token_budget {args.token_budget} < min feasible "
                f"sink+local+min_topk={min_budget} (sink={pariskv_sink}, local={pariskv_local})"
            )

    torch.manual_seed(args.seed)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tokenizer = e2e.AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model_kwargs: dict[str, Any] = {
        "local_files_only": args.local_files_only,
        "torch_dtype": dtype,
        "attn_implementation": args.attn_implementation,
        "device_map": "auto" if args.device_map == "auto" else {"": 0},
    }
    model = e2e.AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).eval()
    args._sts_draft_model = None
    if args.method == "sts" and args.sts_draft_model:
        draft_tokenizer = e2e.AutoTokenizer.from_pretrained(
            args.sts_draft_model, local_files_only=args.local_files_only
        )
        if int(draft_tokenizer.vocab_size) != int(tokenizer.vocab_size):
            raise ValueError(
                "sts real draft requires a same-family tokenizer: target vocab_size="
                f"{tokenizer.vocab_size} draft vocab_size={draft_tokenizer.vocab_size}"
            )
        draft_kwargs = dict(model_kwargs)
        draft_model = e2e.AutoModelForCausalLM.from_pretrained(
            args.sts_draft_model, **draft_kwargs
        ).eval()
        args._sts_draft_model = draft_model
    thinking = _thinking_chat_enabled(tokenizer)
    print(
        json.dumps(
            {
                "event": "start",
                "method": args.method,
                "suite": args.suite,
                "model": args.model,
                "chat_template": "native" if thinking else "none",
                "enable_thinking": False if thinking else None,
                "max_position_embeddings": int(getattr(model.config, "max_position_embeddings", 0) or 0),
                "ruler_tokens": int(args.ruler_tokens),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    report: dict[str, Any] = {
        "method": args.method,
        "model": args.model,
        "chat_template": "native" if thinking else "none",
        "enable_thinking": False if thinking else None,
        "max_position_embeddings": int(getattr(model.config, "max_position_embeddings", 0) or 0),
        "num_attention_heads": int(getattr(model.config, "num_attention_heads", 0) or 0),
        "num_key_value_heads": int(getattr(model.config, "num_key_value_heads", 0) or 0),
        "ruler_tokens": int(args.ruler_tokens),
    }
    if args.method in {"quest", "quest_channel"}:
        report["token_budget"] = int(args.token_budget)
        report["quest_chunk_size"] = int(args.quest_chunk_size)
        report["quest_first_two_layers_dense"] = True
        if args.quest_budget_p is not None:
            report["quest_budget_p"] = float(args.quest_budget_p)
            report["quest_budget_mode"] = "grid"
        if args.method == "quest_channel":
            report["denom_channel"] = True
            report["bg_mode"] = str(args.quest_bg_mode)
            report["bg_alpha"] = float(args.quest_bg_alpha)
            report["bg_delta"] = float(args.quest_bg_delta)
            report["bg_margin"] = float(args.quest_bg_margin)
            report["bg_estimator"] = str(args.quest_bg_mode)
            report["bg_source"] = "host_quest_page_repr"
            report["note"] = (
                "quest_calibrated = page true-max (oracle, not transferable). "
                "upper_bound is unsafe. ub_margin subtracts a constant (host-native). "
                "page_mean_key / ub_calibrated are conservative. "
                "alpha_true is the two-axis scan: bg = m + α(true−m) + δ (exact V). "
                "trunc_true is the −∞ control (not α=0)."
            )
    if args.method == "quest_calibrated":
        report.update(
            quest_cal.report_metadata(
                token_budget=int(args.token_budget),
                chunk_size=int(args.quest_chunk_size),
                budget_p=args.quest_budget_p,
            )
        )
    if args.method in {"snapkv", "snapkv_channel"}:
        report.update(
            snapkv.report_metadata(
                token_budget=int(args.token_budget),
                sink=int(str(args.sink).split(",")[0]),
                local_window=int(str(args.local_window).split(",")[0]),
                obs_window=int(args.snapkv_obs_window),
                pool_kernel=int(args.snapkv_pool_kernel),
                budget_p=args.snapkv_budget_p,
                use_denom_channel=(args.method == "snapkv_channel"),
            )
        )
    if args.method == "retrieval_attention":
        report.update(
            ra.report_metadata(
                token_budget=int(args.token_budget),
                sink=int(args.ra_sink),
                local_window=int(args.ra_local),
                budget_p=args.ra_budget_p,
                n_queries=int(args.ra_n_queries),
                knn=int(args.ra_knn),
                nlist=int(args.ra_nlist),
                nprobe=int(args.ra_nprobe),
                nprobe_q=int(args.ra_nprobe_q),
            )
        )
    if args.method == "sts" and args._sts_draft_model is not None:
        report.update(
            sts.report_metadata_real(
                token_budget=int(args.token_budget),
                budget_p=args.sts_budget_p,
                draft_model_path=str(args.sts_draft_model),
                n_draft_layers=int(args._sts_draft_model.config.num_hidden_layers),
                n_target_layers=int(model.config.num_hidden_layers),
                calib_tokens=int(args.sts_calib_tokens),
                calib_topk=int(args.sts_calib_topk),
            )
        )
    elif args.method == "sts":
        report.update(
            sts.report_metadata(
                token_budget=int(args.token_budget),
                budget_p=args.sts_budget_p,
                draft_layer=int(args.sts_draft_layer),
                calib_tokens=int(args.sts_calib_tokens),
                calib_topk=int(args.sts_calib_topk),
            )
        )
    if args.method == "pariskv":
        report.update(
            pariskv.report_metadata(
                token_budget=int(args.token_budget),
                budget_p=args.pariskv_budget_p,
                sink=int(str(args.sink).split(",")[0]),
                local_window=int(str(args.local_window).split(",")[0]),
                final_topk=int(getattr(args, "_last_pariskv_final_topk", 0)),
                n_subspaces=int(args.pariskv_subspaces),
                rho=float(args.pariskv_rho),
                beta=float(args.pariskv_beta),
            )
        )
    if args.suite in {"longbench", "both"}:
        report["longbench"] = run_longbench(model, tokenizer, args)
    if args.suite in {"ruler", "both"}:
        report["ruler"] = run_ruler(model, tokenizer, args)
    if args.suite == "infinitebench":
        report["infinitebench"] = run_infinitebench(model, tokenizer, args)
    if args.method == "sts" and getattr(args, "_last_sts_head_map_stats", None):
        report["sts_head_map_stats_last_sample"] = args._last_sts_head_map_stats
    if args.method == "sts" and getattr(args, "_last_sts_gen_stats", None):
        report["sts_gen_stats_last_sample"] = args._last_sts_gen_stats
    if args.method == "pariskv" and getattr(args, "_last_pariskv_stats", None):
        report["pariskv_stats_last_sample"] = args._last_pariskv_stats
    print(json.dumps({"event": "done", **report}, sort_keys=True), flush=True)
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
