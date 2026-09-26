#!/usr/bin/env python3
"""Paired differences + task-stratified bootstrap 95% CIs for per-prompt
quality results (jsonl rows with ``suite``, ``task``, ``_id`` and ``score``, as
written by ``benchmarks/eval_task_utility.py`` / ``benchmarks/run_ablation_arm.py``).

Protocol: strict (task, _id) pairing (intersection only, drops reported per
comparison), two-stage macro average (per-task mean, then unweighted mean over
tasks), task-stratified bootstrap (resample prompts with replacement inside
each task, n_task fixed), 95% percentile CI. Pure CPU statistics; no model, no
GPU.

Default configuration (reproduces the background-ablation table): Llama-3.1-8B-
Instruct at 128K, RULER tasks niah_multiquery, niah_multivalue, cwe, fwe and
qa (50 prompts each; ``qa`` is the harness-native fact task, see
RULER_TASK_META["qa"]["note"] in eval_task_utility.py), variants

  dense  full attention
  A      truncation, IVF-probed selector (nprobe=128)
  B      PQ-HSA default (exact top-k + PQ-score background)
  C      global value-mean background, PQ-driven mass
  C2     global value-mean background, PQ-free mass

at budgets p in {1%, 2%}, read from ``<input-dir>/full_<variant>_p{1,2}.jsonl``
and ``<input-dir>/full_dense.jsonl``.

Other variants / files can be added with ``--method NAME=PATH[,PATH...]`` and
compared with ``--pair A,B`` (reported diff = A - B). The bootstrap stream of
each comparison is seeded with ``np.random.default_rng([seed,
crc32('<group>|<a> - <b>')])``; keep the default ``--group`` and ``--seed`` to
reproduce the reported intervals exactly.

Usage:
  python benchmarks/stats/paired_bootstrap_ci.py --input-dir results/quality
  python benchmarks/stats/paired_bootstrap_ci.py --input-dir results/quality \
      --method T_p1=results/quality/T_p1.jsonl --pair B_p1,T_p1 --pair T_p1,A_p1
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import zlib
from typing import Dict, List, Sequence, Tuple

import numpy as np

B_DEFAULT = 10000
SEED = 20260828
CI_LEVEL = 0.95
ROUND_NDIGITS = 6

TASKS5 = ["niah_multiquery", "niah_multivalue", "cwe", "fwe", "qa"]

# Seed-derivation label of the default comparison group (see module docstring).
DEFAULT_GROUP = "e21_full_8b_5task_128k"

METHOD_LABELS = {
    "dense": "dense (full attention)",
    "A_p1": "A truncation, p=1% (mode=sparse, nprobe=128)",
    "A_p2": "A truncation, p=2% (mode=sparse, nprobe=128)",
    "B_p1": "B PQ-HSA (default), p=1%",
    "B_p2": "B PQ-HSA (default), p=2%",
    "C_p1": "C uniform-value background, PQ-driven mass, p=1%",
    "C_p2": "C uniform-value background, PQ-driven mass, p=2%",
    "C2_p1": "C2 uniform-value + PQ-free mass, p=1%",
    "C2_p2": "C2 uniform-value + PQ-free mass, p=2%",
}

SOURCES = {
    "dense": ["full_dense.jsonl"],
    "A_p1": ["full_A_p1.jsonl"],
    "A_p2": ["full_A_p2.jsonl"],
    "B_p1": ["full_B_p1.jsonl"],
    "B_p2": ["full_B_p2.jsonl"],
    "C_p1": ["full_C_p1.jsonl"],
    "C_p2": ["full_C_p2.jsonl"],
    "C2_p1": ["full_C2_p1.jsonl"],
    "C2_p2": ["full_C2_p2.jsonl"],
}

# (a, b) -> reported diff is a - b. Core mechanism-decomposition triple per
# budget, plus B-vs-dense (absolute cost of the exact/background scheme
# relative to full attention).
PAIRS_PER_BUDGET = [
    ("B", "A"),   # background vs truncation
    ("B", "C"),   # PQ value-weighting vs uniform value (same PQ-driven mass)
    ("C", "C2"),  # PQ-calibrated mass vs PQ-free mass
    ("B", "C2"),  # full PQ vs fully PQ-free background
    ("B", "dense"),
    ("A", "dense"),
]


def load_scores(
    basenames: Sequence[str],
    input_dir: str = ".",
    tasks: Sequence[str] = TASKS5,
    suite: str = "ruler",
) -> Tuple[Dict[Tuple[str, str], float], dict]:
    """Load per-prompt scores keyed by (task, _id). Relative paths are resolved
    against ``input_dir``. A truncated trailing line (run still writing) is
    skipped and counted; any other malformed line raises."""
    scores: Dict[Tuple[str, str], float] = {}
    meta = {
        "files": [],
        "missing_files": [],
        "n_rows": 0,
        "n_duplicate_keys": 0,
        "n_unparsable_trailing_lines": 0,
    }
    for name in basenames:
        path = name if os.path.isabs(name) else os.path.join(input_dir, name)
        if not os.path.exists(path):
            meta["missing_files"].append(name)
            continue
        st = os.stat(path)
        lines = open(path, "r", encoding="utf-8").read().splitlines()
        n_rows_file = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                if i == len(lines) - 1:
                    meta["n_unparsable_trailing_lines"] += 1
                    continue
                raise
            if rec.get("suite") != suite or rec.get("task") not in tasks:
                continue
            key = (rec["task"], rec["_id"])
            n_rows_file += 1
            meta["n_rows"] += 1
            if key in scores:
                meta["n_duplicate_keys"] += 1
            scores[key] = float(rec["score"])
        meta["files"].append(
            {
                "name": name,
                "bytes": int(st.st_size),
                "mtime_utc": datetime.datetime.fromtimestamp(
                    st.st_mtime, datetime.timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "n_rows": n_rows_file,
            }
        )
    return scores, meta


def per_task_means(scores: Dict[Tuple[str, str], float], tasks: Sequence[str]) -> Dict[str, float]:
    out = {}
    for task in tasks:
        vals = [v for (t, _), v in scores.items() if t == task]
        if vals:
            out[task] = float(np.mean(vals))
    return out


def rng_for(group: str, label: str, seed: int) -> np.random.Generator:
    tag = zlib.crc32(f"{group}|{label}".encode("utf-8"))
    return np.random.default_rng([seed, tag])


def paired_compare(
    group: str,
    tasks: Sequence[str],
    a_name: str,
    a_scores: Dict[Tuple[str, str], float],
    b_name: str,
    b_scores: Dict[Tuple[str, str], float],
    n_boot: int,
    seed: int,
) -> dict:
    label = f"{a_name} - {b_name}"

    diffs: Dict[str, np.ndarray] = {}
    a_vals: Dict[str, np.ndarray] = {}
    b_vals: Dict[str, np.ndarray] = {}
    dropped = {"a_only": 0, "b_only": 0}
    per_task_n = {}

    for task in tasks:
        ids_a = {k[1] for k in a_scores if k[0] == task}
        ids_b = {k[1] for k in b_scores if k[0] == task}
        if not ids_a and not ids_b:
            continue
        common = sorted(ids_a & ids_b)
        dropped["a_only"] += len(ids_a - ids_b)
        dropped["b_only"] += len(ids_b - ids_a)
        if not common:
            per_task_n[task] = 0
            continue
        av = np.array([a_scores[(task, i)] for i in common], dtype=np.float64)
        bv = np.array([b_scores[(task, i)] for i in common], dtype=np.float64)
        a_vals[task] = av
        b_vals[task] = bv
        diffs[task] = av - bv
        per_task_n[task] = len(common)

    used_tasks = [t for t in tasks if t in diffs]
    if not used_tasks:
        return {
            "label": label,
            "diff_direction": f"{a_name} minus {b_name}",
            "status": "missing",
            "note": "no paired samples for any task",
        }

    gen = rng_for(group, label, seed)
    boot_task = np.empty((len(used_tasks), n_boot), dtype=np.float64)
    for row, task in enumerate(used_tasks):
        d = diffs[task]
        n = d.shape[0]
        idx = gen.integers(0, n, size=(n_boot, n))
        boot_task[row] = d[idx].mean(axis=1)
    boot_macro = boot_task.mean(axis=0)

    lo_q = 100.0 * (1.0 - CI_LEVEL) / 2.0
    hi_q = 100.0 - lo_q

    def ci(arr: np.ndarray) -> Tuple[float, float]:
        lo, hi = np.percentile(arr, [lo_q, hi_q])
        return float(lo), float(hi)

    macro_a = float(np.mean([a_vals[t].mean() for t in used_tasks]))
    macro_b = float(np.mean([b_vals[t].mean() for t in used_tasks]))
    macro_diff = float(np.mean([diffs[t].mean() for t in used_tasks]))
    lo, hi = ci(boot_macro)

    n_paired_total = int(sum(per_task_n.get(t, 0) for t in used_tasks))
    all_n = [per_task_n[t] for t in used_tasks]
    weights = np.array(all_n, dtype=np.float64)
    sample_weighted_diff = float(
        np.sum(weights * np.array([diffs[t].mean() for t in used_tasks])) / weights.sum()
    )

    per_task = {}
    for row, task in enumerate(used_tasks):
        t_lo, t_hi = ci(boot_task[row])
        per_task[task] = {
            "n_paired": int(per_task_n[task]),
            "mean_a": float(a_vals[task].mean()),
            "mean_b": float(b_vals[task].mean()),
            "paired_diff": float(diffs[task].mean()),
            "ci95_low": t_lo,
            "ci95_high": t_hi,
            "contains_zero": bool(t_lo <= 0.0 <= t_hi),
            "n_nonzero_diff": int(np.count_nonzero(diffs[task])),
        }

    return {
        "label": label,
        "diff_direction": f"{a_name} minus {b_name}",
        "status": "ok",
        "method_a": a_name,
        "method_b": b_name,
        "n_tasks_used": len(used_tasks),
        "tasks_used": used_tasks,
        "tasks_expected": list(tasks),
        "tasks_missing": [t for t in tasks if t not in diffs],
        "n_paired": n_paired_total,
        "n_dropped_a_only": int(dropped["a_only"]),
        "n_dropped_b_only": int(dropped["b_only"]),
        "n_per_task": {t: int(per_task_n[t]) for t in used_tasks},
        "macro_a": macro_a,
        "macro_b": macro_b,
        "macro_paired_diff": macro_diff,
        "macro_paired_diff_sample_weighted": sample_weighted_diff,
        "ci95_low": lo,
        "ci95_high": hi,
        "ci95_contains_zero": bool(lo <= 0.0 <= hi),
        "bootstrap_se": float(np.std(boot_macro, ddof=1)),
        "bootstrap_mean": float(np.mean(boot_macro)),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "per_task": per_task,
    }


def round_floats(obj, ndigits: int = ROUND_NDIGITS):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, ndigits) for v in obj]
    return obj


def _parse_method(spec: str) -> Tuple[str, List[str]]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--method expects NAME=PATH[,PATH...], got {spec!r}")
    name, paths = spec.split("=", 1)
    return name.strip(), [x for x in paths.split(",") if x.strip()]


def _parse_pair(spec: str) -> Tuple[str, str]:
    parts = [x.strip() for x in spec.split(",")]
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(f"--pair expects A,B, got {spec!r}")
    return parts[0], parts[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input-dir", default=os.path.join("results", "quality"),
                    help="directory holding the per-prompt jsonl files (default: results/quality)")
    ap.add_argument("--output", default=None,
                    help="output json (default: <input-dir>/paired_bootstrap_ci.json)")
    ap.add_argument("--n-boot", type=int, default=B_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--group", default=DEFAULT_GROUP,
                    help="comparison-group label; part of the per-comparison seed derivation")
    ap.add_argument("--tasks", default=",".join(TASKS5))
    ap.add_argument("--suite", default="ruler")
    ap.add_argument("--method", action="append", default=[], type=_parse_method,
                    help="add or override a variant: NAME=PATH[,PATH...] (repeatable)")
    ap.add_argument("--pair", action="append", default=[], type=_parse_pair,
                    help="comparison A,B reported as A - B (repeatable); default: the "
                         "B-A, B-C, C-C2, B-C2, B-dense, A-dense set at p1 and p2")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                    help="recorded in the output json only")
    args = ap.parse_args()

    tasks = [t for t in args.tasks.split(",") if t.strip()]
    output = args.output or os.path.join(args.input_dir, "paired_bootstrap_ci.json")
    sources: Dict[str, List[str]] = {k: list(v) for k, v in SOURCES.items()}
    for name, paths in args.method:
        sources[name] = paths

    group = args.group
    result = {
        "title": (
            f"{args.model}, RULER tasks ({'/'.join(tasks)}) -- same-selector "
            "background-treatment ablation, paired differences"
        ),
        "model": args.model,
        "tasks": tasks,
        "protocol": {
            "pairing": "strict on (task, sample_id); intersection only, drops reported per comparison",
            "macro_average": "two-stage: per-task mean, then unweighted mean over tasks",
            "bootstrap": (
                "task-stratified: each replicate resamples samples with replacement inside "
                "each task (n_task fixed), then recomputes the macro-average paired diff; "
                "per-task CIs come from the same replicate matrix"
            ),
            "ci": "percentile interval, level 0.95 (2.5/97.5)",
            "n_boot": args.n_boot,
            "seed": args.seed,
            "seed_derivation": "np.random.default_rng([seed, crc32('<group>|<a> - <b>')]) per comparison",
            "sign_convention": "every entry reports diff = method_a - method_b (see diff_direction)",
            "rounding": f"floats computed in float64, rounded to {ROUND_NDIGITS} decimals only when writing json",
            "gpu": "none (CPU-only, no torch/model)",
            "reference_seed_source": "eval_task_utility.py --seed 0",
        },
        "method_labels": {k: METHOD_LABELS.get(k, k) for k in sources},
        "groups": {},
        "missing": [],
    }

    loaded = {}
    source_meta = {}
    for method, files in sources.items():
        scores, meta = load_scores(files, args.input_dir, tasks, args.suite)
        source_meta[method] = meta
        if not scores:
            result["missing"].append(
                {"method": method, "expected_files": list(files), "reason": "file(s) not found or empty"}
            )
            continue
        loaded[method] = scores

    group_out = {
        "desc": result["title"],
        "tasks_expected": list(tasks),
        "methods_available": sorted(loaded),
        "methods_missing": sorted(set(sources) - set(loaded)),
        "sources": source_meta,
        "macro_average_per_method": {},
        "per_task_average_per_method": {},
        "comparisons": {},
    }

    for method, scores in loaded.items():
        ptm = per_task_means(scores, tasks)
        group_out["per_task_average_per_method"][method] = ptm
        group_out["macro_average_per_method"][method] = (
            float(np.mean(list(ptm.values()))) if ptm else None
        )

    pairs: List[Tuple[str, str]] = []
    if args.pair:
        for a_name, b_name in args.pair:
            if a_name in loaded and b_name in loaded and (a_name, b_name) not in pairs:
                pairs.append((a_name, b_name))
            elif (a_name, b_name) not in pairs:
                result["missing"].append(
                    {"pair": f"{a_name} - {b_name}", "reason": "one or both variants not loaded"}
                )
    else:
        for budget in ("p1", "p2"):
            for a_arm, b_arm in PAIRS_PER_BUDGET:
                a_name = "dense" if a_arm == "dense" else f"{a_arm}_{budget}"
                b_name = "dense" if b_arm == "dense" else f"{b_arm}_{budget}"
                if a_name in loaded and b_name in loaded and (a_name, b_name) not in pairs:
                    pairs.append((a_name, b_name))

    for a_name, b_name in pairs:
        entry = paired_compare(group, tasks, a_name, loaded[a_name], b_name, loaded[b_name], args.n_boot, args.seed)
        group_out["comparisons"][entry["label"]] = entry

    result["groups"][group] = group_out

    out_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(out_dir, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(round_floats(result), fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"=== {group}")
    print(f"  macro: {group_out['macro_average_per_method']}")
    for label, entry in group_out["comparisons"].items():
        if entry.get("status") != "ok":
            print(f"  {label}: {entry.get('status')} ({entry.get('note','')})")
            continue
        print(
            f"  {label}: diff={entry['macro_paired_diff']:+.4f} "
            f"CI[{entry['ci95_low']:+.4f},{entry['ci95_high']:+.4f}] "
            f"zero={'YES' if entry['ci95_contains_zero'] else 'no'} "
            f"n={entry['n_paired']} tasks={entry['n_tasks_used']} "
            f"drop(a/b)={entry['n_dropped_a_only']}/{entry['n_dropped_b_only']}"
        )
    if result["missing"]:
        print("=== missing")
        for m in result["missing"]:
            print(f"  {m}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
