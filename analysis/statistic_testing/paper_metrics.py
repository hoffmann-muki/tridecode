#!/usr/bin/env python3
"""Compute the paper's beam-search efficiency metrics from final_out JSONL files.

The paper reports two core efficiency metrics:

* Memory efficiency per token:
  (max(memory_usage) - model_memory) / (input_len + output_len)
* Decoding speed in tokens per second:
  output_len / time_taken

For an origin/tree pair, this script also computes the multiplicative gains
used in the paper:

* memory_gain_x = origin_mem_per_token / tree_mem_per_token
* speed_gain_x = tree_tok_per_sec / origin_tok_per_sec

The script scans a final_out-style directory structure:

  {base_dir}/{model}/origin/{dataset}/{beam}_{samples}.jsonl
  {base_dir}/{model}/tree/{dataset}/{beam}_{samples}.jsonl

and writes one summary row per paired file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterable, List


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _safe_mean(values: Iterable[Any]) -> float:
    cleaned = [float(v) for v in values if _is_finite(v)]
    return fmean(cleaned) if cleaned else math.nan


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    if not _is_finite(numerator) or not _is_finite(denominator):
        return math.nan
    denominator = float(denominator)
    if denominator == 0:
        return math.nan
    return float(numerator) / denominator


def _normalize_dataset_name(value: str) -> str:
    cleaned = value.strip().upper().replace("-", "_").replace(" ", "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned


def _default_output_path(dataset: str | None, beam: int | None) -> Path:
    if dataset and beam is not None:
        return Path(f"analysis/results/paper_metrics_{dataset.lower()}_beam{beam}.csv")
    if dataset:
        return Path(f"analysis/results/paper_metrics_{dataset.lower()}.csv")
    if beam is not None:
        return Path(f"analysis/results/paper_metrics_beam{beam}.csv")
    return Path("analysis/results/paper_metrics_summary.csv")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sample_metrics(obj: Dict[str, Any]) -> Dict[str, float]:
    memory_usage = [float(v) for v in obj.get("memory_usage", []) if _is_finite(v)]
    peak_memory = max(memory_usage) if memory_usage else math.nan
    model_memory = float(obj.get("model_memory", math.nan))
    input_len = float(obj.get("input_len", math.nan))
    output_len = float(obj.get("output_len", math.nan))
    
    # Pure decoding time: time of last token - time of first token (excluding prefill)
    time_metric = [float(v) for v in obj.get("time_metric", []) if _is_finite(v)]
    if len(time_metric) > 1:
        time_taken = time_metric[-1] - time_metric[0]
    else:
        # Fallback to the overall time if fine-grained metrics are missing or single step
        time_taken = float(obj.get("time_taken", math.nan))
        
    input_kv_memory = float(obj.get("input_kv_memory", math.nan))
    score = obj.get("score", math.nan)

    total_tokens = input_len + output_len if _is_finite(input_len) and _is_finite(output_len) else math.nan
    peak_over_model = peak_memory - model_memory if _is_finite(peak_memory) and _is_finite(model_memory) else math.nan
    mem_per_token = _safe_ratio(peak_over_model, total_tokens)
    tok_per_sec = _safe_ratio(output_len, time_taken)

    return {
        "input_len": input_len,
        "output_len": output_len,
        "model_memory": model_memory,
        "input_kv_memory": input_kv_memory,
        "peak_memory": peak_memory,
        "peak_over_model": peak_over_model,
        "mem_per_token": mem_per_token,
        "time_taken": time_taken,
        "tok_per_sec": tok_per_sec,
        "score": float(score) if _is_finite(score) else math.nan,
    }


def summarize_file(path: Path) -> Dict[str, float]:
    return summarize_records(load_jsonl(path))


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, float]:
    metrics = [sample_metrics(obj) for obj in records]
    return {
        "count": float(len(metrics)),
        "mean_input_len": _safe_mean(r["input_len"] for r in metrics),
        "mean_output_len": _safe_mean(r["output_len"] for r in metrics),
        "mean_model_memory": _safe_mean(r["model_memory"] for r in metrics),
        "mean_input_kv_memory": _safe_mean(r["input_kv_memory"] for r in metrics),
        "mean_peak_memory": _safe_mean(r["peak_memory"] for r in metrics),
        "mean_peak_over_model": _safe_mean(r["peak_over_model"] for r in metrics),
        "mean_mem_per_token": _safe_mean(r["mem_per_token"] for r in metrics),
        "mean_time_taken": _safe_mean(r["time_taken"] for r in metrics),
        "mean_tok_per_sec": _safe_mean(r["tok_per_sec"] for r in metrics),
        "mean_score": _safe_mean(r["score"] for r in metrics),
    }


def parse_pair_metadata(origin_path: Path) -> Dict[str, Any]:
    try:
        model = origin_path.parents[2].name
        dataset = origin_path.parents[0].name
        beam_str, samples_str = origin_path.stem.split("_", 1)
        beam = int(beam_str)
        samples = int(samples_str)
    except (IndexError, ValueError):
        raise ValueError(f"Unrecognized final_out path layout: {origin_path}")

    tree_path = origin_path.parents[2] / "tree" / dataset / origin_path.name
    return {
        "model": model,
        "dataset": dataset,
        "beam": beam,
        "samples": samples,
        "tree_path": tree_path,
    }


def summarize_pair(origin_path: Path) -> Dict[str, Any]:
    meta = parse_pair_metadata(origin_path)
    tree_path = meta.pop("tree_path")

    if not tree_path.exists():
        raise FileNotFoundError(f"Missing tree file for {origin_path}: {tree_path}")

    origin_records = load_jsonl(origin_path)
    tree_records = load_jsonl(tree_path)

    if len(origin_records) != len(tree_records):
        raise ValueError(
            f"Paired files have different row counts: {origin_path} ({len(origin_records)}) vs {tree_path} ({len(tree_records)})"
        )

    origin_ids = [obj.get("id") for obj in origin_records]
    tree_ids = [obj.get("id") for obj in tree_records]
    if origin_ids != tree_ids:
        raise ValueError(f"Paired files have different sample ids: {origin_path} vs {tree_path}")

    origin_stats = summarize_records(origin_records)
    tree_stats = summarize_records(tree_records)

    memory_gain_x = _safe_ratio(origin_stats["mean_mem_per_token"], tree_stats["mean_mem_per_token"])
    speed_gain_x = _safe_ratio(tree_stats["mean_tok_per_sec"], origin_stats["mean_tok_per_sec"])
    peak_gain_x = _safe_ratio(origin_stats["mean_peak_over_model"], tree_stats["mean_peak_over_model"])

    result: Dict[str, Any] = {
        **meta,
        "origin_path": str(origin_path),
        "tree_path": str(tree_path),
        "memory_gain_x": memory_gain_x,
        "memory_savings_pct": (1.0 - _safe_ratio(tree_stats["mean_mem_per_token"], origin_stats["mean_mem_per_token"])) * 100.0
        if _is_finite(memory_gain_x) else math.nan,
        "speed_gain_x": speed_gain_x,
        "speedup_pct": (speed_gain_x - 1.0) * 100.0 if _is_finite(speed_gain_x) else math.nan,
        "peak_over_model_gain_x": peak_gain_x,
        "origin_count": origin_stats["count"],
        "tree_count": tree_stats["count"],
    }

    for prefix, stats in (("origin", origin_stats), ("tree", tree_stats)):
        for key, value in stats.items():
            result[f"{prefix}_{key}"] = value

    result["score_delta_mean"] = result["tree_mean_score"] - result["origin_mean_score"]
    return result


def iter_origin_files(base_dir: Path, dataset: str | None = None, beam: int | None = None) -> Iterable[Path]:
    dataset_filter = _normalize_dataset_name(dataset) if dataset else None
    for path in sorted(base_dir.glob("*/origin/*/*.jsonl")):
        if path.name.endswith("_results.jsonl"):
            continue
        try:
            meta = parse_pair_metadata(path)
        except ValueError:
            continue
        if dataset_filter and _normalize_dataset_name(meta["dataset"]) != dataset_filter:
            continue
        if beam is not None and meta["beam"] != beam:
            continue
        if meta["tree_path"].exists():
            yield path


def write_csv(rows: List[Dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_csv.write_text("")
        return

    fieldnames = [
        "model",
        "dataset",
        "beam",
        "samples",
        "origin_path",
        "tree_path",
        "memory_gain_x",
        "memory_savings_pct",
        "speed_gain_x",
        "speedup_pct",
        "peak_over_model_gain_x",
        "score_delta_mean",
        "origin_count",
        "tree_count",
        "origin_mean_input_len",
        "tree_mean_input_len",
        "origin_mean_output_len",
        "tree_mean_output_len",
        "origin_mean_model_memory",
        "tree_mean_model_memory",
        "origin_mean_input_kv_memory",
        "tree_mean_input_kv_memory",
        "origin_mean_peak_memory",
        "tree_mean_peak_memory",
        "origin_mean_peak_over_model",
        "tree_mean_peak_over_model",
        "origin_mean_mem_per_token",
        "tree_mean_mem_per_token",
        "origin_mean_time_taken",
        "tree_mean_time_taken",
        "origin_mean_tok_per_sec",
        "tree_mean_tok_per_sec",
        "origin_mean_score",
        "tree_mean_score",
    ]

    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize paper metrics from final_out origin/tree JSONL pairs."
    )
    parser.add_argument(
        "--base_dir",
        default="reproduction/final_out",
        help="Root directory containing {model}/origin/{dataset}/ and {model}/tree/{dataset}/.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset filter, e.g. HUMAN_EVAL, CNN, MATH500, GSM8K, or WMT.",
    )
    parser.add_argument(
        "--beam",
        type=int,
        default=None,
        help="Optional beam-width filter, e.g. 3 or 6.",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Destination CSV for paired summary metrics. Defaults to a filter-aware filename.",
    )
    parser.add_argument(
        "--include_single_beam",
        action="store_true",
        help="Also include beam=1 origin-only files as standalone rows.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    dataset = _normalize_dataset_name(args.dataset) if args.dataset else None
    output_csv = Path(args.output_csv) if args.output_csv else _default_output_path(dataset, args.beam)
    rows: List[Dict[str, Any]] = []

    for origin_path in iter_origin_files(base_dir, dataset=dataset, beam=args.beam):
        rows.append(summarize_pair(origin_path))

    if args.include_single_beam:
        for origin_path in sorted(base_dir.glob("*/origin/*/1_*.jsonl")):
            if origin_path.name.endswith("_results.jsonl"):
                continue
            try:
                meta = parse_pair_metadata(origin_path)
            except ValueError:
                continue
            origin_stats = summarize_file(origin_path)
            rows.append(
                {
                    **meta,
                    "origin_path": str(origin_path),
                    "tree_path": "",
                    "memory_gain_x": math.nan,
                    "memory_savings_pct": math.nan,
                    "speed_gain_x": math.nan,
                    "speedup_pct": math.nan,
                    "peak_over_model_gain_x": math.nan,
                    "score_delta_mean": math.nan,
                    "origin_count": origin_stats["count"],
                    "tree_count": math.nan,
                    **{f"origin_{k}": v for k, v in origin_stats.items()},
                }
            )

    rows.sort(
        key=lambda row: (
            row.get("model", ""),
            row.get("dataset", ""),
            int(row.get("beam", 0)),
            int(row.get("samples", 0)),
        )
    )
    write_csv(rows, output_csv)
    print(f"Saved {len(rows)} paired metric rows to {output_csv}")


if __name__ == "__main__":
    main()