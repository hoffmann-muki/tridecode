#!/usr/bin/env python3
"""Score tridecode HumanEval outputs with the OpenAI HumanEval evaluator.

The tridecode generation scripts write records with ``id`` and ``output``.
HumanEval expects ``task_id`` and ``completion``. This script performs that
conversion, runs the evaluator directly from a local HumanEval checkout, then
merges the pass/fail scores back into tridecode-style JSONL files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Iterable, List


def _convert_tridecode_to_humaneval(source_path: Path, target_path: Path) -> None:
    """Convert a tridecode JSONL file into HumanEval JSONL format."""

    with source_path.open("r", encoding="utf-8") as source_file, target_path.open(
        "w", encoding="utf-8"
    ) as target_file:
        for raw_line in source_file:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            record = json.loads(raw_line)
            target_file.write(
                json.dumps(
                    {
                        "task_id": record["id"],
                        "completion": record["output"],
                    }
                )
                + "\n"
            )


def _read_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            records.append(json.loads(raw_line))
    return records


def _run_humaneval_evaluator(human_eval_root: Path, sample_path: Path) -> Path:
    """Run HumanEval directly against a converted sample file."""

    subprocess.run(
        ["python", "human_eval/evaluate_functional_correctness.py", str(sample_path)],
        cwd=str(human_eval_root),
        check=True,
    )
    return Path(f"{sample_path}_results.jsonl")


def _merge_scores(raw_records: List[dict], result_path: Path) -> List[dict]:
    scores = [1 if row["passed"] else 0 for row in _read_jsonl(result_path)]
    if len(scores) != len(raw_records):
        raise ValueError(
            f"Score count mismatch: {len(scores)} scores for {len(raw_records)} records in {result_path}"
        )

    merged_records: List[dict] = []
    for record, score in zip(raw_records, scores):
        merged = dict(record)
        merged["score"] = score
        merged_records.append(merged)
    return merged_records


def _iter_jsonl_files(input_dir: Path) -> Iterable[Path]:
    return sorted(path for path in input_dir.glob("*.jsonl") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score tridecode HumanEval outputs with the HumanEval evaluator."
    )
    parser.add_argument(
        "--tridecode-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root directory containing out/, tmp_out/, and final_out/.",
    )
    parser.add_argument(
        "--human-eval-root",
        type=Path,
        default=Path.home() / "human-eval",
        help="Path to a local openai/human-eval checkout.",
    )
    parser.add_argument(
        "--model",
        default="MISTRAL",
        help="Model directory name under out/ and final_out/.",
    )
    parser.add_argument(
        "--decode-types",
        nargs="+",
        default=["origin", "tree"],
        help="Decode subdirectories to process.",
    )
    parser.add_argument(
        "--dataset",
        default="HUMAN_EVAL",
        help="Dataset directory name under the decode type.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output root. Defaults to tridecode/final_out.",
    )
    parser.add_argument(
        "--temp-root",
        type=Path,
        default=None,
        help="Optional temporary output root. Defaults to tridecode/tmp_out.",
    )
    args = parser.parse_args()

    tridecode_root = args.tridecode_root.resolve()
    human_eval_root = args.human_eval_root.resolve()
    output_root = (args.output_root or (tridecode_root / "final_out")).resolve()
    temp_root = (args.temp_root or (tridecode_root / "tmp_out")).resolve()

    for decode_type in args.decode_types:
        input_dir = tridecode_root / "out" / args.model / decode_type / args.dataset
        if not input_dir.exists():
            print(f"Skipping missing directory: {input_dir}")
            continue

        converted_dir = temp_root / args.model / decode_type / args.dataset
        scored_dir = output_root / args.model / decode_type / args.dataset
        converted_dir.mkdir(parents=True, exist_ok=True)
        scored_dir.mkdir(parents=True, exist_ok=True)

        for source_path in _iter_jsonl_files(input_dir):
            converted_path = converted_dir / source_path.name
            _convert_tridecode_to_humaneval(source_path, converted_path)

            result_path = _run_humaneval_evaluator(human_eval_root, converted_path)
            raw_records = _read_jsonl(source_path)
            merged_records = _merge_scores(raw_records, result_path)

            output_path = scored_dir / source_path.name
            with output_path.open("w", encoding="utf-8") as handle:
                for record in merged_records:
                    handle.write(json.dumps(record) + "\n")

            print(f"Wrote scored output to {output_path}")


if __name__ == "__main__":
    main()