#!/usr/bin/env python3
"""Run inference + evaluation over all discovered checkpoints and rank them."""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def parse_checkpoint_step(path: Path) -> int:
    m = re.search(r"checkpoint-(\d+)$", path.name)
    return int(m.group(1)) if m else 10**9


def discover_checkpoints(root: Path) -> List[Path]:
    if not root.exists():
        return []
    checkpoints = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    checkpoints.sort(key=parse_checkpoint_step)
    return checkpoints


def run_cmd(cmd: List[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def read_accuracy(eval_json_path: Path) -> Optional[float]:
    if not eval_json_path.exists():
        return None
    payload = json.loads(eval_json_path.read_text(encoding="utf-8"))

    # Primary metric from MindCube evaluator.
    if isinstance(payload, dict):
        results = payload.get("results", {})
        if isinstance(results, dict) and "gen_cogmap_accuracy" in results:
            return float(results["gen_cogmap_accuracy"])
        if "accuracy" in payload:
            return float(payload["accuracy"])
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints for one task")
    parser.add_argument("--task", default="plain_cgmap_ffr_out")
    parser.add_argument("--checkpoints-root", default="checkpoints/sft/plain_cgmap_ffr_out")
    parser.add_argument(
        "--input-file",
        default="data/prompts/general/MindCube_tinybench_plain_cgmap_ffr_out.jsonl",
    )
    parser.add_argument("--results-dir", default="data/results/sft/plain_cgmap_ffr_out")
    parser.add_argument("--eval-dir", default="data/evaluate/sft/plain_cgmap_ffr_out")
    parser.add_argument("--model-type", default="qwen2.5vl")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--max-checkpoints", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    checkpoints_root = Path(args.checkpoints_root)
    input_file = Path(args.input_file)
    results_dir = Path(args.results_dir)
    eval_dir = Path(args.eval_dir)

    if not input_file.exists():
        raise FileNotFoundError(f"Missing tinybench prompt file: {input_file}")

    checkpoints = discover_checkpoints(checkpoints_root)
    if args.max_checkpoints > 0:
        checkpoints = checkpoints[: args.max_checkpoints]

    if not checkpoints:
        print(f"No checkpoints found at {checkpoints_root}")
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, str]] = []

    for ckpt in checkpoints:
        step = parse_checkpoint_step(ckpt)
        if step == 10**9:
            step_name = ckpt.name
        else:
            step_name = f"checkpoint-{step}"

        infer_out = results_dir / f"MindCube_tinybench_{args.task}_{step_name}_responses.jsonl"
        eval_out = eval_dir / f"MindCube_tinybench_{args.task}_{step_name}_eval_results.json"

        run_cmd(
            [
                args.python_bin,
                "scripts/run_inference.py",
                "--model-type",
                args.model_type,
                "--model-path",
                str(ckpt),
                "--input-file",
                str(input_file),
                "--output-file",
                str(infer_out),
            ]
        )

        run_cmd(
            [
                args.python_bin,
                "scripts/run_evaluation.py",
                "-i",
                str(infer_out),
                "-o",
                str(eval_out),
            ]
        )

        acc = read_accuracy(eval_out)
        summary_rows.append(
            {
                "checkpoint": str(ckpt),
                "step": str(step if step != 10**9 else "unknown"),
                "accuracy": "" if acc is None else f"{acc:.6f}",
                "inference_output": str(infer_out),
                "evaluation_output": str(eval_out),
            }
        )

    summary_rows.sort(
        key=lambda r: float(r["accuracy"]) if r["accuracy"] else -1.0,
        reverse=True,
    )

    summary_csv = eval_dir / "best_checkpoint_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "checkpoint",
                "step",
                "accuracy",
                "inference_output",
                "evaluation_output",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json = eval_dir / "best_checkpoint_summary.json"
    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    print(f"Summary written to {summary_csv}")
    if summary_rows and summary_rows[0]["accuracy"]:
        print(
            f"Best checkpoint: {summary_rows[0]['checkpoint']} with accuracy {summary_rows[0]['accuracy']}"
        )


if __name__ == "__main__":
    main()
