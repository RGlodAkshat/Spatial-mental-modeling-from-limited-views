#!/usr/bin/env python3
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


def parse_checkpoint_step(path: Path) -> int:
    m = re.search(r"checkpoint-(\d+)$", path.name)
    return int(m.group(1)) if m else 10**9


def discover_checkpoints(root: Path):
    if not root.exists():
        return []
    checkpoints = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    checkpoints.sort(key=parse_checkpoint_step)
    return checkpoints


def run_cmd(cmd):
    subprocess.run(cmd, check=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints for one task")
    parser.add_argument("--task", default="plain_cgmap_ffr_out")
    parser.add_argument("--checkpoints-root", default="checkpoints/sft/plain_cgmap_ffr_out")
    parser.add_argument("--input-file", default="data/prompts/general/MindCube_tinybench_plain_cgmap_ffr_out.jsonl")
    parser.add_argument("--results-dir", default="data/results/sft/plain_cgmap_ffr_out")
    parser.add_argument("--eval-dir", default="data/evaluate/sft/plain_cgmap_ffr_out")
    parser.add_argument("--model-type", default="qwen2.5vl")
    parser.add_argument("--python-bin", default=sys.executable)
    return parser


def main():
    args = build_parser().parse_args()
    checkpoints = discover_checkpoints(Path(args.checkpoints_root))
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.eval_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for ckpt in checkpoints:
        step = parse_checkpoint_step(ckpt)
        step_name = ckpt.name if step == 10**9 else f"checkpoint-{step}"
        infer_out = Path(args.results_dir) / f"MindCube_tinybench_{args.task}_{step_name}_responses.jsonl"
        eval_out = Path(args.eval_dir) / f"MindCube_tinybench_{args.task}_{step_name}_eval_results.json"
        run_cmd([args.python_bin, "scripts/run_inference.py", "--model-type", args.model_type, "--model-path", str(ckpt), "--input-file", args.input_file, "--output-file", str(infer_out)])
        run_cmd([args.python_bin, "scripts/run_evaluation.py", "-i", str(infer_out), "-o", str(eval_out)])
        rows.append({"checkpoint": str(ckpt), "step": str(step if step != 10**9 else "unknown"), "inference_output": str(infer_out), "evaluation_output": str(eval_out)})
    with (Path(args.eval_dir) / "best_checkpoint_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "step", "inference_output", "evaluation_output"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
