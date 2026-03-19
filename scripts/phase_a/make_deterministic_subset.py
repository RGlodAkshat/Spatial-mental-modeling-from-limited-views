#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def load_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", default="data/raw/MindCube_train.jsonl")
    parser.add_argument("--tinybench-input", default="data/raw/MindCube_tinybench.jsonl")
    parser.add_argument("--train-size", type=int, default=50)
    parser.add_argument("--tinybench-size", type=int, default=50)
    parser.add_argument("--seed", default="phase_a_v1")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--train-output", default="data/raw/subsets/MindCube_train_50.jsonl")
    parser.add_argument("--tinybench-output", default="data/raw/subsets/MindCube_tinybench_50.jsonl")
    parser.add_argument("--manifest-output", default="data/manifests/phase_a_deterministic_50_50_manifest.json")
    args = parser.parse_args()

    train_items = load_jsonl(Path(args.train_input))
    tiny_items = load_jsonl(Path(args.tinybench_input))

    def subset(items, size):
        ranked = sorted(items, key=lambda x: hashlib.sha256(f"{args.seed}|{x.get(args.id_field,'')}".encode("utf-8")).hexdigest())
        return ranked[:size]

    train_sub = subset(train_items, args.train_size)
    tiny_sub = subset(tiny_items, args.tinybench_size)
    save_jsonl(train_sub, Path(args.train_output))
    save_jsonl(tiny_sub, Path(args.tinybench_output))

    manifest = {
        "seed": args.seed,
        "id_field": args.id_field,
        "train": {"input": args.train_input, "output": args.train_output, "actual_size": len(train_sub)},
        "tinybench": {"input": args.tinybench_input, "output": args.tinybench_output, "actual_size": len(tiny_sub)},
    }
    Path(args.manifest_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest_output).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
