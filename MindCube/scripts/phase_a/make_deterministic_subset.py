#!/usr/bin/env python3
"""Create deterministic train/tinybench subsets and a manifest for Phase A smoke runs."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def stable_id_from_item(item: Dict[str, Any], id_field: str, index: int, name: str) -> str:
    if id_field not in item:
        raise ValueError(f"{name} row {index} is missing required id field '{id_field}'")

    raw_id = item[id_field]
    if raw_id is None:
        raise ValueError(f"{name} row {index} has null '{id_field}'")

    stable_id = str(raw_id)
    if not stable_id or (isinstance(raw_id, str) and not raw_id.strip()):
        raise ValueError(f"{name} row {index} has empty '{id_field}'")

    return stable_id


def validate_dataset(items: List[Dict[str, Any]], id_field: str, name: str) -> List[Tuple[int, str, Dict[str, Any]]]:
    stable_rows: List[Tuple[int, str, Dict[str, Any]]] = []
    seen: Dict[str, int] = {}
    duplicates: Dict[str, List[int]] = {}

    for index, item in enumerate(items):
        stable_id = stable_id_from_item(item, id_field, index, name)
        stable_rows.append((index, stable_id, item))
        if stable_id in seen:
            duplicates.setdefault(stable_id, [seen[stable_id]]).append(index)
        else:
            seen[stable_id] = index

    if duplicates:
        duplicate_lines = ", ".join(
            f"{item_id}: {positions}" for item_id, positions in sorted(duplicates.items())
        )
        raise ValueError(f"{name} has duplicate '{id_field}' values: {duplicate_lines}")

    return stable_rows


def build_ranked_records(
    stable_rows: List[Tuple[int, str, Dict[str, Any]]], seed: str
) -> List[Dict[str, Any]]:
    ranked_records: List[Dict[str, Any]] = []
    for input_index, stable_id, item in stable_rows:
        selection_hash = hashlib.sha256(f"{seed}|{stable_id}".encode("utf-8")).hexdigest()
        ranked_records.append(
            {
                "input_index": input_index,
                "id": stable_id,
                "selection_hash": selection_hash,
                "item": item,
            }
        )

    ranked_records.sort(key=lambda record: (record["selection_hash"], record["id"]))
    for selection_rank, record in enumerate(ranked_records, start=1):
        record["selection_rank"] = selection_rank

    return ranked_records


def select_subset(
    items: List[Dict[str, Any]], size: int, seed: str, id_field: str, name: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if size > len(items):
        raise ValueError(f"Requested subset size {size}, but {name} has only {len(items)} rows")

    stable_rows = validate_dataset(items, id_field, name)
    ranked_records = build_ranked_records(stable_rows, seed)
    selected_by_hash = ranked_records[:size]

    # Keep output order stable with respect to the source file for backward compatibility.
    selected_in_output_order = sorted(selected_by_hash, key=lambda record: record["input_index"])
    selected_items = [record["item"] for record in selected_in_output_order]

    manifest_rows: List[Dict[str, Any]] = []
    for output_order_index, record in enumerate(selected_in_output_order, start=1):
        manifest_rows.append(
            {
                "id": record["id"],
                "input_index": record["input_index"],
                "selection_rank": record["selection_rank"],
                "selection_hash": record["selection_hash"],
                "output_order_index": output_order_index,
            }
        )

    subset_metadata = {
        "selection_method": "sha256(seed|stable_id)",
        "seed": seed,
        "id_field": id_field,
        "rank_sort": ["selection_hash", "id"],
        "output_order": "preserve_input_order",
        "selected_ids_selection_order": [record["id"] for record in selected_by_hash],
        "selected_ids_output_order": [record["id"] for record in selected_in_output_order],
    }

    return selected_items, manifest_rows, subset_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic 50/50 subset generator for MindCube")
    parser.add_argument("--train-input", default="data/raw/MindCube_train.jsonl")
    parser.add_argument("--tinybench-input", default="data/raw/MindCube_tinybench.jsonl")
    parser.add_argument("--train-size", type=int, default=50)
    parser.add_argument("--tinybench-size", type=int, default=50)
    parser.add_argument("--seed", default="phase_a_v1")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--train-output", default="data/raw/subsets/MindCube_train_50.jsonl")
    parser.add_argument("--tinybench-output", default="data/raw/subsets/MindCube_tinybench_50.jsonl")
    parser.add_argument(
        "--manifest-output",
        default="data/manifests/phase_a_deterministic_50_50_manifest.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    train_input = Path(args.train_input)
    tiny_input = Path(args.tinybench_input)
    train_output = Path(args.train_output)
    tiny_output = Path(args.tinybench_output)
    manifest_output = Path(args.manifest_output)

    if not train_input.exists():
        raise FileNotFoundError(f"Missing train input: {train_input}")
    if not tiny_input.exists():
        raise FileNotFoundError(f"Missing tinybench input: {tiny_input}")

    train_items = load_jsonl(train_input)
    tiny_items = load_jsonl(tiny_input)

    selected_train, train_manifest_rows, train_metadata = select_subset(
        train_items, args.train_size, args.seed, args.id_field, "train subset"
    )
    selected_tiny, tiny_manifest_rows, tiny_metadata = select_subset(
        tiny_items, args.tinybench_size, args.seed, args.id_field, "tinybench subset"
    )

    save_jsonl(selected_train, train_output)
    save_jsonl(selected_tiny, tiny_output)

    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": args.seed,
        "id_field": args.id_field,
        "selection_method": "sha256(seed|stable_id)",
        "selection_sort": ["selection_hash", "id"],
        "output_order": "preserve_input_order",
        "train": {
            "input": str(train_input),
            "output": str(train_output),
            "requested_size": args.train_size,
            "actual_size": len(selected_train),
            "metadata": train_metadata,
            "selected": train_manifest_rows,
        },
        "tinybench": {
            "input": str(tiny_input),
            "output": str(tiny_output),
            "requested_size": args.tinybench_size,
            "actual_size": len(selected_tiny),
            "metadata": tiny_metadata,
            "selected": tiny_manifest_rows,
        },
    }

    with manifest_output.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("Deterministic subset generation complete")
    print(f"Train subset: {train_output} ({len(selected_train)} rows)")
    print(f"Tinybench subset: {tiny_output} ({len(selected_tiny)} rows)")
    print(f"Manifest: {manifest_output}")


if __name__ == "__main__":
    main()
