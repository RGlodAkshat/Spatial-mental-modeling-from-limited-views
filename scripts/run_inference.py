#!/usr/bin/env python3
"""MindCube Inference Script"""

import argparse
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference.open_source import OpenSourceInferenceEngine
from inference.closed_source import ClosedSourceInferenceEngine


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference with MindCube models")
    parser.add_argument("--model-type", type=str, default="qwen2.5vl")
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--backend", type=str, default="transformers", choices=["transformers", "vllm"])
    parser.add_argument("--input-file", type=str, required=True)
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output-file", type=str)
    output_group.add_argument("--output-dir", type=str)
    parser.add_argument("--image-root", type=str, default="./data/")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--config", type=str)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    return parser


def generate_output_filename(input_file: str, model_type: str, model_path: str = None) -> str:
    input_path = Path(input_file)
    input_stem = input_path.stem
    if model_path and "/" in model_path:
        model_id = model_path.split("/")[-1].lower()
    elif model_path and os.path.exists(model_path):
        model_id = Path(model_path).name.lower()
    else:
        model_id = model_type.lower()
    return f"{input_stem}_{model_id}_responses.jsonl"


def create_inference_engine(args: argparse.Namespace):
    model_type = args.model_type.lower()
    if model_type in OpenSourceInferenceEngine.list_supported_models():
        if not args.model_path and model_type in ['qwen2.5vl', 'qwen', 'qwen2.5-vl']:
            args.model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
        kwargs = {
            'backend': args.backend,
            'max_new_tokens': args.max_new_tokens,
            'generation_config': {'temperature': args.temperature, 'top_p': args.top_p, 'do_sample': args.temperature > 0},
            'single_gpu': not args.multi_gpu,
        }
        return OpenSourceInferenceEngine.create_engine(model_type, args.model_path, **kwargs)
    if model_type in ClosedSourceInferenceEngine.list_supported_models():
        return ClosedSourceInferenceEngine(model_type)
    raise ValueError(f"Unsupported model type: {model_type}")


def main():
    parser = setup_parser()
    args = parser.parse_args()

    if args.list_models:
        print(OpenSourceInferenceEngine.list_supported_models())
        print(ClosedSourceInferenceEngine.list_supported_models())
        return

    if args.output_dir:
        temp_model_path = args.model_path or ("Qwen/Qwen2.5-VL-3B-Instruct" if args.model_type.lower() in ['qwen2.5vl', 'qwen', 'qwen2.5-vl'] else args.model_type)
        output_filename = generate_output_filename(args.input_file, args.model_type, temp_model_path)
        args.output_file = os.path.join(args.output_dir, output_filename)
        os.makedirs(args.output_dir, exist_ok=True)

    engine = create_inference_engine(args)
    engine.batch_infer(
        data_file=args.input_file,
        output_file=args.output_file,
        image_root=args.image_root,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
