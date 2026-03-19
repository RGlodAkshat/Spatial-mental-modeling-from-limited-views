#!/usr/bin/env python3
"""MindCube SFT Data Conversion Script"""

import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.training.data_formatters import convert_prompts_to_sft_format, batch_convert_prompts_to_sft, list_supported_models


def get_default_sft_output_dir(model_type: str) -> str:
    return f"./data/prompts/training/{model_type}/"


def main():
    parser = argparse.ArgumentParser(description="Convert MindCube prompt data to model-specific SFT formats")
    parser.add_argument('--input', '-i')
    parser.add_argument('--output', '-o')
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--model', '-m', choices=['qwen2.5vl', 'llava', 'instructblip'])
    parser.add_argument('--list_models', action='store_true')
    parser.add_argument('--quiet', '-q', action='store_true')
    args = parser.parse_args()

    if args.list_models:
        for m in list_supported_models():
            print(m)
        return

    if args.input_dir:
        args.output_dir = args.output_dir or get_default_sft_output_dir(args.model)
        batch_convert_prompts_to_sft(args.input_dir, args.output_dir, args.model)
    else:
        if not args.output:
            from src.training.data_formatters import get_formatter
            formatter = get_formatter(args.model)
            base_name = os.path.basename(args.input)
            output_filename = formatter.get_output_filename(base_name)
            args.output = os.path.join(get_default_sft_output_dir(args.model), output_filename)
        convert_prompts_to_sft_format(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
