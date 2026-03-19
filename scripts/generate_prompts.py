#!/usr/bin/env python3
"""MindCube Prompt Generation Script"""

import sys
import os
import argparse
from typing import cast

current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.prompt_generation import generate_task_prompts, batch_generate_prompts
from src.prompt_generation.processors import validate_scaffold_data, quick_prompt_sample, generate_all_task_prompts, get_default_prompt_output_dir
from src.utils import ensure_dir
from src.prompt_generation.generators import list_task_types, TaskType


def main():
    parser = argparse.ArgumentParser(description='MindCube Prompt Generation Framework')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', '-i', type=str)
    group.add_argument('--batch_dir', '-b', type=str)
    group.add_argument('--list_tasks', action='store_true')
    parser.add_argument('--task', '-t', choices=['raw_qa','ff_rsn','aug_cgmap_in','aug_cgmap_out','plain_cgmap_out','plain_cgmap_ffr_out','aug_cgmap_ffr_out','cgmap_in_ffr_out'], default=None)
    parser.add_argument('--output', '-o', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--all_tasks', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--quiet', '-q', action='store_true')
    args = parser.parse_args()

    if args.list_tasks:
        for t in list_task_types():
            print(t)
        return

    if args.validate:
        print(validate_scaffold_data(args.input))
        return

    if args.preview:
        task_type = cast(TaskType, args.task) if args.task else None
        quick_prompt_sample(args.input, task_type, args.samples)
        return

    if args.batch_dir:
        output_dir = args.output_dir or f"{args.batch_dir}_prompts"
        if args.task:
            task_type = cast(TaskType, args.task)
            batch_generate_prompts(args.batch_dir, output_dir, [task_type], auto_detect=False)
        else:
            batch_generate_prompts(args.batch_dir, output_dir, auto_detect=True)
        return

    if args.all_tasks:
        generate_all_task_prompts(args.input, args.output_dir)
        return

    if not args.output:
        default_output_dir = get_default_prompt_output_dir()
        ensure_dir(default_output_dir)
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(default_output_dir, f"{base_name}_{args.task}.jsonl") if args.task else os.path.join(default_output_dir, f"{base_name}.jsonl")

    task_type = cast(TaskType, args.task) if args.task else None
    generate_task_prompts(args.input, args.output, task_type, auto_detect=(args.task is None))


if __name__ == "__main__":
    main()
