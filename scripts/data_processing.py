#!/usr/bin/env python3
"""
MindCube Scaffold Data Processing Script

Main entry point for MindCube scaffold data generation pipeline.
Supports cognitive map generation, reasoning chain creation, and full pipeline processing.

This script handles the SCAFFOLD GENERATION phase only.
For PROMPT GENERATION, use generate_prompts.py instead.
"""

import sys
import os
import argparse
from typing import cast

current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.scaffold_curation.processors import process_data, batch_process, TaskType as ScaffoldTaskType


def main():
    parser = argparse.ArgumentParser(description='MindCube Data Processing Pipeline')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', '-i', type=str, help='Input JSONL file')
    group.add_argument('--batch_dir', '-b', type=str, help='Directory containing input JSONL files for batch processing')
    parser.add_argument('--output', '-o', type=str, help='Output file path (for single file processing)')
    parser.add_argument('--output_dir', type=str, help='Output directory (for batch processing)')
    parser.add_argument('--task', '-t', choices=['cogmap', 'reasoning', 'full_pipeline'], default='full_pipeline')
    parser.add_argument('--reasoning-setting', choices=['rotation', 'translation', 'among', 'around'])
    parser.add_argument('--format', choices=['full', 'shortened', 'qwen'], default='full')
    parser.add_argument('--quiet', '-q', action='store_true')
    args = parser.parse_args()

    task_mapping = {'cogmap': 'cognitive_map', 'reasoning': 'reasoning', 'full_pipeline': 'full_pipeline'}
    internal_task_type = cast(ScaffoldTaskType, task_mapping.get(args.task, args.task))

    if args.batch_dir:
        if not args.output_dir:
            args.output_dir = f"{args.batch_dir}_processed"
        batch_process(args.batch_dir, args.output_dir, internal_task_type, args.format, "both", False, args.reasoning_setting)
    else:
        process_data(args.input, args.output, internal_task_type, args.format, "both", False, args.reasoning_setting)


if __name__ == "__main__":
    main()
