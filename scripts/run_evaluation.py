#!/usr/bin/env python3
"""MindCube Evaluation Script"""

import sys
import os
import argparse

current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.evaluation.evaluator import evaluate, auto_evaluate
from src.evaluation import quick_start_guide, batch_evaluate


def main():
    parser = argparse.ArgumentParser(description='MindCube Evaluation Framework')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', '-i', type=str)
    group.add_argument('--batch_dir', '-b', type=str)
    group.add_argument('--guide', '-g', action='store_true')
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument('--task', '-t', choices=['basic', 'cogmap', 'cognitive_map'], default='cogmap')
    task_group.add_argument('--auto', '-a', action='store_true')
    parser.add_argument('--output', '-o', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--quiet', '-q', action='store_true')
    args = parser.parse_args()

    if args.guide:
        quick_start_guide()
        return

    if args.batch_dir:
        batch_evaluate(args.batch_dir, args.output_dir)
        return

    if args.auto:
        results = auto_evaluate(args.input, args.output)
    else:
        results = evaluate(args.input, args.task, args.output, include_detailed_metrics=not (args.task in ['cogmap', 'cognitive_map'] and args.quick))

    if not args.quiet:
        accuracy = results['results']['gen_cogmap_accuracy'] * 100
        print(f"Accuracy: {accuracy:.1f}%")


if __name__ == "__main__":
    main()
