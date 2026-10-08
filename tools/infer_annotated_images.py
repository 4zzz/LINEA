#!/usr/bin/env python3
"""Run a Monolines3D checkpoint on annotation/image pairs in a directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.infer_dataset import run_inference
from util.inference_cli import (
    add_argument,
    add_data_loading_args,
    add_inference_option_args,
    add_model_args,
    add_output_args,
    validate_output_args,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Monolines3D checkpoint on recursively discovered image.jpg.json "
            "annotations and their adjacent images, preserving input paths in the output."
        ),
    )
    add_model_args(parser)
    add_data_loading_args(parser, include_split=False)
    add_inference_option_args(parser)
    add_output_args(parser)
    add_argument(parser, '--input-dir', required=True, type=Path)
    add_argument(parser, '--max-samples', type=int, default=None)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_output_args(parser, args)
    args.input_dir = args.input_dir.expanduser().resolve()
    if not args.input_dir.is_dir():
        parser.error('--input-dir must be an existing directory.')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive.')
    if args.num_workers < 0:
        parser.error('--num-workers must be non-negative.')
    if args.max_samples is not None and args.max_samples < 0:
        parser.error('--max-samples must be non-negative.')

    args.split = 'test'
    args.output_naming = 'images_structure'
    args.dataset_root = args.input_dir
    args.dataset_args_overrides = {
        'mono3d_dataset_root': str(args.input_dir),
        'mono3d_single_scene_root': True,
        'mono3d_scene_annotations_dir': '.',
        'mono3d_scene_images_dir': '.',
        # A directory cannot be a ZIP file, so always read the loose annotations.
        'mono3d_scene_annotations_file': '.',
        'mono3d_strict': False,
    }
    return args


def main(argv: Sequence[str] | None = None) -> None:
    run_inference(parse_args(argv), required_dataset_name='monolines3d')


if __name__ == '__main__':
    try:
        main()
    except ValueError as error:
        sys.exit(str(error))
