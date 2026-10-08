#!/usr/bin/env python3
"""Run inference for images from a directory."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from functools import partial
import sys
from pathlib import Path
from typing import Iterator, Sequence
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util.inference_cli import (
    add_argument,
    add_data_loading_args,
    add_inference_option_args,
    add_model_args,
    add_output_args,
    validate_output_args,
)
from util.inference_output_helpers import (
    build_prediction_file_meta,
    create_inference_files
)
from util.slconfig import DictAction
from util.create_model import create_eval_model_from_checkpoint
from datasets.monolines3d import Monolines3D
from datasets import build_dataset, BatchImageCollateFunction
from models.linea.matcher import build_matcher
from models.linea.criterion import LINEACriterion

REPO_ROOT = Path(__file__).resolve().parents[1]

class InferenceImageCollateFunction(BatchImageCollateFunction):
    """Preserve ground truth in original-image pixels before batch padding."""

    def __call__(self, items):
        prepared_items = []
        for image, target in items:
            target = target.copy()
            if 'lines' in target:
                height, width = target['orig_size'].tolist()
                target['original_lines2d'] = target['lines'] * target['lines'].new_tensor(
                    [width, height, width, height]
                )
            prepared_items.append((image, target))
        return super().__call__(prepared_items)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a LINEA checkpoint on images from a directory.",
    )
    add_model_args(parser)
    add_data_loading_args(parser, include_split=True)
    add_inference_option_args(parser)
    add_output_args(parser)
    add_argument(parser, '--max-samples', type=int, default=None)
    add_argument(
        parser,
        "--dataset-arg",
        dest="dataset_args_overrides",
        nargs="+",
        action=DictAction,
        default={},
        metavar="KEY=VALUE",
        help=(
            "Override checkpoint arguments only for dataset creation. "
            "Multiple KEY=VALUE pairs may follow this option."
        ),
    )
    add_argument(
        parser,
        "--output-naming",
        choices=['seq_flat', 'images_structure'],
        default='seq_flat',
    )
    add_argument(
        parser,
        "--dataset-root",
        type=Path,
        default=None
    )

    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_output_args(parser, args)
    if (args.output_naming == 'images_structure' and args.dataset_root is None):
        parser.error("When --output-naming sets 'images_structure', --dataset-root must be specified")
    elif args.dataset_root is not None:
        args.dataset_root = args.dataset_root.resolve()
    if args.max_samples is not None and args.max_samples < 0:
        parser.error('--max_samples must be non-negative.')
    return args


def dataset_base_name(sample_idx, target, *, output_dir, output_naming='seq_flat', dataset_root=None):
    """Choose the output base path for a dataset sample."""
    if output_dir is None:
        return None
    output_dir = Path(output_dir)
    if output_naming == 'seq_flat':
        return output_dir / f'eval_{sample_idx:03}'
    if output_naming == 'images_structure':
        if 'image_path' not in target:
            raise ValueError('Image path is missing in targets')
        relative_path = Path(target['image_path']).relative_to(Path(dataset_root))
        return output_dir / (str(relative_path) + '_infer')
    raise ValueError('Unknown output naming value')


def run_inference(args: argparse.Namespace, *, required_dataset_name: str | None = None) -> None:
    """Run dataset inference with already parsed options and optional checkpoint validation."""

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    model, model_args, model_meta = create_eval_model_from_checkpoint(
        args.checkpoint,
        raw_outputs=True,
    )

    if required_dataset_name is not None and model_args.dataset_name != required_dataset_name:
        raise ValueError(
            f"This tool requires a {required_dataset_name} checkpoint; "
            f"the checkpoint uses dataset {model_args.dataset_name!r}."
        )

    dataset = build_dataset(
        image_set=args.split, 
        args=argparse.Namespace(**(vars(model_args) | args.dataset_args_overrides)), 
        write_dataset_info=False
    )

    collate_type = InferenceImageCollateFunction if args.lines_2d_png else BatchImageCollateFunction
    if model_args.eval_spatial_size is not None:
        collate_fn = collate_type(base_size=model_args.eval_spatial_size[0])
    else:
        collate_fn = collate_type()

    dataloader = DataLoader(
        dataset,
        args.batch_size,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    matcher = build_matcher(model_args)
    criterion = LINEACriterion(
        model_args.num_classes, 
        matcher=matcher, 
        weight_dict={},
        focal_alpha=model_args.focal_alpha, 
        losses=[], 
        line3d_alignment=getattr(model_args, 'line3d_alignment', 'xyz_shift'),
        line3d_loss_type=getattr(model_args, 'line3d_loss_type', 'l1'),
        line3d_smooth_l1_beta=getattr(model_args, 'line3d_smooth_l1_beta', 1.0),
        line3d_z_loss_type=getattr(model_args, 'line3d_z_loss_type', 'l1'),
    )

    model = model.to(device)
    model.eval()
    criterion.to(device)

    wants_records = args.prediction_record or args.single_prediction_record is not None
    file_meta = build_prediction_file_meta(args, model_args, model_meta, REPO_ROOT) if wants_records else None
    base_name_fn = partial(
        dataset_base_name, output_dir=args.output_dir,
        output_naming=args.output_naming, dataset_root=args.dataset_root,
    )
    batch_idx, sample_idx = 0, 0
    with torch.no_grad(), ExitStack() as stack:
        for samples, targets in dataloader:
            if args.max_samples is not None and sample_idx >= args.max_samples:
                break

            samples = samples.to(device)
            targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
            orig_target_sizes = torch.tensor(
                np.array([[tgt['orig_size'][1].item(), tgt['orig_size'][0].item()] for tgt in targets])
            ).to(device)

            raw_outputs, lines, scores = model(samples, orig_target_sizes, targets)
            matching_costs = matcher.compute_costs(raw_outputs, targets)
            indices = matcher.match_from_costs(matching_costs)

            alignments = criterion.loss_lines3d(
                raw_outputs, targets, indices,
                num_boxes=0, return_alignments=True,
            ) if 'pred_lines3d' in raw_outputs else None

            for idx in range(len(targets)):
                if args.max_samples is not None and sample_idx >= args.max_samples:
                    break
                create_inference_files(
                    idx, sample_idx, raw_outputs, lines, scores, targets,
                    matching_costs, indices, alignments,
                    checkpoint=args.checkpoint, device=str(device),
                    dataset_name=model_args.dataset_name,
                    line3d_alignment=getattr(model_args, 'line3d_alignment', 'xyz_shift'),
                    base_name_fn=base_name_fn, pred_threshold=args.pred_threshold,
                    simple_json=args.simple_json, glb_model=args.glb_model,
                    lines_2d_png=args.lines_2d_png, prediction_record=args.prediction_record,
                    single_prediction_record=args.single_prediction_record,
                    prediction_record_backend=args.prediction_record_backend,
                    prediction_record_save_exact_sample=args.prediction_record_save_exact_sample,
                    prediction_record_save_matching=args.prediction_record_save_matching,
                    prediction_record_matching_top_k=args.prediction_record_matching_top_k,
                    prediction_record_save_full_matching_cost_matrix=args.prediction_record_save_full_matching_cost_matrix,
                    split=args.split,
                    samples=samples, matcher=matcher, file_meta=file_meta,
                    stack=stack,
                )
                sample_idx += 1
            batch_idx += 1


def main(argv: Sequence[str] | None = None) -> None:
    run_inference(parse_args(argv))


if __name__ == "__main__":
    main()
