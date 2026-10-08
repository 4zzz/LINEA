#!/usr/bin/env python3
"""Run infer_dataset.py using metadata recorded beside a training checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util.inference_cli import (
    add_argument, add_model_args, add_data_loading_args, add_inference_option_args,
    add_output_args, validate_output_args,
)


DEFAULT_PYTHON = (
    str(REPO_ROOT / 'venv314' / 'bin' / 'python')
    if (REPO_ROOT / 'venv314' / 'bin' / 'python').is_file()
    else sys.executable
)


class InferenceError(RuntimeError):
    pass


def load_effective_args(path: Path) -> dict[str, Any]:
    try:
        with path.open() as f:
            metadata = json.load(f)
    except FileNotFoundError as exc:
        raise InferenceError(f"Effective config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InferenceError(f"Invalid effective config {path}: {exc}") from exc

    args = metadata.get('args')
    if not isinstance(args, dict):
        raise InferenceError(f"Expected an 'args' object in {path}.")
    return args


def default_output_directory(checkpoint: Path, split: str) -> Path:
    return checkpoint.parent / 'inference' / f'{checkpoint.name}_{split}'


def build_command(
    *, checkpoint: Path, split: str, output_directory: Path, pred_threshold: float,
    device: str | None = None, batch_size: int | None = None,
    num_workers: int | None = None, max_samples: int | None = None,
    prediction_record: bool = False, single_prediction_record: Path | None = None,
    simple_json: bool = False, glb_model: bool = False, lines_2d_png: bool = False,
    prediction_record_save_exact_sample: bool = False,
    prediction_record_backend: str | None = None,
    prediction_record_include_codebase_diff: bool = False,
    prediction_record_save_matching: bool = False, prediction_record_matching_top_k: int = 5,
    prediction_record_save_full_matching_cost_matrix: bool = False,
    passthrough: Sequence[str] = (), python: str = DEFAULT_PYTHON,
) -> list[str]:
    command = [
        python, str(REPO_ROOT / 'tools' / 'infer_dataset.py'),
        '--checkpoint', str(checkpoint), '--split', split,
        '--output-dir', str(output_directory), '--pred-threshold', str(pred_threshold),
    ]
    if single_prediction_record is not None:
        single_prediction_record = Path(single_prediction_record)
        if not single_prediction_record.is_absolute():
            single_prediction_record = output_directory / single_prediction_record
    for option, value in (
        ('--device', device), ('--batch-size', batch_size), ('--num-workers', num_workers),
        ('--max-samples', max_samples), ('--single-prediction-record', single_prediction_record),
        ('--prediction-record-backend', prediction_record_backend),
        ('--prediction-record-matching-top-k', prediction_record_matching_top_k),
    ):
        if value is not None:
            command.extend([option, str(value)])
    for option, enabled in (
        ('--prediction-record', prediction_record), ('--simple-json', simple_json),
        ('--glb-model', glb_model), ('--lines-2d-png', lines_2d_png),
        ('--prediction-record-save-exact-sample', prediction_record_save_exact_sample),
        ('--prediction-record-save-matching', prediction_record_save_matching),
        ('--prediction-record-save-full-matching-cost-matrix', prediction_record_save_full_matching_cost_matrix),
    ):
        if enabled:
            command.append(option)
    if prediction_record_include_codebase_diff:
        command.append('--prediction-record-include-codebase-diff')
    command.extend(passthrough)
    return command


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Infer a checkpoint using its recorded effective configuration.',
    )
    add_model_args(parser)
    add_data_loading_args(parser)
    add_inference_option_args(parser)
    add_output_args(parser)
    parser.set_defaults(device=None, batch_size=None, num_workers=None)
    add_argument(parser, '--effective-config', type=Path, default=None)
    add_argument(parser, '--max-samples', type=int, default=None)
    add_argument(parser, '--python', default=DEFAULT_PYTHON)
    add_argument(parser, '--dry-run', action='store_true')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args, passthrough = parser.parse_known_args(argv)
    checkpoint = args.checkpoint.expanduser()
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint
    checkpoint = checkpoint.resolve()

    output_directory = args.output_dir or default_output_directory(checkpoint, args.split)
    output_directory = output_directory.expanduser()
    if not output_directory.is_absolute():
        output_directory = REPO_ROOT / output_directory
    args.output_dir = output_directory = output_directory.resolve()
    validate_output_args(parser, args)
    if args.max_samples is not None and args.max_samples < 0:
        parser.error('--max-samples must be nonnegative')
    if args.batch_size is not None and args.batch_size < 1:
        parser.error('--batch-size must be at least 1')
    if args.num_workers is not None and args.num_workers < 0:
        parser.error('--num-workers must be nonnegative')
    if not checkpoint.is_file():
        raise InferenceError(f"Checkpoint does not exist: {checkpoint}")

    effective_config = args.effective_config or checkpoint.parent / 'effective_config.json'
    effective_config = effective_config.expanduser()
    if not effective_config.is_absolute():
        effective_config = REPO_ROOT / effective_config
    effective_args = load_effective_args(effective_config.resolve())

    command = build_command(
        checkpoint=checkpoint, split=args.split, output_directory=output_directory,
        pred_threshold=args.pred_threshold, device=args.device, batch_size=args.batch_size,
        num_workers=args.num_workers, max_samples=args.max_samples,
        prediction_record=args.prediction_record, single_prediction_record=args.single_prediction_record,
        simple_json=args.simple_json, glb_model=args.glb_model, lines_2d_png=args.lines_2d_png,
        prediction_record_save_exact_sample=args.prediction_record_save_exact_sample,
        prediction_record_backend=args.prediction_record_backend,
        prediction_record_include_codebase_diff=args.prediction_record_include_codebase_diff,
        prediction_record_save_matching=args.prediction_record_save_matching,
        prediction_record_matching_top_k=args.prediction_record_matching_top_k,
        prediction_record_save_full_matching_cost_matrix=args.prediction_record_save_full_matching_cost_matrix,
        passthrough=passthrough, python=args.python,
    )

    print(f"Checkpoint: {checkpoint}")
    print(f"Effective config: {effective_config.resolve()}")
    print(f"LINEA3D: {effective_args.get('linea3d') is True}")
    print(f"Output: {output_directory}")
    print(f"Command: {shlex.join(command)}")

    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except InferenceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
