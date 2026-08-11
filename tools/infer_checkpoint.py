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
    *,
    checkpoint: Path,
    split: str,
    output_directory: Path,
    fit_affine: bool,
    dont_save_sample: bool,
    pred_threshold: float,
    device: str | None,
    batch_size: int | None,
    num_workers: int | None,
    max_samples: int | None,
    single_file: str | None,
    save_png_visualization: bool,
    passthrough: Sequence[str],
    python: str,
) -> list[str]:
    command = [
        python,
        str(REPO_ROOT / 'tools' / 'infer_dataset.py'),
        '--model',
        str(checkpoint),
        '--split',
        split,
        '--output_directory',
        str(output_directory),
        '--pred_threshold',
        str(pred_threshold),
    ]
    if dont_save_sample:
        command.append('--dont_save_sample')
    if fit_affine:
        command.append('--fit-affine')
    if device is not None:
        command.extend(['--device', device])
    if batch_size is not None:
        command.extend(['--batch_size', str(batch_size)])
    if num_workers is not None:
        command.extend(['--num_workers', str(num_workers)])
    if max_samples is not None:
        command.extend(['--max_samples', str(max_samples)])
    if single_file is not None:
        command.extend(['--single_file', single_file])
    if save_png_visualization:
        command.append('--save_png_visualization')
    command.extend(passthrough)
    return command


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Infer a checkpoint using its recorded effective configuration.',
    )
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='test')
    parser.add_argument('-o', '--output-directory', type=Path, default=None)
    parser.add_argument('--effective-config', type=Path, default=None)
    parser.add_argument('--pred-threshold', '--pred_threshold', type=float, default=0.0)
    parser.add_argument('--device', default=None)
    parser.add_argument('--batch-size', '--batch_size', type=int, default=None)
    parser.add_argument('--num-workers', '--num_workers', type=int, default=None)
    parser.add_argument('--max-samples', '--max_samples', type=int, default=None)
    parser.add_argument('--single-file', '--single_file', default=None)
    parser.add_argument('--save-png-visualization', action='store_true')
    parser.add_argument(
        '--fit-affine',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Override automatic affine fitting based on args.linea3d.',
    )
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument('-d', '--dont-save-sample', action='store_true', dest='dont_save_sample')
    sample_group.add_argument('--save-sample', action='store_false', dest='dont_save_sample')
    parser.set_defaults(dont_save_sample=True)
    parser.add_argument('--python', default=DEFAULT_PYTHON)
    parser.add_argument('--dry-run', action='store_true')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args, passthrough = parser.parse_known_args(argv)

    checkpoint = args.checkpoint.expanduser()
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise InferenceError(f"Checkpoint does not exist: {checkpoint}")

    effective_config = args.effective_config or checkpoint.parent / 'effective_config.json'
    if not effective_config.is_absolute():
        effective_config = REPO_ROOT / effective_config
    effective_args = load_effective_args(effective_config.resolve())
    is_linea3d = effective_args.get('linea3d') is True
    fit_affine = is_linea3d if args.fit_affine is None else args.fit_affine

    output_directory = args.output_directory or default_output_directory(checkpoint, args.split)
    if not output_directory.is_absolute():
        output_directory = REPO_ROOT / output_directory
    output_directory = output_directory.resolve()

    command = build_command(
        checkpoint=checkpoint,
        split=args.split,
        output_directory=output_directory,
        fit_affine=fit_affine,
        dont_save_sample=args.dont_save_sample,
        pred_threshold=args.pred_threshold,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        single_file=args.single_file,
        save_png_visualization=args.save_png_visualization,
        passthrough=passthrough,
        python=args.python,
    )

    print(f"Checkpoint: {checkpoint}")
    print(f"Effective config: {effective_config.resolve()}")
    print(f"LINEA3D: {is_linea3d}")
    print(f"Fit affine: {fit_affine}")
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
