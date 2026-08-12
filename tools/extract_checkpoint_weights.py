#!/usr/bin/env python3
"""Create a compact inference checkpoint from a LINEA training checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import torch


class CheckpointConversionError(RuntimeError):
    pass


def default_output_path(source: Path) -> Path:
    if source.suffix:
        return source.with_name(f"{source.stem}_weights{source.suffix}")
    return source.with_name(f"{source.name}_weights.pth")


def select_model_state(
    checkpoint: Mapping[str, Any],
    source: str = "auto",
) -> tuple[Mapping[str, Any], str]:
    selected_source = source
    if source == "auto":
        selected_source = "ema" if "ema" in checkpoint else "model"

    if selected_source == "ema":
        ema = checkpoint.get("ema")
        if not isinstance(ema, Mapping) or not isinstance(ema.get("module"), Mapping):
            raise CheckpointConversionError(
                "EMA weights requested, but checkpoint['ema']['module'] is missing."
            )
        return ema["module"], selected_source

    model = checkpoint.get("model")
    if not isinstance(model, Mapping):
        raise CheckpointConversionError("Checkpoint does not contain a model state dictionary.")
    return model, selected_source


def compact_checkpoint(
    checkpoint: Mapping[str, Any],
    source: str = "auto",
) -> tuple[dict[str, Any], str]:
    if "args" not in checkpoint:
        raise CheckpointConversionError(
            "Checkpoint does not contain 'args'; LINEA inference needs it to construct the model."
        )
    model_state, selected_source = select_model_state(checkpoint, source)
    return {"model": model_state, "args": checkpoint["args"]}, selected_source


def convert_checkpoint(
    source_path: Path,
    output_path: Path,
    *,
    source: str = "auto",
    overwrite: bool = False,
) -> str:
    source_path = source_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not source_path.is_file():
        raise CheckpointConversionError(f"Checkpoint does not exist: {source_path}")
    if source_path == output_path:
        raise CheckpointConversionError("Input and output paths must be different.")
    if output_path.exists() and not overwrite:
        raise CheckpointConversionError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )

    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(checkpoint, Mapping):
        raise CheckpointConversionError("Checkpoint root must be a dictionary.")
    compact, selected_source = compact_checkpoint(checkpoint, source)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent, delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
        torch.save(compact, temp_path)
        os.replace(temp_path, output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return selected_source


def format_size(size: int) -> str:
    return f"{size / 1024**2:.2f} MiB"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="training checkpoint to convert")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output path (default: CHECKPOINT_weights.EXT)",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "model", "ema"),
        default="auto",
        help="weights to extract; auto prefers EMA when available",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    source_path = args.checkpoint.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_output_path(source_path)
    )
    selected_source = convert_checkpoint(
        source_path,
        output_path,
        source=args.source,
        overwrite=args.overwrite,
    )
    print(f"Input:  {source_path} ({format_size(source_path.stat().st_size)})")
    print(f"Output: {output_path} ({format_size(output_path.stat().st_size)})")
    print(f"Weights: {selected_source}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CheckpointConversionError as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
