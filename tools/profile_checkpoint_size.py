#!/usr/bin/env python3
"""Report tensor storage used by model components and training checkpoint state."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


def iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from iter_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from iter_tensors(child)


def tensor_stats(value: Any) -> dict[str, int]:
    tensors = list(iter_tensors(value))
    return {
        "num_tensors": len(tensors),
        "numel": sum(tensor.numel() for tensor in tensors),
        "bytes": sum(tensor.numel() * tensor.element_size() for tensor in tensors),
    }


def state_dict_groups(state_dict: dict[str, Any], depth: int = 1) -> dict[str, dict[str, int]]:
    groups = defaultdict(lambda: {"num_tensors": 0, "numel": 0, "bytes": 0})
    for name, value in state_dict.items():
        parts = name.split(".")
        group_name = ".".join(parts[:depth])
        stats = tensor_stats(value)
        for key, amount in stats.items():
            groups[group_name][key] += amount
    return dict(groups)


def optimizer_state_fields(optimizer: dict[str, Any]) -> dict[str, dict[str, int]]:
    fields = defaultdict(lambda: {"num_tensors": 0, "numel": 0, "bytes": 0})
    for state in optimizer.get("state", {}).values():
        for name, value in state.items():
            stats = tensor_stats(value)
            for key, amount in stats.items():
                fields[name][key] += amount
    return dict(fields)


def profile_checkpoint(path: Path, component_depth: int = 1) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint root must be a dictionary.")

    sections = {name: tensor_stats(value) for name, value in checkpoint.items()}
    model_state = checkpoint.get("model", {})
    model_groups = state_dict_groups(model_state, component_depth) if isinstance(model_state, dict) else {}
    optimizer = checkpoint.get("optimizer", {})
    optimizer_fields = optimizer_state_fields(optimizer) if isinstance(optimizer, dict) else {}
    tensor_bytes = sum(section["bytes"] for section in sections.values())
    return {
        "path": str(path),
        "file_bytes": path.stat().st_size,
        "tensor_bytes": tensor_bytes,
        "serialization_overhead_bytes": path.stat().st_size - tensor_bytes,
        "sections": sections,
        "model_components": model_groups,
        "optimizer_state_fields": optimizer_fields,
    }


def format_bytes(size: int) -> str:
    return f"{size / 1_000_000:.2f} MB ({size / 1024**2:.2f} MiB)"


def print_stats(label: str, stats: dict[str, int], indent: str = "") -> None:
    print(
        f"{indent}{label:<28} {stats['numel'] / 1_000_000:>9.3f} M values  "
        f"{format_bytes(stats['bytes'])}"
    )


def print_profile(profile: dict[str, Any]) -> None:
    print(f"Checkpoint: {profile['path']}")
    print(f"File size:  {format_bytes(profile['file_bytes'])}")
    print(f"Tensor data: {format_bytes(profile['tensor_bytes'])}")
    print(f"Overhead:   {format_bytes(profile['serialization_overhead_bytes'])}")

    print("\nTop-level sections:")
    for name, stats in sorted(profile["sections"].items(), key=lambda item: -item[1]["bytes"]):
        print_stats(name, stats, "  ")

    if profile["model_components"]:
        print("\nModel components:")
        for name, stats in sorted(
            profile["model_components"].items(), key=lambda item: -item[1]["bytes"]
        ):
            print_stats(name, stats, "  ")

    if profile["optimizer_state_fields"]:
        print("\nOptimizer tensor fields:")
        for name, stats in sorted(
            profile["optimizer_state_fields"].items(), key=lambda item: -item[1]["bytes"]
        ):
            print_stats(name, stats, "  ")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument(
        "--component-depth",
        type=int,
        default=1,
        help="number of state-dict name segments used to group model tensors",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    if args.component_depth < 1:
        raise SystemExit("--component-depth must be at least 1")
    profiles = []
    for checkpoint in args.checkpoints:
        path = checkpoint.expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Checkpoint does not exist: {path}")
        profiles.append(profile_checkpoint(path, args.component_depth))

    if args.json:
        print(json.dumps(profiles, indent=2))
    else:
        for index, profile in enumerate(profiles):
            if index:
                print("\n" + "=" * 100 + "\n")
            print_profile(profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
