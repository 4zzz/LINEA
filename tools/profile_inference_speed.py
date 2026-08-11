#!/usr/bin/env python3
"""Benchmark checkpoint inference latency by LINEA model component."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class BenchmarkError(RuntimeError):
    pass


def percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile of an empty sequence.")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def summarize_times(values_ms: Sequence[float]) -> dict[str, float]:
    if not values_ms:
        raise ValueError("No timings were recorded.")
    ordered = sorted(float(value) for value in values_ms)
    return {
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "p90_ms": percentile(ordered, 0.90),
        "p95_ms": percentile(ordered, 0.95),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "std_ms": statistics.pstdev(ordered),
    }


def parse_input_size(values: Sequence[int] | None, model_args) -> tuple[int, int]:
    if values:
        if len(values) == 1:
            return values[0], values[0]
        if len(values) == 2:
            return values[0], values[1]
        raise BenchmarkError("--input-size expects SIZE or HEIGHT WIDTH.")

    configured = getattr(model_args, "eval_spatial_size", 640)
    if isinstance(configured, int):
        return configured, configured
    if isinstance(configured, (list, tuple)) and len(configured) == 2:
        return int(configured[0]), int(configured[1])
    return 640, 640


def _create_model(model_args):
    from models.registry import MODULE_BUILD_FUNCS

    module_name = getattr(model_args, "modelname")
    if module_name not in MODULE_BUILD_FUNCS._module_dict:
        raise BenchmarkError(f"Unknown model module {module_name!r}.")
    return MODULE_BUILD_FUNCS.get(module_name)(model_args)


class ComponentTimer:
    def __init__(self, device: torch.device):
        self.device = device
        self.cuda = device.type == "cuda"
        self.starts = {}
        self.samples = {}
        self.handles = []

    def register(self, name, module):
        self.samples[name] = []

        def before(_module, _inputs):
            if self.cuda:
                marker = torch.cuda.Event(enable_timing=True)
                marker.record()
            else:
                marker = time.perf_counter()
            self.starts[name] = marker

        def after(_module, _inputs, _output):
            start = self.starts.pop(name)
            if self.cuda:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self.samples[name].append((start, end))
            else:
                self.samples[name].append((time.perf_counter() - start) * 1000.0)

        self.handles.append(module.register_forward_pre_hook(before))
        self.handles.append(module.register_forward_hook(after))

    def finish(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)
            return {
                name: [start.elapsed_time(end) for start, end in samples]
                for name, samples in self.samples.items()
            }
        return self.samples

    def close(self):
        for handle in self.handles:
            handle.remove()


def _timed_call(device, function):
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function()
        end.record()
        return result, (start, end)
    start = time.perf_counter()
    result = function()
    return result, (time.perf_counter() - start) * 1000.0


def _resolve_timing(device, timing):
    if device.type == "cuda":
        return timing[0].elapsed_time(timing[1])
    return timing


def benchmark(args) -> dict:
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise BenchmarkError(f"Checkpoint does not exist: {checkpoint_path}")
    if args.warmup < 0 or args.iterations < 1 or args.batch_size < 1:
        raise BenchmarkError("Warmup must be non-negative; iterations and batch size must be positive.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise BenchmarkError("CUDA was requested but is not available.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    model_args = checkpoint["args"]
    if not hasattr(model_args, "linea3d"):
        model_args.linea3d = False
    if not hasattr(model_args, "line3d_pred_strategy"):
        model_args.line3d_pred_strategy = "direct"
    if str(getattr(model_args, "backbone", "")).startswith("HGNetv2"):
        model_args.pretrained = False

    height, width = parse_input_size(args.input_size, model_args)
    model, postprocessor = _create_model(model_args)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    model.load_state_dict(state)
    model = model.deploy().to(device).eval()
    postprocessor = postprocessor.deploy().to(device).eval()

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    images = torch.rand(args.batch_size, 3, height, width, device=device)
    if args.channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    original_sizes = torch.tensor([[width, height]], device=device).repeat(args.batch_size, 1)
    camera_k = torch.tensor(
        [[max(height, width), 0.0, width / 2.0],
         [0.0, max(height, width), height / 2.0],
         [0.0, 0.0, 1.0]],
        device=device,
    )
    targets = [
        {
            "size": torch.tensor([height, width], device=device),
            "orig_size": torch.tensor([height, width], device=device),
            "camera_K": camera_k,
        }
        for _ in range(args.batch_size)
    ]

    amp_enabled = args.amp and device.type == "cuda"
    amp_context = lambda: torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled)

    def model_forward():
        with amp_context():
            return model(images, targets)

    def complete_forward():
        outputs = model_forward()
        with amp_context():
            return postprocessor(outputs, original_sizes)

    torch.backends.cudnn.benchmark = device.type == "cuda"
    with torch.inference_mode():
        for _ in range(args.warmup):
            complete_forward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        component_timer = ComponentTimer(device)
        component_timer.register("backbone", model.backbone)
        component_timer.register("encoder", model.encoder)
        component_timer.register("decoder", model.decoder)
        component_timer.register("postprocessor", postprocessor)
        total_timings = []
        for _ in range(args.iterations):
            _, timing = _timed_call(device, complete_forward)
            total_timings.append(timing)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        components = component_timer.finish()
        component_timer.close()
        total_ms = [_resolve_timing(device, timing) for timing in total_timings]

    total_summary = summarize_times(total_ms)
    result = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_version": torch.__version__,
        "batch_size": args.batch_size,
        "input_size": [height, width],
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "amp": amp_enabled,
        "channels_last": args.channels_last,
        "linea3d": bool(model_args.linea3d),
        "line3d_pred_strategy": model_args.line3d_pred_strategy,
        "total": total_summary,
        "throughput_images_per_second": args.batch_size * 1000.0 / total_summary["mean_ms"],
        "components": {name: summarize_times(values) for name, values in components.items()},
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }
    return result


def print_result(result: dict) -> None:
    print(f"Checkpoint: {result['checkpoint']}")
    print(f"Device: {result['device_name']} ({result['device']})")
    print(
        f"Input: batch={result['batch_size']}, size={result['input_size'][0]}x{result['input_size'][1]}, "
        f"AMP={result['amp']}, channels_last={result['channels_last']}"
    )
    print(
        f"Warmup/measured iterations: {result['warmup_iterations']}/{result['measured_iterations']}"
    )
    print("\nLatency (milliseconds):")
    print(f"  {'section':<16} {'mean':>9} {'median':>9} {'p90':>9} {'p95':>9} {'min':>9} {'max':>9}")
    sections = {"total": result["total"], **result["components"]}
    for name, stats in sections.items():
        print(
            f"  {name:<16} {stats['mean_ms']:>9.3f} {stats['median_ms']:>9.3f} "
            f"{stats['p90_ms']:>9.3f} {stats['p95_ms']:>9.3f} "
            f"{stats['min_ms']:>9.3f} {stats['max_ms']:>9.3f}"
        )
    print(f"\nThroughput: {result['throughput_images_per_second']:.2f} images/s")
    if result["peak_memory_bytes"] is not None:
        print(f"Peak allocated GPU memory: {result['peak_memory_bytes'] / 1024**2:.2f} MiB")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-size", nargs="+", type=int, default=None, metavar="SIZE")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    result = benchmark(args)
    print_result(result)
    if args.json_output is not None:
        output_path = args.json_output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Saved JSON report to {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
