#!/usr/bin/env python3
"""Safely continue a LINEA training run from its recorded output directory."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util.slconfig import SLConfig  # noqa: E402


STRUCTURAL_CONFIG_KEYS = {
    "backbone",
    "dec_layers",
    "dim_feedforward",
    "feat_channels_decoder",
    "feat_strides",
    "hidden_dim",
    "hybrid_encoder",
    "in_channels_encoder",
    "line3d_pred_strategy",
    "linea3d",
    "model_parameters",
    "modelname",
    "mogev2bb_base_model",
    "mogev2bb_neck_config_override",
    "mogev2bb_use_neck",
    "nheads",
    "num_classes",
    "num_feature_levels",
    "num_queries",
    "query_dim",
    "reg_max",
}

SAME_OUTPUT_ALLOWED_CHANGES = {"epochs", "output_dir"}
METADATA_FILENAMES = (
    "cmdline.json",
    "effective_config.json",
    "codebase.json",
    "machine.json",
)


class ContinuationError(RuntimeError):
    pass


def _parse_scalar(value: str) -> Any:
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    return value


def parse_overrides(items: list[str]) -> tuple[dict[str, Any], dict[str, str]]:
    parsed: dict[str, Any] = {}
    raw: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ContinuationError(f"Invalid override {item!r}; expected KEY=VALUE.")
        key, value = item.split("=", 1)
        if not key:
            raise ContinuationError(f"Invalid override {item!r}; key cannot be empty.")
        values = [_parse_scalar(part) for part in value.split(",")]
        parsed[key] = values[0] if len(values) == 1 else values
        raw[key] = value
    return parsed, raw


def recorded_overrides(source_output: Path) -> tuple[dict[str, Any], dict[str, str]]:
    cmdline_path = source_output / "cmdline.json"
    if not cmdline_path.is_file():
        return {}, {}
    argv = load_json(cmdline_path).get("argv", [])
    if not isinstance(argv, list) or "--options" not in argv:
        return {}, {}

    option_index = argv.index("--options") + 1
    items = []
    for value in argv[option_index:]:
        if not isinstance(value, str) or "=" not in value:
            break
        items.append(value)
    return parse_overrides(items)


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open() as f:
            value = json.load(f)
    except FileNotFoundError as exc:
        raise ContinuationError(f"Required metadata file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContinuationError(f"Invalid JSON metadata in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContinuationError(f"Expected a JSON object in {path}.")
    return value


def resolve_config(
    source_output: Path,
    saved_args: dict[str, Any],
    config_override: str | None,
) -> Path:
    configured = config_override or saved_args.get("config_file")
    if not configured:
        raise ContinuationError(
            "Could not determine the original config. Pass it explicitly with --config."
        )

    path = Path(configured).expanduser()
    candidates = [path] if path.is_absolute() else [REPO_ROOT / path]

    cmdline_path = source_output / "cmdline.json"
    if cmdline_path.exists() and not path.is_absolute():
        recorded_cwd = load_json(cmdline_path).get("cwd")
        if recorded_cwd:
            candidates.append(Path(recorded_cwd) / path)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ContinuationError(
        f"Config file {configured!r} does not exist. Pass its current path with --config."
    )


def resolve_checkpoint(source_output: Path, requested: str | None) -> Path:
    if requested and requested != "latest":
        path = Path(requested).expanduser()
        candidates = [path] if path.is_absolute() else [source_output / path, REPO_ROOT / path]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise ContinuationError(f"Requested checkpoint does not exist: {requested}")

    rolling = source_output / "checkpoint.pth"
    if rolling.is_file():
        return rolling.resolve()

    numbered = sorted(
        source_output.glob("checkpoint[0-9][0-9][0-9][0-9].pth"),
        key=lambda path: int(path.stem.removeprefix("checkpoint")),
    )
    if numbered:
        return numbered[-1].resolve()
    raise ContinuationError(
        f"No checkpoint found in {source_output}. Expected checkpoint.pth or "
        "checkpointNNNN.pth. Runs made with --no_save_checkpoints cannot be continued."
    )


def read_last_logged_epoch(source_output: Path) -> int | None:
    log_path = source_output / "log.txt"
    if not log_path.is_file():
        return None
    last_epoch = None
    with log_path.open() as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            epoch = record.get("epoch")
            if isinstance(epoch, int):
                last_epoch = epoch
    return last_epoch


def checkpoint_epoch_hint(checkpoint: Path, last_logged_epoch: int | None) -> int | None:
    suffix = checkpoint.stem.removeprefix("checkpoint")
    if suffix.isdigit():
        return int(suffix)
    if checkpoint.name == "checkpoint.pth":
        return last_logged_epoch
    return None


def load_merged_config(config: Path, overrides: dict[str, Any]) -> dict[str, Any]:
    cfg = SLConfig.fromfile(str(config))
    if overrides:
        cfg.merge_from_dict(overrides)
    return cfg._cfg_dict.to_dict()


def normalize_config_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize_config_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_config_value(item) for item in value]
    return value


def changed_config_values(
    saved_args: dict[str, Any],
    merged_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    changes = {}
    for key, current in merged_config.items():
        previous = saved_args.get(key, "<missing>")
        if normalize_config_value(previous) != normalize_config_value(current):
            changes[key] = {"previous": previous, "current": current}
    return changes


def ensure_destination(source_output: Path, destination: Path) -> None:
    if destination == source_output:
        return
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise ContinuationError(
                f"New output directory is not empty: {destination}. "
                "Choose a new directory to avoid mixing experiments."
            )


def validate_continuation(
    *,
    source_output: Path,
    destination: Path,
    checkpoint: Path,
    saved_args: dict[str, Any],
    merged_config: dict[str, Any],
    changes: dict[str, dict[str, Any]],
    runtime_changes: dict[str, dict[str, Any]],
    last_logged_epoch: int | None,
) -> None:
    if merged_config.get("use_warmup", False):
        raise ContinuationError(
            "Safe continuation with use_warmup=True is not supported because main.py "
            "does not currently restore the saved warmup-scheduler state. Resume after "
            "setting use_warmup=false in a new output directory, or add warmup restoration first."
        )

    structural_changes = sorted(STRUCTURAL_CONFIG_KEYS.intersection(changes))
    if structural_changes:
        raise ContinuationError(
            "True resume cannot safely change model/optimizer topology: "
            + ", ".join(structural_changes)
            + ". A future --weights-only mode should be used for these changes."
        )

    same_output = destination == source_output
    disallowed_same_output = sorted(set(changes) - SAME_OUTPUT_ALLOWED_CHANGES)
    disallowed_same_output.extend(sorted(runtime_changes))
    if same_output and disallowed_same_output:
        raise ContinuationError(
            "Reusing the original output directory is only allowed for an epoch extension. "
            "These settings changed: "
            + ", ".join(disallowed_same_output)
            + ". Pass --output-dir with a new directory."
        )

    old_epochs = saved_args.get("epochs")
    new_epochs = merged_config.get("epochs")
    if not isinstance(new_epochs, int):
        raise ContinuationError("The merged config must contain an integer epochs value.")
    if same_output and isinstance(old_epochs, int) and new_epochs < old_epochs:
        raise ContinuationError(
            f"Cannot reduce epochs from {old_epochs} to {new_epochs} in the same output directory."
        )

    checkpoint_epoch = checkpoint_epoch_hint(checkpoint, last_logged_epoch)
    if checkpoint_epoch is not None and new_epochs <= checkpoint_epoch:
        raise ContinuationError(
            f"Target epochs={new_epochs} would not run after checkpoint epoch "
            f"{checkpoint_epoch}. Increase --epochs."
        )
    if (
        same_output
        and last_logged_epoch is not None
        and checkpoint_epoch is not None
        and checkpoint_epoch < last_logged_epoch
    ):
        raise ContinuationError(
            f"Checkpoint epoch {checkpoint_epoch} is older than the last logged epoch "
            f"{last_logged_epoch}. Use a new --output-dir when branching from an older checkpoint."
        )


def runtime_arguments(args: argparse.Namespace, saved_args: dict[str, Any]) -> list[str]:
    command = [
        "--device",
        str(args.device or saved_args.get("device", "cuda")),
        "--seed",
        str(args.seed if args.seed is not None else saved_args.get("seed", 42)),
        "--num_workers",
        str(
            args.num_workers
            if args.num_workers is not None
            else saved_args.get("num_workers", 10)
        ),
        "--print_freq",
        str(
            args.print_freq
            if args.print_freq is not None
            else saved_args.get("print_freq", 500)
        ),
    ]

    prefetch = (
        args.prefetch_factor
        if args.prefetch_factor is not None
        else saved_args.get("prefetch_factor")
    )
    if prefetch is not None:
        command.extend(["--prefetch_factor", str(prefetch)])

    amp = args.amp if args.amp is not None else bool(saved_args.get("amp", False))
    no_save = (
        args.no_save_checkpoints
        if args.no_save_checkpoints is not None
        else bool(saved_args.get("no_save_checkpoints", False))
    )
    no_eval = (
        args.no_eval_during_train
        if args.no_eval_during_train is not None
        else bool(saved_args.get("no_eval_during_train", False))
    )
    find_unused = (
        args.find_unused_params
        if args.find_unused_params is not None
        else bool(saved_args.get("find_unused_params", False))
    )

    if amp:
        command.append("--amp")
    if no_save:
        command.append("--no_save_checkpoints")
    if no_eval:
        command.append("--no_eval_during_train")
    if find_unused:
        command.append("--find_unused_params")
    return command


def changed_runtime_values(
    args: argparse.Namespace,
    saved_args: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    requested = {
        "amp": args.amp,
        "device": args.device,
        "seed": args.seed,
    }
    changes = {}
    for key, current in requested.items():
        if current is not None and current != saved_args.get(key):
            changes[key] = {"previous": saved_args.get(key), "current": current}

    previous_world_size = saved_args.get("world_size", 1)
    if args.nproc_per_node != previous_world_size:
        changes["nproc_per_node"] = {
            "previous": previous_world_size,
            "current": args.nproc_per_node,
        }
    return changes


def build_command(
    *,
    args: argparse.Namespace,
    config: Path,
    checkpoint: Path,
    raw_overrides: dict[str, str],
    saved_args: dict[str, Any],
) -> list[str]:
    torchrun = Path(args.torchrun).expanduser()
    if not torchrun.is_absolute():
        repo_candidate = REPO_ROOT / torchrun
        executable = str(repo_candidate if repo_candidate.exists() else torchrun)
    else:
        executable = str(torchrun)

    command = [
        executable,
        f"--master_port={args.master_port}",
        f"--nproc_per_node={args.nproc_per_node}",
        "main.py",
        "-c",
        str(config),
        "--resume",
        str(checkpoint),
    ]
    command.extend(runtime_arguments(args, saved_args))
    if raw_overrides:
        command.append("--options")
        command.extend(f"{key}={value}" for key, value in raw_overrides.items())
    return command


def archive_metadata(source_output: Path, continuation_dir: Path) -> None:
    metadata_dir = continuation_dir / "source_metadata"
    metadata_dir.mkdir(parents=True)
    for filename in METADATA_FILENAMES:
        source = source_output / filename
        if source.is_file():
            shutil.copy2(source, metadata_dir / filename)


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Continue a LINEA experiment using its recorded config and latest checkpoint. "
            "Only an epoch extension may reuse the source output directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # Extend an experiment to 300 total epochs in the same directory.
  %(prog)s output/my_run --epochs 300

  # Fork from a selected checkpoint with changed loss settings.
  %(prog)s output/my_run --checkpoint checkpoint0049.pth \\
      --output-dir output/my_run_weight_05 --epochs 300 \\
      --set line3d_loss_weight_end=0.5

  # Validate and print the command without creating files or training.
  %(prog)s output/my_run --epochs 300 --dry-run
""",
    )
    result.add_argument("experiment_output", help="Existing experiment output directory.")
    result.add_argument(
        "--checkpoint",
        help="Checkpoint path or filename relative to the experiment directory; default: latest.",
    )
    result.add_argument("--config", help="Override the recorded config file path.")
    result.add_argument("--epochs", type=int, help="Total target epoch count.")
    result.add_argument(
        "--set",
        dest="settings",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Config override; repeat for multiple values.",
    )
    result.add_argument(
        "--output-dir",
        help="New output directory, required for training-semantic config changes.",
    )
    result.add_argument("--dry-run", action="store_true", help="Validate and print without running.")
    result.add_argument("--master-port", type=int, default=7777)
    result.add_argument("--nproc-per-node", type=int, default=1)
    result.add_argument("--torchrun", default="venv314/bin/torchrun")
    result.add_argument("--device")
    result.add_argument("--seed", type=int)
    result.add_argument("--num-workers", type=int)
    result.add_argument("--prefetch-factor", type=int)
    result.add_argument("--print-freq", type=int)
    result.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the original AMP setting.",
    )
    checkpoint_group = result.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--save-checkpoints",
        dest="no_save_checkpoints",
        action="store_false",
        help="Save checkpoints in the continued run.",
    )
    checkpoint_group.add_argument(
        "--no-save-checkpoints",
        dest="no_save_checkpoints",
        action="store_true",
        help="Do not save checkpoints in the continued run.",
    )
    result.set_defaults(no_save_checkpoints=None)

    evaluation_group = result.add_mutually_exclusive_group()
    evaluation_group.add_argument(
        "--eval-during-train",
        dest="no_eval_during_train",
        action="store_false",
        help="Run validation during continued training.",
    )
    evaluation_group.add_argument(
        "--no-eval-during-train",
        dest="no_eval_during_train",
        action="store_true",
        help="Skip validation during continued training.",
    )
    result.set_defaults(no_eval_during_train=None)
    result.add_argument(
        "--find-unused-params",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return result


def run(cli_args: list[str] | None = None) -> int:
    args = parser().parse_args(cli_args)
    source_output = Path(args.experiment_output).expanduser().resolve()
    if not source_output.is_dir():
        raise ContinuationError(f"Experiment output directory does not exist: {source_output}")

    effective = load_json(source_output / "effective_config.json")
    saved_args = effective.get("args")
    if not isinstance(saved_args, dict):
        raise ContinuationError(
            f"Missing args object in {source_output / 'effective_config.json'}."
        )

    recorded_parsed, recorded_raw = recorded_overrides(source_output)
    requested_parsed, requested_raw = parse_overrides(args.settings)
    if "output_dir" in requested_parsed:
        raise ContinuationError("Use --output-dir instead of --set output_dir=...")
    parsed_overrides = {**recorded_parsed, **requested_parsed}
    raw_overrides = {**recorded_raw, **requested_raw}
    if args.epochs is not None:
        parsed_overrides["epochs"] = args.epochs
        raw_overrides["epochs"] = str(args.epochs)

    destination = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else source_output
    )
    if destination != source_output:
        parsed_overrides["output_dir"] = str(destination)
        raw_overrides["output_dir"] = str(destination)

    config = resolve_config(source_output, saved_args, args.config)
    checkpoint = resolve_checkpoint(source_output, args.checkpoint)
    merged_config = load_merged_config(config, parsed_overrides)
    changes = changed_config_values(saved_args, merged_config)
    runtime_changes = changed_runtime_values(args, saved_args)
    last_logged_epoch = read_last_logged_epoch(source_output)

    ensure_destination(source_output, destination)
    validate_continuation(
        source_output=source_output,
        destination=destination,
        checkpoint=checkpoint,
        saved_args=saved_args,
        merged_config=merged_config,
        changes=changes,
        runtime_changes=runtime_changes,
        last_logged_epoch=last_logged_epoch,
    )

    command = build_command(
        args=args,
        config=config,
        checkpoint=checkpoint,
        raw_overrides=raw_overrides,
        saved_args=saved_args,
    )

    print(f"Source output: {source_output}")
    print(f"Checkpoint:    {checkpoint}")
    print(f"Config:        {config}")
    print(f"Destination:   {destination}")
    if changes:
        print("Config changes:")
        for key, values in sorted(changes.items()):
            print(f"  {key}: {values['previous']!r} -> {values['current']!r}")
    else:
        print("Config changes: none")
    if runtime_changes:
        print("Runtime changes:")
        for key, values in sorted(runtime_changes.items()):
            print(f"  {key}: {values['previous']!r} -> {values['current']!r}")
    print("Command:")
    print(f"  {shlex.join(command)}")

    if args.dry_run:
        return 0

    destination.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f%z")
    continuation_dir = destination / "continuations" / timestamp
    continuation_dir.mkdir(parents=True)
    archive_metadata(source_output, continuation_dir)

    provenance_path = continuation_dir / "continuation.json"
    provenance = {
        "status": "running",
        "started_at": dt.datetime.now().astimezone().isoformat(),
        "source_output": str(source_output),
        "destination": str(destination),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch_hint": checkpoint_epoch_hint(checkpoint, last_logged_epoch),
        "config": str(config),
        "config_changes": changes,
        "runtime_changes": runtime_changes,
        "command": command,
        "cwd": str(REPO_ROOT),
    }
    write_json(provenance_path, provenance)

    try:
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    except OSError as exc:
        provenance["status"] = "launch_failed"
        provenance["error"] = str(exc)
        provenance["finished_at"] = dt.datetime.now().astimezone().isoformat()
        write_json(provenance_path, provenance)
        raise ContinuationError(f"Could not launch training: {exc}") from exc

    provenance["status"] = "success" if completed.returncode == 0 else "failed"
    provenance["returncode"] = completed.returncode
    provenance["finished_at"] = dt.datetime.now().astimezone().isoformat()
    write_json(provenance_path, provenance)
    return completed.returncode


def main() -> int:
    try:
        return run()
    except ContinuationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
