#!/usr/bin/env python3
"""Run a LINEA checkpoint on one or more explicitly selected image files."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import BatchImageCollateFunction
from tools.inference_cli import add_argument
from util.git_utils import git_output, git_repository_root
from util.line_model_export import save_line_model_glb
from util.prediction_record import PredictionRecord, make_prediction_file, save, utc_timestamp


DEFAULT_OUTPUT_NAME = "predictions.json.gz"
FULL_FRAME_DIAGONAL_MM = math.hypot(36.0, 24.0)

# Numeric constants keep this compatible with Pillow versions without ExifTags.IFD.
EXIF_IFD = 0x8769
TAG_FOCAL_LENGTH = 37386
TAG_FOCAL_LENGTH_35MM = 41989
TAG_PIXEL_X_DIMENSION = 40962
TAG_PIXEL_Y_DIMENSION = 40963
TAG_FOCAL_PLANE_X_RESOLUTION = 41486
TAG_FOCAL_PLANE_Y_RESOLUTION = 41487
TAG_FOCAL_PLANE_RESOLUTION_UNIT = 41488
RESOLUTION_UNIT_TO_MM = {
    2: 25.4,  # inch
    3: 10.0,  # centimeter
    4: 1.0,   # millimeter
    5: 0.001, # micrometer
}


class ImageInferenceError(RuntimeError):
    pass


def collect_image_paths(
    positional: Sequence[Path],
    image_groups: Sequence[Sequence[Path]],
    list_files: Sequence[Path],
) -> list[Path]:
    candidates = list(positional)
    for group in image_groups:
        candidates.extend(group)
    for list_file in list_files:
        try:
            lines = list_file.expanduser().read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise ImageInferenceError(f"Could not read image list {list_file}: {error}") from error
        candidates.extend(Path(line.strip()) for line in lines if line.strip() and not line.lstrip().startswith("#"))

    result = []
    seen = set()
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if path in seen:
            continue
        if not path.is_file():
            raise ImageInferenceError(f"Image does not exist: {path}")
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:
            raise ImageInferenceError(f"Not a readable image: {path} ({error})") from error
        result.append(path)
        seen.add(path)
    if not result:
        raise ImageInferenceError("No images supplied. Use positional paths, --image, or --image-list.")
    return result


def parse_camera_k(values: Sequence[float] | None) -> np.ndarray | None:
    if values is None:
        return None
    if len(values) == 4:
        fx, fy, cx, cy = values
        return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    if len(values) == 9:
        return np.asarray(values, dtype=np.float32).reshape(3, 3)
    raise ImageInferenceError("--camera-k expects either fx fy cx cy or all 9 row-major matrix values.")


def _positive_float(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return value if math.isfinite(value) and value > 0.0 else None


def _read_exif_tags(image: Image.Image) -> dict:
    exif = image.getexif()
    tags = dict(exif)
    try:
        tags.update(exif.get_ifd(EXIF_IFD))
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return tags


def camera_k_from_exif(image_path: Path) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    """Estimate K from EXIF, preferring physical focal-plane metadata."""
    try:
        with Image.open(image_path) as image:
            width, height = image.size
            tags = _read_exif_tags(image)
    except OSError as error:
        raise ImageInferenceError(f"Could not read EXIF from {image_path}: {error}") from error

    focal_mm = _positive_float(tags.get(TAG_FOCAL_LENGTH))
    x_resolution = _positive_float(tags.get(TAG_FOCAL_PLANE_X_RESOLUTION))
    y_resolution = _positive_float(tags.get(TAG_FOCAL_PLANE_Y_RESOLUTION))
    resolution_unit = tags.get(TAG_FOCAL_PLANE_RESOLUTION_UNIT)
    millimeters_per_unit = RESOLUTION_UNIT_TO_MM.get(resolution_unit)

    if focal_mm and x_resolution and millimeters_per_unit:
        exif_width = _positive_float(tags.get(TAG_PIXEL_X_DIMENSION)) or float(width)
        sensor_width_mm = exif_width / x_resolution * millimeters_per_unit
        fx = focal_mm / sensor_width_mm * width
        if y_resolution:
            exif_height = _positive_float(tags.get(TAG_PIXEL_Y_DIMENSION)) or float(height)
            sensor_height_mm = exif_height / y_resolution * millimeters_per_unit
            fy = focal_mm / sensor_height_mm * height
            focal_assumption = "focal-plane X/Y resolution"
        else:
            fy = fx
            focal_assumption = "focal-plane X resolution and square pixels"
        method = "exif_focal_plane_resolution"
        source_tags = {
            "focal_length_mm": focal_mm,
            "focal_plane_x_resolution": x_resolution,
            "focal_plane_y_resolution": y_resolution,
            "focal_plane_resolution_unit": resolution_unit,
            "pixel_x_dimension": exif_width,
            "pixel_y_dimension": _positive_float(tags.get(TAG_PIXEL_Y_DIMENSION)),
        }
    else:
        focal_35mm = _positive_float(tags.get(TAG_FOCAL_LENGTH_35MM))
        if focal_35mm is None:
            return None, None
        focal_pixels = focal_35mm * math.hypot(width, height) / FULL_FRAME_DIAGONAL_MM
        fx = fy = focal_pixels
        method = "exif_35mm_equivalent"
        focal_assumption = "35mm-equivalent diagonal field of view and square pixels"
        source_tags = {"focal_length_35mm": focal_35mm}

    matrix = np.asarray(
        [[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    metadata = {
        "source": "exif_estimate",
        "method": method,
        "K_original_pixels": matrix.tolist(),
        "image_size": [width, height],
        "source_tags": source_tags,
        "assumptions": [
            focal_assumption,
            "principal point at image center",
            "zero skew",
            "lens distortion ignored",
        ],
    }
    return matrix, metadata


def _extract_camera_k(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        if "K" in value:
            value = value["K"]
        elif isinstance(value.get("camera"), dict) and "K" in value["camera"]:
            value = value["camera"]["K"]
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ImageInferenceError(f"Camera K must have shape [3, 3], got {matrix.shape}.")
    return matrix


def load_camera_k_json(path: Path | None) -> Any:
    if path is None:
        return None
    try:
        return json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ImageInferenceError(f"Could not load camera intrinsics from {path}: {error}") from error


def camera_k_for_image(image_path: Path, shared_k: np.ndarray | None, source: Any) -> np.ndarray | None:
    if shared_k is not None:
        return shared_k.copy()
    if source is None:
        return None
    if isinstance(source, list):
        return _extract_camera_k(source)
    if not isinstance(source, dict):
        raise ImageInferenceError("Camera intrinsics JSON must be a 3x3 matrix or an image-to-matrix object.")
    if "K" in source or "camera" in source:
        return _extract_camera_k(source)

    keys = (str(image_path), str(image_path.resolve()), image_path.name)
    for key in keys:
        if key in source:
            return _extract_camera_k(source[key])
    raise ImageInferenceError(f"No camera intrinsics found for {image_path} in the JSON mapping.")


def default_output_path(checkpoint: Path) -> Path:
    return checkpoint.parent / "inference" / f"{checkpoint.name}_images" / DEFAULT_OUTPUT_NAME


def default_output_directory(checkpoint: Path) -> Path:
    return checkpoint.parent / "inference" / f"{checkpoint.name}_images"


def image_output_directory(root: Path, index: int, image_path: Path) -> Path:
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", image_path.stem).strip("._-")
    return root / f"{index + 1:03d}-{safe_stem or 'image'}"


def _make_transform(model_args):
    dataset_name = getattr(model_args, "dataset_name", "coco")
    if dataset_name == "monolines3d":
        from datasets.monolines3d import make_coco_transforms
    elif dataset_name == "coco":
        from datasets.coco import make_coco_transforms
    else:
        raise ImageInferenceError(f"Unsupported checkpoint dataset_name: {dataset_name}")
    return make_coco_transforms("test", model_args)


class ImagePathDataset(Dataset):
    def __init__(self, paths, camera_ks, camera_metadata, transform):
        self.paths = paths
        self.camera_ks = camera_ks
        self.camera_metadata = camera_metadata
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        width, height = image.size
        target = {
            "image_path": str(path),
            "image_id": str(path),
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
            "size": torch.tensor([height, width], dtype=torch.int64),
        }
        camera_k = self.camera_ks[index]
        if camera_k is not None:
            target["camera_K"] = torch.from_numpy(camera_k.copy())
            target["camera_K_metadata"] = self.camera_metadata[index]
        image, target = self.transform(image, target)
        return image, target


def _create_model(model_args):
    from models.registry import MODULE_BUILD_FUNCS

    class_module = getattr(model_args, "modelname")
    if class_module not in MODULE_BUILD_FUNCS._module_dict:
        raise ImageInferenceError(f"Unknown model module {class_module!r}.")
    return MODULE_BUILD_FUNCS.get(class_module)(model_args)


class InferenceModel(nn.Module):
    def __init__(self, model, postprocessor):
        super().__init__()
        self.model = model.deploy()
        self.postprocessor = postprocessor.deploy()

    def forward(self, images, original_sizes, targets):
        raw_outputs = self.model(images, targets)
        lines, scores = self.postprocessor(raw_outputs, original_sizes)
        return raw_outputs, lines, scores


def _move_targets_to_device(targets, device):
    return [{key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()} for target in targets]


def _sample_raw_outputs(raw_outputs, index):
    result = {}
    for key, value in raw_outputs.items():
        if "aux" in key or key == "dn_meta":
            continue
        result[key] = value[index] if torch.is_tensor(value) else value
    return result


def _build_prediction(raw_outputs, lines, scores, index, threshold):
    sample_scores = scores[index]
    keep = sample_scores > threshold
    prediction = {
        "lines2d": [
            {"endpoints": lines[index][query], "score": sample_scores[query]}
            for query in range(len(sample_scores))
            if keep[query]
        ]
    }
    if "pred_lines3d" in raw_outputs:
        prediction["lines3d"] = [
            {"endpoints": raw_outputs["pred_lines3d"][index][query], "score": sample_scores[query]}
            for query in range(len(sample_scores))
            if keep[query]
        ]
    if "pred_line_depths" in raw_outputs:
        prediction["line_depths"] = [
            {"depths": raw_outputs["pred_line_depths"][index][query], "score": sample_scores[query]}
            for query in range(len(sample_scores))
            if keep[query]
        ]
    return prediction


def _draw_prediction(image_path, lines, scores, threshold):
    image = Image.open(image_path).convert("RGB")
    drawing = ImageDraw.Draw(image)
    for line, score in zip(lines, scores):
        if score <= threshold:
            continue
        endpoints = line.detach().cpu().tolist()
        drawing.line(endpoints, fill="red", width=5)
        drawing.text((endpoints[0], endpoints[1]), f"{score.item():.2f}", fill="blue")
    return image


git_root = git_repository_root(REPO_ROOT)


def _git_output(*command):
    return git_output(command, git_root)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", type=Path, help="image paths")
    add_argument(parser, "--checkpoint", required=True, type=Path)
    add_argument(parser, "--image", nargs="+", action="append", type=Path, default=[])
    add_argument(parser, "--image-list", action="append", type=Path, default=[])
    prediction_group = parser.add_mutually_exclusive_group()
    add_argument(
        prediction_group,
        "--prediction-files",
        action="store_true",
        help="save prediction.json.gz in each numbered image directory",
    )
    add_argument(
        prediction_group,
        "-p",
        "-o",
        "--single-prediction-file",
        "--output",
        type=Path,
        default=None,
        help="save all image records into one JSON, JSON.GZ, or HDF5 file",
    )
    add_argument(parser, "--glb-models", action="store_true",
                 help="save model.glb in each numbered image directory")
    add_argument(parser, "--output-dir", type=Path, default=None)
    add_argument(parser, "--model-line-radius", type=float, default=None,
                 help="tube radius used in GLB models; defaults to a fraction of scene extent")
    add_argument(parser, "--batch-size", type=int, default=1)
    add_argument(parser, "--num-workers", type=int, default=0)
    add_argument(parser, "--device", default="cuda")
    add_argument(parser, "--pred-threshold", type=float, default=0.0)
    add_argument(parser, "--camera-k", nargs="+", type=float, default=None, metavar="VALUE")
    add_argument(parser, "--camera-k-json", type=Path, default=None)
    add_argument(
        parser,
        "--camera-k-from-exif",
        action="store_true",
        help="estimate missing camera intrinsics from EXIF metadata",
    )
    add_argument(parser, "--save-visualizations", action="store_true")
    add_argument(parser, "--visualization-dir", type=Path, default=None)
    add_argument(parser, "--save-input", action="store_true", help="include transformed image tensors in raw_data")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.batch_size < 1:
        raise ImageInferenceError("--batch-size must be at least 1.")
    if not (
        args.prediction_files
        or args.single_prediction_file
        or args.glb_models
        or args.save_visualizations
    ):
        raise ImageInferenceError(
            "Select at least one output: --prediction-files, --single-prediction-file, "
            "--glb-models, or --save-visualizations."
        )
    if (args.prediction_files or args.glb_models) and args.output_dir is None:
        raise ImageInferenceError("--output-dir is required with --prediction-files or --glb-models.")
    if args.model_line_radius is not None and args.model_line_radius <= 0:
        raise ImageInferenceError("--model-line-radius must be positive.")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise ImageInferenceError(f"Checkpoint does not exist: {checkpoint_path}")
    image_paths = collect_image_paths(args.images, args.image, args.image_list)
    shared_k = parse_camera_k(args.camera_k)
    camera_source = load_camera_k_json(args.camera_k_json)
    if shared_k is not None and camera_source is not None:
        raise ImageInferenceError("Use only one of --camera-k and --camera-k-json.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_args = checkpoint["args"]
    if not hasattr(model_args, "linea3d"):
        model_args.linea3d = False
    if not hasattr(model_args, "line3d_pred_strategy"):
        model_args.line3d_pred_strategy = "direct"
    if args.glb_models and not model_args.linea3d:
        raise ImageInferenceError("--glb-models requires a LINEA3D checkpoint.")
    if str(getattr(model_args, "backbone", "")).startswith("HGNetv2"):
        model_args.pretrained = False

    needs_camera_k = model_args.linea3d and model_args.line3d_pred_strategy == "uv_depth"
    camera_ks = []
    camera_metadata = []
    for path in image_paths:
        camera_k = camera_k_for_image(path, shared_k, camera_source)
        if shared_k is not None:
            metadata = {"source": "command_line", "K_original_pixels": camera_k.tolist()}
        elif camera_source is not None:
            metadata = {"source": "json", "K_original_pixels": camera_k.tolist()}
        elif args.camera_k_from_exif:
            camera_k, metadata = camera_k_from_exif(path)
            if camera_k is not None:
                print(
                    f"EXIF intrinsics for {path}: method={metadata['method']}, "
                    f"fx={camera_k[0, 0]:.3f}, fy={camera_k[1, 1]:.3f}, "
                    f"cx={camera_k[0, 2]:.3f}, cy={camera_k[1, 2]:.3f}"
                )
            else:
                print(f"No usable focal-length EXIF metadata in {path}.")
        else:
            metadata = None
        camera_ks.append(camera_k)
        camera_metadata.append(metadata)
    if needs_camera_k and any(camera_k is None for camera_k in camera_ks):
        raise ImageInferenceError(
            "This uv_depth checkpoint requires camera intrinsics. Supply --camera-k "
            "fx fy cx cy (or 9 matrix values), --camera-k-json, or "
            "--camera-k-from-exif when usable EXIF metadata is available."
        )

    model, postprocessor = _create_model(model_args)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    model.load_state_dict(state)

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)
    inference_model = InferenceModel(model, postprocessor).to(device).eval()

    dataset = ImagePathDataset(image_paths, camera_ks, camera_metadata, _make_transform(model_args))
    eval_size = getattr(model_args, "eval_spatial_size", None)
    base_size = eval_size[0] if isinstance(eval_size, (list, tuple)) else eval_size
    collate = BatchImageCollateFunction(base_size=base_size) if base_size is not None else BatchImageCollateFunction()
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate,
        num_workers=args.num_workers,
    )

    output_path = (
        args.single_prediction_file.expanduser().resolve()
        if args.single_prediction_file is not None else None
    )
    output_directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None else default_output_directory(checkpoint_path).resolve()
    )
    visualization_dir = args.visualization_dir
    if visualization_dir is None:
        visualization_dir = output_directory
    visualization_dir = visualization_dir.expanduser().resolve()

    file_meta = {
        "codebase": {
            "commit": _git_output("git", "rev-parse", "HEAD"),
            "diff": _git_output("git", "diff", "HEAD"),
        },
        "export": {
            "created_at_utc": utc_timestamp(),
            "weights_path": os.path.abspath(checkpoint_path),
            "device": str(device),
            "prediction_threshold": args.pred_threshold,
        },
        "model": {
            "training_dataset_name": getattr(model_args, "dataset_name", None),
            "linea3d": bool(model_args.linea3d),
            "line3d_pred_strategy": model_args.line3d_pred_strategy,
        },
    }
    records = []
    image_index = 0
    with torch.no_grad():
        for samples, targets in dataloader:
            original_sizes = torch.stack([
                target["orig_size"][[1, 0]] for target in targets
            ]).to(device)
            targets_device = _move_targets_to_device(targets, device)
            raw_outputs, lines, scores = inference_model(samples.to(device), original_sizes, targets_device)

            for batch_index, target in enumerate(targets):
                image_path = Path(target["image_path"])
                sample_directory = image_output_directory(
                    output_directory, image_index, image_path
                )
                if args.prediction_files or output_path is not None:
                    record = PredictionRecord(
                        record_id=f"image_{image_index:06d}",
                        raw_data={
                            "input": samples[batch_index] if args.save_input else {},
                            "target": target,
                            "output_raw": _sample_raw_outputs(raw_outputs, batch_index),
                        },
                        losses={},
                        prediction=_build_prediction(
                            raw_outputs, lines, scores, batch_index, args.pred_threshold
                        ),
                        meta={
                            "image_path": str(image_path),
                            "original_size": target["orig_size"],
                            "camera_K": target.get("camera_K"),
                            "camera_K_metadata": target.get("camera_K_metadata"),
                        },
                    )
                    if args.prediction_files:
                        prediction_file = make_prediction_file(
                            dataset_name="images",
                            codebase_name="LINEA",
                            records=[record],
                            meta=file_meta,
                        )
                        prediction_path = sample_directory / "prediction.json.gz"
                        save(prediction_file, prediction_path)
                        print(f"Saved prediction to {prediction_path}")
                    else:
                        records.append(record)

                if args.glb_models:
                    if "pred_lines3d" not in raw_outputs:
                        raise ImageInferenceError(
                            "LINEA3D output does not contain pred_lines3d."
                        )
                    keep = scores[batch_index] > args.pred_threshold
                    model_path = sample_directory / "model.glb"
                    save_line_model_glb(
                        model_path,
                        raw_outputs["pred_lines3d"][batch_index][keep],
                        line_radius=args.model_line_radius,
                        prediction_name="predictions_raw",
                    )
                    print(f"Saved 3D model to {model_path}")

                if args.save_visualizations:
                    if args.visualization_dir is None:
                        visualization_path = sample_directory / "visualization.png"
                    else:
                        visualization_path = (
                            visualization_dir
                            / f"{image_index + 1:03d}-{image_path.stem}.png"
                        )
                    visualization_path.parent.mkdir(parents=True, exist_ok=True)
                    image = _draw_prediction(
                        image_path, lines[batch_index], scores[batch_index], args.pred_threshold
                    )
                    image.save(visualization_path)
                image_index += 1

    if output_path is not None:
        prediction_file = make_prediction_file(
            dataset_name="images",
            codebase_name="LINEA",
            records=records,
            meta=file_meta,
        )
        save(prediction_file, output_path)
        print(f"Saved {len(records)} predictions to {output_path}")
    if args.save_visualizations:
        print(f"Saved visualizations to {visualization_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ImageInferenceError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
