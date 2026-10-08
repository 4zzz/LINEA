#!/usr/bin/env python3
"""Run a LINEA checkpoint on one or more explicitly selected image files."""

from __future__ import annotations

import argparse
import json
import math
import re
from contextlib import ExitStack
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import BatchImageCollateFunction
from util.inference_cli import (
    add_argument, add_data_loading_args, add_inference_option_args,
    add_model_args, add_output_args, validate_output_args,
)
from util.inference_output_helpers import (
    build_prediction_file_meta, create_inference_files, prediction_record_path,
)
from util.create_model import create_eval_model_from_checkpoint


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
        with Image.open(path) as source:
            image = source.convert("RGB")
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


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", type=Path, help="image paths")
    add_model_args(parser)
    add_data_loading_args(parser, include_split=False)
    add_inference_option_args(parser)
    add_output_args(parser, aliases={
        '--prediction-record': ('--prediction-files',),
        '--single-prediction-record': ('-p', '-o', '--single-prediction-file', '--output'),
        '--glb-model': ('--glb-models',),
        '--simple-json': ('--simple-json-files',),
        '--lines-2d-png': ('--save-visualizations',),
        '--prediction-record-save-exact-sample': ('--save-input',),
    })
    add_argument(parser, "--image", nargs="+", action="append", type=Path, default=[])
    add_argument(parser, "--image-list", action="append", type=Path, default=[])
    add_argument(parser, "--camera-k", nargs="+", type=float, default=None, metavar="VALUE")
    add_argument(parser, "--camera-k-json", type=Path, default=None)
    add_argument(
        parser,
        "--camera-k-from-exif",
        action="store_true",
        help="estimate missing camera intrinsics from EXIF metadata",
    )
    add_argument(parser, "--visualization-dir", type=Path, default=None)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = make_parser()
    args = parser.parse_args(argv)
    validate_output_args(parser, args)
    if args.batch_size < 1:
        parser.error('--batch-size must be at least 1.')
    if args.num_workers < 0:
        parser.error('--num-workers must be nonnegative.')
    if args.prediction_record_save_matching or args.prediction_record_matching_top_k != 5:
        parser.error('Matching options require ground-truth targets and are unavailable for explicit images.')
    if args.visualization_dir is not None and not args.lines_2d_png:
        parser.error('--visualization-dir requires --lines-2d-png.')
    return args


def create_image_inference_files(
    *, args, file_meta, stack, image_index, batch_index,
    samples, targets, raw_outputs, lines, scores,
):
    target = targets[batch_index]
    image_path = Path(target['image_path'])
    sample_directory = (
        image_output_directory(args.output_dir, image_index, image_path)
        if args.output_dir is not None else None
    )
    output_paths = {}
    if args.simple_json:
        path = sample_directory / 'prediction.json'
        if args.prediction_record and prediction_record_path(
            sample_directory / 'prediction', args.prediction_record_backend, per_sample=True,
        ) == path:
            path = sample_directory / 'prediction_simple.json'
        output_paths['simple_json'] = path

    if args.glb_model:
        if 'pred_lines3d' not in raw_outputs:
            raise ImageInferenceError('LINEA3D output does not contain pred_lines3d.')
        output_paths['glb_model'] = sample_directory / 'model.glb'

    if args.lines_2d_png:
        output_paths['lines_2d_png'] = (
            sample_directory / 'visualization.png' if args.visualization_dir is None else
            args.visualization_dir / f'{image_index + 1:03d}-{image_path.stem}.png'
        )

    return create_inference_files(
        batch_index, image_index, raw_outputs, lines, scores, targets,
        None, None, None, checkpoint=args.checkpoint, device=args.device,
        dataset_name='images', split='image',
        base_name_fn=lambda index, target: sample_directory / 'prediction' if sample_directory is not None else None,
        output_paths=output_paths, pred_threshold=args.pred_threshold,
        simple_json=args.simple_json, glb_model=args.glb_model, lines_2d_png=args.lines_2d_png,
        prediction_record=args.prediction_record, single_prediction_record=args.single_prediction_record,
        prediction_record_backend=args.prediction_record_backend,
        prediction_record_save_exact_sample=args.prediction_record_save_exact_sample,
        samples=samples, file_meta=file_meta, stack=stack,
        line_depths=raw_outputs.get('pred_line_depths'), glb_prediction_name='predictions_raw',
        record_meta={
            'image_path': str(image_path), 'original_size': target['orig_size'],
            'camera_K': target.get('camera_K'),
            'camera_K_metadata': target.get('camera_K_metadata'),
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise ImageInferenceError(f"Checkpoint does not exist: {checkpoint_path}")
    image_paths = collect_image_paths(args.images, args.image, args.image_list)
    shared_k = parse_camera_k(args.camera_k)
    camera_source = load_camera_k_json(args.camera_k_json)
    if shared_k is not None and camera_source is not None:
        raise ImageInferenceError("Use only one of --camera-k and --camera-k-json.")

    inference_model, model_args, model_meta = create_eval_model_from_checkpoint(checkpoint_path, raw_outputs=True)
    if args.glb_model and not model_args.linea3d:
        raise ImageInferenceError('--glb-model requires a LINEA3D checkpoint.')
    needs_camera_k = model_meta['needs_camera_k']
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

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)
    inference_model = inference_model.to(device).eval()
    args.device = str(device)
    args.checkpoint = checkpoint_path

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

    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().resolve()
    if args.single_prediction_record is not None:
        args.single_prediction_record = args.single_prediction_record.expanduser().resolve()
    if args.visualization_dir is not None:
        args.visualization_dir = args.visualization_dir.expanduser().resolve()

    wants_records = args.prediction_record or args.single_prediction_record is not None
    file_meta = None
    if wants_records:
        file_meta = build_prediction_file_meta(args, model_args, model_meta, REPO_ROOT, has_ground_truth=False)
        file_meta['export'].update(device=str(device), prediction_threshold=args.pred_threshold)
        file_meta['model'] = {
            'training_dataset_name': getattr(model_args, 'dataset_name', None),
            'linea3d': bool(model_args.linea3d),
            'line3d_pred_strategy': model_args.line3d_pred_strategy,
        }

    image_index = 0
    with torch.no_grad(), ExitStack() as stack:
        for samples, targets in dataloader:
            samples = samples.to(device)
            targets = [{key: value.to(device) if torch.is_tensor(value) else value
                        for key, value in target.items()} for target in targets]
            original_sizes = torch.stack([target['orig_size'].flip(0) for target in targets])
            raw_outputs, lines, scores = inference_model(samples, original_sizes, targets)
            for batch_index in range(len(targets)):
                create_image_inference_files(
                    args=args, file_meta=file_meta, stack=stack,
                    image_index=image_index, batch_index=batch_index, samples=samples,
                    targets=targets, raw_outputs=raw_outputs, lines=lines, scores=scores,
                )
                image_index += 1
    if args.single_prediction_record is not None:
        path = prediction_record_path(args.single_prediction_record, args.prediction_record_backend)
        print(f'Saved {image_index} predictions to {path}')

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ImageInferenceError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
