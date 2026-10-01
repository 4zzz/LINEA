#!/usr/bin/env python3
"""Interactively inspect LINEA3D prediction records with Open3D."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util.prediction_record import (  # noqa: E402
    HDF5_DATASET_REF,
    HDF5_MAGIC,
    HDF5_STORAGE_VERSION,
    PredictionRecord,
    load as load_prediction_file,
)


PREDICTION_COLORS = np.asarray(
    [[0.10, 0.72, 0.95], [0.98, 0.73, 0.18]], dtype=np.float64
)
GROUND_TRUTH_COLOR = np.asarray([0.24, 0.92, 0.47], dtype=np.float64)
MATCH_COLOR = np.asarray([0.96, 0.31, 0.66], dtype=np.float64)
CAMERA_COLOR = np.asarray([0.95, 0.55, 0.16], dtype=np.float64)


def _read_json_bytes(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    if payload.startswith(b"\x1f\x8b"):
        payload = gzip.decompress(payload)
    return json.loads(payload.decode("utf-8"))


def _decode_hdf5_value(value: Any, arrays: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {HDF5_DATASET_REF}:
            array = arrays[value[HDF5_DATASET_REF]][()]
            return array.item() if getattr(array, "ndim", 0) == 0 else array
        return {key: _decode_hdf5_value(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_hdf5_value(item, arrays) for item in value]
    return value


class PredictionSource:
    """A prediction file whose HDF5 records are materialized on demand."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        with self.path.open("rb") as file:
            self.is_hdf5 = file.read(len(HDF5_MAGIC)) == HDF5_MAGIC

        if self.is_hdf5:
            self._open_hdf5_header()
            self._prediction_file = None
        else:
            self._prediction_file = load_prediction_file(self.path)
            self.dataset_name = self._prediction_file.dataset_name
            self.codebase_name = self._prediction_file.codebase_name
            self.record_count = len(self._prediction_file.records)

    def _open_hdf5_header(self) -> None:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("HDF5 input requires h5py: pip install h5py") from exc
        with h5py.File(self.path, "r") as file:
            version = int(file.attrs.get("hdf5_storage_version", 0))
            if version != HDF5_STORAGE_VERSION:
                raise ValueError(f"Unsupported HDF5 prediction storage version {version}")
            header = json.loads(file["header_json"][()].tobytes())
            header = _decode_hdf5_value(header, file["arrays"])
            self.dataset_name = str(header.get("dataset_name", ""))
            self.codebase_name = str(header.get("codebase_name", ""))
            self.record_count = len(file["records"])

    def load_record(self, index: int) -> PredictionRecord:
        if not 0 <= index < self.record_count:
            raise IndexError(index)
        if not self.is_hdf5:
            return self._prediction_file.records[index]

        import h5py

        with h5py.File(self.path, "r") as file:
            group = file["records"][f"{index:08d}"]
            manifest = json.loads(group["manifest_json"][()].tobytes())
            return PredictionRecord.from_dict(
                _decode_hdf5_value(manifest, group["arrays"])
            )

    def record_label(self, index: int) -> str:
        if not self.is_hdf5:
            record_id = self._prediction_file.records[index].record_id
            return record_id or f"record {index}"
        return f"record {index}"


@dataclass(frozen=True)
class RecordRef:
    source: PredictionSource
    index: int

    @property
    def label(self) -> str:
        return f"{self.source.path.name} :: {self.source.record_label(self.index)}"


@dataclass
class LineLayer:
    lines: np.ndarray
    scores: np.ndarray


@dataclass
class VisualRecord:
    record: PredictionRecord
    prediction_layers: dict[str, LineLayer]
    prediction_2d: LineLayer
    ground_truth: np.ndarray
    ground_truth_2d: np.ndarray
    matching_pairs: list[dict[str, Any]]
    camera_k: np.ndarray | None
    image_size: tuple[int, int] | None
    camera_to_world: np.ndarray | None
    pose_source: str | None
    image_path: Path | None
    image: np.ndarray | None
    notes: list[str]


def _as_lines3d(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0, 2, 3), dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 2, 3), dtype=np.float64)
    if array.ndim == 1 and array.size == 6:
        array = array.reshape(1, 2, 3)
    elif array.ndim == 2 and array.shape == (2, 3):
        array = array.reshape(1, 2, 3)
    elif array.ndim == 2 and array.shape[1] == 6:
        array = array.reshape(-1, 2, 3)
    elif array.ndim == 3 and array.shape[1:] == (2, 3):
        pass
    else:
        return np.empty((0, 2, 3), dtype=np.float64)
    return array[np.isfinite(array).all(axis=(1, 2))]


def _prediction_layer(value: Any) -> LineLayer:
    if value is None:
        return LineLayer(_as_lines3d(None), np.empty(0, dtype=np.float64))
    if isinstance(value, list) and value and isinstance(value[0], dict):
        lines, scores = [], []
        for item in value:
            line = _as_lines3d(item.get("endpoints", item.get("line")))
            if len(line) != 1:
                continue
            lines.append(line[0])
            scores.append(float(item.get("score", 1.0)))
        return LineLayer(
            np.asarray(lines, dtype=np.float64).reshape(-1, 2, 3),
            np.asarray(scores, dtype=np.float64),
        )
    lines = _as_lines3d(value)
    return LineLayer(lines, np.ones(len(lines), dtype=np.float64))


def _as_lines2d(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0, 2, 2), dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 2, 2), dtype=np.float64)
    if array.ndim == 1 and array.size == 4:
        array = array.reshape(1, 2, 2)
    elif array.ndim == 2 and array.shape == (2, 2):
        array = array.reshape(1, 2, 2)
    elif array.ndim == 2 and array.shape[1] == 4:
        array = array.reshape(-1, 2, 2)
    elif array.ndim == 3 and array.shape[1:] == (2, 2):
        pass
    else:
        return np.empty((0, 2, 2), dtype=np.float64)
    return array[np.isfinite(array).all(axis=(1, 2))]


def _prediction_layer2d(value: Any) -> LineLayer:
    if value is None:
        return LineLayer(_as_lines2d(None), np.empty(0, dtype=np.float64))
    if isinstance(value, list) and value and isinstance(value[0], dict):
        lines, scores = [], []
        for item in value:
            line = _as_lines2d(item.get("endpoints", item.get("line")))
            if len(line) != 1:
                continue
            lines.append(line[0])
            scores.append(float(item.get("score", 1.0)))
        return LineLayer(
            np.asarray(lines, dtype=np.float64).reshape(-1, 2, 2),
            np.asarray(scores, dtype=np.float64),
        )
    lines = _as_lines2d(value)
    return LineLayer(lines, np.ones(len(lines), dtype=np.float64))


def _matrix(value: Any, shape: tuple[int, int]) -> np.ndarray | None:
    try:
        result = np.asarray(value, dtype=np.float64).reshape(shape)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result).all() else None


def _walk_dicts(value: Any, prefix: str = "") -> Iterable[tuple[str, dict[str, Any]]]:
    if not isinstance(value, dict):
        return
    yield prefix, value
    for key, child in value.items():
        if isinstance(child, dict):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk_dicts(child, child_prefix)


def find_camera_to_world(record: PredictionRecord) -> tuple[np.ndarray | None, str | None]:
    roots = {"meta": record.meta, "target": record.raw_data.get("target", {})}
    c2w_keys = ("camera_to_world", "camera_to_world_matrix", "c2w")
    w2c_keys = ("world_to_camera", "world_to_camera_matrix", "w2c", "extrinsic")
    for root_name, root in roots.items():
        for prefix, mapping in _walk_dicts(root):
            location = f"{root_name}.{prefix}".rstrip(".")
            for key in c2w_keys:
                matrix = _matrix(mapping.get(key), (4, 4))
                if matrix is not None:
                    return matrix, f"{location}.{key}"
            for key in w2c_keys:
                matrix = _matrix(mapping.get(key), (4, 4))
                if matrix is not None:
                    try:
                        return np.linalg.inv(matrix), f"inverse({location}.{key})"
                    except np.linalg.LinAlgError:
                        pass
            rotation = _matrix(mapping.get("R"), (3, 3))
            translation = np.asarray(mapping.get("T", []), dtype=np.float64).reshape(-1)
            if rotation is not None and translation.size == 3 and np.isfinite(translation).all():
                world_to_camera = np.eye(4)
                world_to_camera[:3, :3] = rotation
                world_to_camera[:3, 3] = translation
                try:
                    return np.linalg.inv(world_to_camera), f"inverse({location}.R/T)"
                except np.linalg.LinAlgError:
                    pass
    return None, None


def _first_value(record: PredictionRecord, keys: tuple[str, ...]) -> Any:
    roots = (record.raw_data.get("target", {}), record.meta)
    for root in roots:
        for _, mapping in _walk_dicts(root):
            for key in keys:
                if key in mapping and mapping[key] is not None:
                    return mapping[key]
    return None


def _image_size(record: PredictionRecord) -> tuple[int, int] | None:
    value = _first_value(record, ("orig_size", "original_size", "size"))
    try:
        height, width = (int(item) for item in np.asarray(value).reshape(-1)[:2])
    except (TypeError, ValueError):
        return None
    return (height, width) if height > 0 and width > 0 else None


def _resolve_image_path(
    record: PredictionRecord, prediction_path: Path, image_roots: list[Path]
) -> Path | None:
    value = _first_value(record, ("image_path",))
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(prediction_path.parent / path)
    candidates.extend(root / path.name for root in image_roots)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _tensor_preview(value: Any, image_size: tuple[int, int] | None) -> np.ndarray | None:
    try:
        image = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if image.ndim != 3:
        return None
    if image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] not in (1, 3, 4):
        return None
    if image_size is not None:
        height, width = image_size
        image = image[:height, :width]
    image = image[..., :3].astype(np.float32)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    finite = np.isfinite(image)
    if not finite.any():
        return None
    low, high = np.percentile(image[finite], (1, 99))
    image = (image - low) / max(float(high - low), 1e-6)
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def prepare_visual_record(
    record: PredictionRecord,
    prediction_path: Path,
    image_roots: list[Path] | None = None,
) -> VisualRecord:
    image_roots = image_roots or []
    layers = {
        key: _prediction_layer(record.prediction.get(key))
        for key in ("lines3d_fitted", "lines3d")
        if key in record.prediction
    }
    target = record.raw_data.get("target", {})
    prediction_2d = _prediction_layer2d(record.prediction.get("lines2d"))
    ground_truth = _as_lines3d(target.get("lines3d"))
    ground_truth_2d = _as_lines2d(target.get("lines"))
    matching = record.meta.get("matching", {})
    pairs = matching.get("pairs", []) if isinstance(matching, dict) else []

    camera_k = _matrix(_first_value(record, ("camera_K", "K")), (3, 3))
    image_size = _image_size(record)
    camera_to_world, pose_source = find_camera_to_world(record)
    image_path = _resolve_image_path(record, prediction_path, image_roots)
    image = None
    notes = []
    if image_path is not None:
        try:
            from PIL import Image

            image = np.asarray(Image.open(image_path).convert("RGB"))
        except (ImportError, OSError) as exc:
            notes.append(f"Could not read image: {exc}")
    else:
        model_input = record.raw_data.get("input")
        image = _tensor_preview(model_input, image_size) if model_input is not None else None
        if image is not None:
            notes.append("Image is a contrast-normalized model-input preview")
    return VisualRecord(
        record=record,
        prediction_layers=layers,
        prediction_2d=prediction_2d,
        ground_truth=ground_truth,
        ground_truth_2d=ground_truth_2d,
        matching_pairs=pairs if isinstance(pairs, list) else [],
        camera_k=camera_k,
        image_size=image_size,
        camera_to_world=camera_to_world,
        pose_source=pose_source,
        image_path=image_path,
        image=image,
        notes=notes,
    )


def transform_points(points: np.ndarray, transform: np.ndarray | None) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if transform is None or points.size == 0:
        return points.copy()
    flat = points.reshape(-1, 3)
    homogeneous = np.concatenate([flat, np.ones((len(flat), 1))], axis=1)
    transformed = homogeneous @ transform.T
    return transformed[:, :3].reshape(points.shape)


def camera_extrinsic(camera_to_world: np.ndarray | None) -> np.ndarray:
    """Return the world-to-camera matrix expected by Open3D."""
    if camera_to_world is None:
        return np.eye(4, dtype=np.float64)
    return np.linalg.inv(camera_to_world)


def threshold_layer(layer: LineLayer, threshold: float) -> LineLayer:
    mask = np.isfinite(layer.scores) & (layer.scores >= threshold)
    return LineLayer(layer.lines[mask], layer.scores[mask])


def score_colors(
    scores: np.ndarray, high_color: np.ndarray | None = None
) -> np.ndarray:
    scores = np.clip(np.nan_to_num(scores, nan=0.0), 0.0, 1.0)[:, None]
    if high_color is None:
        return PREDICTION_COLORS[0] * (1.0 - scores) + PREDICTION_COLORS[1] * scores
    high_color = np.asarray(high_color, dtype=np.float64)
    return high_color[None] * (0.35 + 0.65 * scores)


def camera_frustum(
    camera_k: np.ndarray,
    image_size: tuple[int, int],
    depth: float,
    camera_to_world: np.ndarray | None = None,
) -> np.ndarray:
    height, width = image_size
    pixels = np.asarray(
        [[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]],
        dtype=np.float64,
    )
    corners = (np.linalg.inv(camera_k) @ pixels.T).T * float(depth)
    points = np.concatenate([np.zeros((1, 3)), corners], axis=0)
    return transform_points(points, camera_to_world)


def image_plane_corners(
    camera_k: np.ndarray,
    image_size: tuple[int, int],
    depth: float,
    camera_to_world: np.ndarray | None = None,
) -> np.ndarray:
    return camera_frustum(camera_k, image_size, depth, camera_to_world)[1:]


def matching_connectors(
    pairs: list[dict[str, Any]], threshold: float, fitted: bool
) -> np.ndarray:
    connectors = []
    prediction_key = "prediction_line3d_aligned" if fitted else "prediction_line3d"
    for pair in pairs:
        if float(pair.get("score", 1.0)) < threshold:
            continue
        prediction = _as_lines3d(pair.get(prediction_key))
        target = _as_lines3d(pair.get("target_line3d"))
        if len(prediction) == 1 and len(target) == 1:
            connectors.append([prediction[0].mean(axis=0), target[0].mean(axis=0)])
    return np.asarray(connectors, dtype=np.float64).reshape(-1, 2, 3)


def _line_set(o3d: Any, lines: np.ndarray, colors: np.ndarray) -> Any:
    geometry = o3d.geometry.LineSet()
    if len(lines) == 0:
        return geometry
    geometry.points = o3d.utility.Vector3dVector(lines.reshape(-1, 3))
    geometry.lines = o3d.utility.Vector2iVector(
        np.arange(len(lines) * 2, dtype=np.int32).reshape(-1, 2)
    )
    if colors.ndim == 1:
        colors = np.repeat(colors[None], len(lines), axis=0)
    geometry.colors = o3d.utility.Vector3dVector(colors)
    return geometry


def _frustum_line_set(
    o3d: Any, points: np.ndarray, color: np.ndarray = CAMERA_COLOR
) -> Any:
    edges = np.asarray(
        [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]],
        dtype=np.int32,
    )
    geometry = o3d.geometry.LineSet()
    geometry.points = o3d.utility.Vector3dVector(points)
    geometry.lines = o3d.utility.Vector2iVector(edges)
    geometry.colors = o3d.utility.Vector3dVector(np.repeat(color[None], 8, axis=0))
    return geometry


def _image_point_cloud(
    o3d: Any,
    image: np.ndarray,
    camera_k: np.ndarray,
    image_size: tuple[int, int],
    depth: float,
    camera_to_world: np.ndarray | None,
    opacity: float,
    max_points: int = 180_000,
) -> Any:
    height, width = image_size
    stride = max(1, int(np.ceil(np.sqrt((height * width) / max_points))))
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    # Open3D 0.20's transparency shader crashes on some Linux builds. A stable
    # deterministic mask gives the image plane genuine visual see-through.
    keep = ((xs * 73856093 + ys * 19349663) % 1000) < int(opacity * 1000)
    xs = xs[keep]
    ys = ys[keep]
    pixels = np.stack([xs, ys, np.ones_like(xs)], axis=1).astype(np.float64)
    points = (np.linalg.inv(camera_k) @ pixels.T).T * float(depth)
    points = transform_points(points, camera_to_world)

    image_height, image_width = image.shape[:2]
    image_xs = np.minimum((xs * image_width / width).astype(int), image_width - 1)
    image_ys = np.minimum((ys * image_height / height).astype(int), image_height - 1)
    colors = image[image_ys, image_xs, :3].astype(np.float64) / 255.0

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def _image_mesh(o3d: Any, corners: np.ndarray) -> Any:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    mesh.triangle_uvs = o3d.utility.Vector2dVector(
        [[0, 1], [1, 1], [1, 0], [0, 1], [1, 0], [0, 0]]
    )
    mesh.compute_triangle_normals()
    return mesh


def lines2d_to_image_pixels(
    lines: np.ndarray,
    coordinate_size: tuple[int, int],
    image_shape: tuple[int, int],
) -> np.ndarray:
    lines = _as_lines2d(lines)
    if not len(lines):
        return lines
    coordinate_height, coordinate_width = coordinate_size
    image_height, image_width = image_shape
    result = lines.copy()
    # Dataset GT is normalized; inference predictions are normally in pixels.
    if result.min() >= -0.5 and result.max() <= 1.5:
        result[..., 0] *= coordinate_width
        result[..., 1] *= coordinate_height
    result[..., 0] *= image_width / coordinate_width
    result[..., 1] *= image_height / coordinate_height
    return result


def compose_image_texture(
    image: np.ndarray,
    image_size: tuple[int, int],
    predictions: LineLayer,
    ground_truth: np.ndarray,
    threshold: float,
    draw_predictions: bool,
    draw_ground_truth: bool,
    prediction_color: np.ndarray | None = None,
    ground_truth_color: np.ndarray | None = None,
    prediction_width: float | None = None,
    ground_truth_width: float | None = None,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    texture = Image.fromarray(np.ascontiguousarray(image)).convert("RGB")
    draw = ImageDraw.Draw(texture)
    image_height, image_width = image.shape[:2]
    default_width = max(2, round(min(image_height, image_width) / 260))
    prediction_width = max(1, round(prediction_width or default_width))
    ground_truth_width = max(1, round(ground_truth_width or default_width + 2))
    prediction_color = (
        PREDICTION_COLORS[1] if prediction_color is None else prediction_color
    )
    ground_truth_color = (
        GROUND_TRUTH_COLOR if ground_truth_color is None else ground_truth_color
    )

    if draw_ground_truth:
        lines = lines2d_to_image_pixels(
            ground_truth, image_size, (image_height, image_width)
        )
        color = tuple((ground_truth_color * 255).astype(np.uint8))
        endpoint_radius = max(2, ground_truth_width)
        for line in lines:
            points = [tuple(point) for point in line]
            draw.line(points, fill=color, width=ground_truth_width)
            for x, y in points:
                draw.ellipse(
                    (
                        x - endpoint_radius,
                        y - endpoint_radius,
                        x + endpoint_radius,
                        y + endpoint_radius,
                    ),
                    fill=color,
                )

    if draw_predictions:
        filtered = threshold_layer(predictions, threshold)
        lines = lines2d_to_image_pixels(
            filtered.lines, image_size, (image_height, image_width)
        )
        colors = (
            score_colors(filtered.scores, prediction_color) * 255
        ).astype(np.uint8)
        endpoint_radius = max(2, prediction_width)
        for line, color_array in zip(lines, colors):
            color = tuple(color_array)
            points = [tuple(point) for point in line]
            draw.line(points, fill=color, width=prediction_width)
            for x, y in points:
                draw.ellipse(
                    (
                        x - endpoint_radius,
                        y - endpoint_radius,
                        x + endpoint_radius,
                        y + endpoint_radius,
                    ),
                    fill=color,
                )
    return np.asarray(texture)


class PredictionViewer:
    def __init__(
        self,
        o3d: Any,
        references: list[RecordRef],
        image_roots: list[Path],
        threshold: float,
        image_depth: float,
        image_opacity: float,
        image_mode: str,
        line_width: float,
    ):
        self.o3d = o3d
        self.gui = o3d.visualization.gui
        self.rendering = o3d.visualization.rendering
        self.references = references
        self.image_roots = image_roots
        self.threshold = threshold
        self.image_depth = image_depth
        self.image_opacity = image_opacity
        self.image_mode = image_mode
        self.line_width = line_width
        self.prediction_3d_color = PREDICTION_COLORS[1].copy()
        self.prediction_2d_color = np.asarray([0.98, 0.58, 0.12])
        self.ground_truth_3d_color = GROUND_TRUTH_COLOR.copy()
        self.ground_truth_2d_color = np.asarray([0.20, 1.00, 0.40])
        self.match_color = MATCH_COLOR.copy()
        self.camera_color = CAMERA_COLOR.copy()
        self.prediction_3d_width = line_width
        self.prediction_2d_width = line_width
        self.ground_truth_3d_width = line_width + 1.0
        self.ground_truth_2d_width = line_width + 1.0
        self.match_width = max(1.0, line_width - 1.0)
        self.camera_width = 2.0
        self.current: VisualRecord | None = None
        self._geometry_names: set[str] = set()

        app = self.gui.Application.instance
        self.window = app.create_window("LINEA3D prediction viewer", 1480, 900)
        self.scene_widget = self.gui.SceneWidget()
        self.scene_widget.scene = self.rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.025, 0.03, 0.04, 1.0])
        self.panel = self.gui.ScrollableVert(
            8, self.gui.Margins(12, 12, 12, 12)
        )
        self.panel.background_color = self.gui.Color(
            0.025, 0.030, 0.040, 1.0
        )
        self._build_controls()
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)
        self._load_record(0, reset_camera=True)

    def _build_controls(self) -> None:
        self._section_count = 0
        self._section_widgets: dict[str, dict[str, Any]] = {}
        record_section = self._section("Record")
        self.record_combo = self.gui.Combobox()
        for reference in self.references:
            self.record_combo.add_item(reference.label)
        self.record_combo.set_on_selection_changed(
            lambda _text, index: self._load_record(index, reset_camera=True)
        )
        record_section.add_child(self.record_combo)

        prediction_section = self._section("Prediction")
        prediction_section.add_child(self.gui.Label("3D prediction layer"))
        self.layer_combo = self.gui.Combobox()
        self.layer_combo.set_on_selection_changed(
            lambda _text, _index: self._redraw()
        )
        prediction_section.add_child(self.layer_combo)
        prediction_section.add_child(self.gui.Label("Confidence threshold"))
        threshold_row = self.gui.Horiz(6)
        self.threshold_slider = self.gui.Slider(self.gui.Slider.DOUBLE)
        self.threshold_slider.set_limits(0.0, 1.0)
        self.threshold_slider.double_value = self.threshold
        self.threshold_slider.set_on_value_changed(self._threshold_from_slider)
        self.threshold_edit = self.gui.NumberEdit(self.gui.NumberEdit.DOUBLE)
        self.threshold_edit.set_limits(0.0, 1.0)
        self.threshold_edit.double_value = self.threshold
        self.threshold_edit.set_on_value_changed(self._threshold_from_edit)
        threshold_row.add_child(self.threshold_slider)
        threshold_row.add_child(self.threshold_edit)
        prediction_section.add_child(threshold_row)
        prediction_section.add_child(self.gui.Label("Visible / color / width"))
        self.show_predictions = self._style_control(
            prediction_section,
            "3D lines",
            True,
            "prediction_3d_color",
            "prediction_3d_width",
        )
        self.show_2d_predictions = self._style_control(
            prediction_section,
            "2D lines",
            True,
            "prediction_2d_color",
            "prediction_2d_width",
        )

        annotation_section = self._section("Annotation")
        annotation_section.add_child(self.gui.Label("Visible / color / width"))
        self.show_ground_truth = self._style_control(
            annotation_section,
            "3D lines",
            True,
            "ground_truth_3d_color",
            "ground_truth_3d_width",
        )
        self.show_2d_ground_truth = self._style_control(
            annotation_section,
            "2D lines",
            True,
            "ground_truth_2d_color",
            "ground_truth_2d_width",
        )

        matching_section = self._section("Matching", is_open=False)
        matching_section.add_child(self.gui.Label("Visible / color / width"))
        self.show_matching = self._style_control(
            matching_section,
            "Matched pairs",
            False,
            "match_color",
            "match_width",
        )

        drawing_section = self._section("Drawing")
        drawing_section.add_child(self.gui.Label("Camera image mode"))
        self.image_mode_combo = self.gui.Combobox()
        for mode in ("texture", "points", "none"):
            self.image_mode_combo.add_item(mode)
        self.image_mode_combo.selected_index = ("texture", "points", "none").index(
            self.image_mode
        )
        self.image_mode_combo.set_on_selection_changed(
            lambda _text, _index: self._redraw()
        )
        drawing_section.add_child(self.image_mode_combo)
        drawing_section.add_child(self.gui.Label("Visible / color / width"))
        self.show_camera = self._style_control(
            drawing_section,
            "Camera frustum",
            True,
            "camera_color",
            "camera_width",
        )
        self.show_axes = self._checkbox("Coordinate frame", False)
        drawing_section.add_child(self.show_axes)

        camera_buttons = self.gui.Horiz(6)
        self.focus_button = self.gui.Button("Focus camera")
        self.focus_button.set_on_clicked(self._focus_camera)
        camera_buttons.add_child(self.focus_button)
        self.reset_button = self.gui.Button("Fit all")
        self.reset_button.set_on_clicked(self._fit_camera)
        camera_buttons.add_child(self.reset_button)
        drawing_section.add_child(camera_buttons)

        self.status = self.gui.Label("")
        self.panel.add_child(self.status)

    def _section(self, title: str, is_open: bool = True) -> Any:
        gap = None
        if self._section_count:
            gap = self.gui.Vert(0, self.gui.Margins(0, 0, 0, 0))
            gap.add_fixed(14)
            self.panel.add_child(gap)

        header = self.gui.Button(("- " if is_open else "+ ") + title)
        header.background_color = self.gui.Color(0.12, 0.32, 0.44, 1.0)
        header.horizontal_padding_em = 0.8
        header.vertical_padding_em = 0.35
        self.panel.add_child(header)

        section = self.gui.Vert(8, self.gui.Margins(12, 10, 12, 8))
        section.visible = is_open
        self.panel.add_child(section)

        group = {
            "header": header,
            "content": section,
            "gap": gap,
            "open": is_open,
        }
        self._section_widgets[title] = group

        def toggle_section() -> None:
            group["open"] = not group["open"]
            section.visible = group["open"]
            header.text = ("- " if group["open"] else "+ ") + title
            self.window.set_needs_layout()

        header.set_on_clicked(toggle_section)
        self._section_count += 1
        return section

    def _set_section_available(self, title: str, available: bool) -> None:
        group = self._section_widgets[title]
        group["header"].visible = available
        group["content"].visible = available and group["open"]
        if group["gap"] is not None:
            group["gap"].visible = available

    def _style_control(
        self,
        parent: Any,
        label: str,
        checked: bool,
        color_attribute: str,
        width_attribute: str,
    ) -> Any:
        row = self.gui.Horiz(6)
        checkbox = self._checkbox(label, checked)
        row.add_child(checkbox)
        row.add_stretch()

        color_edit = self.gui.ColorEdit()
        color = getattr(self, color_attribute)
        color_edit.color_value = self.gui.Color(*color, 1.0)
        color_edit.set_on_value_changed(
            lambda value, name=color_attribute: self._set_color(name, value)
        )
        row.add_child(color_edit)

        width_edit = self.gui.NumberEdit(self.gui.NumberEdit.DOUBLE)
        width_edit.set_limits(0.5, 24.0)
        width_edit.double_value = getattr(self, width_attribute)
        width_edit.set_on_value_changed(
            lambda value, name=width_attribute: self._set_width(name, value)
        )
        row.add_child(width_edit)
        parent.add_child(row)
        return checkbox

    def _set_color(self, attribute: str, color: Any) -> None:
        setattr(
            self,
            attribute,
            np.asarray([color.red, color.green, color.blue], dtype=np.float64),
        )
        self._redraw()

    def _set_width(self, attribute: str, width: float) -> None:
        setattr(self, attribute, max(0.5, float(width)))
        self._redraw()

    def _checkbox(self, label: str, checked: bool) -> Any:
        checkbox = self.gui.Checkbox(label)
        checkbox.checked = checked
        checkbox.set_on_checked(lambda _checked: self._redraw())
        return checkbox

    def _on_layout(self, context: Any) -> None:
        rect = self.window.content_rect
        panel_width = min(410, max(320, int(rect.width * 0.29)))
        self.panel.frame = self.gui.Rect(
            rect.get_right() - panel_width, rect.y, panel_width, rect.height
        )
        self.scene_widget.frame = self.gui.Rect(
            rect.x, rect.y, rect.width - panel_width, rect.height
        )

    def _threshold_from_slider(self, value: float) -> None:
        self.threshold = value
        self.threshold_edit.double_value = value
        self._redraw()

    def _threshold_from_edit(self, value: float) -> None:
        self.threshold = min(max(value, 0.0), 1.0)
        self.threshold_slider.double_value = self.threshold
        self._redraw()

    def _load_record(self, index: int, reset_camera: bool) -> None:
        reference = self.references[index]
        try:
            record = reference.source.load_record(reference.index)
            self.current = prepare_visual_record(
                record, reference.source.path, self.image_roots
            )
        except Exception as exc:
            self.status.text = f"Could not load record: {exc}"
            return

        has_predictions = bool(
            len(self.current.prediction_2d.lines)
            or any(
                len(layer.lines)
                for layer in self.current.prediction_layers.values()
            )
        )
        has_annotations = bool(
            len(self.current.ground_truth_2d)
            or len(self.current.ground_truth)
        )
        self._set_section_available("Prediction", has_predictions)
        self._set_section_available("Annotation", has_annotations)
        self._set_section_available(
            "Matching", bool(self.current.matching_pairs)
        )

        self.layer_combo.clear_items()
        keys = list(self.current.prediction_layers)
        keys.sort(key=lambda key: (key != "lines3d_fitted", key))
        for key in keys:
            self.layer_combo.add_item(key)
        if keys:
            self.layer_combo.selected_index = 0
        self._redraw()
        if reset_camera:
            self._fit_camera()

    def _selected_image_mode(self) -> str:
        return ("texture", "points", "none")[self.image_mode_combo.selected_index]

    def _selected_layer(self) -> tuple[str | None, LineLayer | None]:
        if self.current is None or not self.current.prediction_layers:
            return None, None
        keys = list(self.current.prediction_layers)
        keys.sort(key=lambda key: (key != "lines3d_fitted", key))
        index = min(self.layer_combo.selected_index, len(keys) - 1)
        key = keys[index]
        return key, self.current.prediction_layers[key]

    def _material(self, shader: str = "unlitLine", width: float | None = None) -> Any:
        material = self.rendering.MaterialRecord()
        material.shader = shader
        if width is not None:
            material.line_width = width
        return material

    def _add_geometry(self, name: str, geometry: Any, material: Any) -> None:
        self.scene_widget.scene.add_geometry(name, geometry, material)
        self._geometry_names.add(name)

    def _clear(self) -> None:
        for name in self._geometry_names:
            self.scene_widget.scene.remove_geometry(name)
        self._geometry_names.clear()

    def _redraw(self) -> None:
        if self.current is None:
            return
        self._clear()
        key, layer = self._selected_layer()
        transform = self.current.camera_to_world
        displayed_predictions = 0
        if self.show_predictions.checked and layer is not None:
            filtered = threshold_layer(layer, self.threshold)
            lines = transform_points(filtered.lines, transform)
            self._add_geometry(
                "predictions",
                _line_set(
                    self.o3d,
                    lines,
                    score_colors(filtered.scores, self.prediction_3d_color),
                ),
                self._material(width=self.prediction_3d_width),
            )
            displayed_predictions = len(lines)

        if self.show_ground_truth.checked and len(self.current.ground_truth):
            lines = transform_points(self.current.ground_truth, transform)
            self._add_geometry(
                "ground_truth",
                _line_set(self.o3d, lines, self.ground_truth_3d_color),
                self._material(width=self.ground_truth_3d_width),
            )

        if self.show_matching.checked and self.current.matching_pairs:
            connectors = matching_connectors(
                self.current.matching_pairs,
                self.threshold,
                fitted=key == "lines3d_fitted",
            )
            connectors = transform_points(connectors, transform)
            self._add_geometry(
                "matching",
                _line_set(self.o3d, connectors, self.match_color),
                self._material(width=self.match_width),
            )

        can_show_camera = self.current.camera_k is not None and self.current.image_size is not None
        if self.show_camera.checked and can_show_camera:
            points = camera_frustum(
                self.current.camera_k,
                self.current.image_size,
                self.image_depth,
                transform,
            )
            self._add_geometry(
                "camera",
                _frustum_line_set(self.o3d, points, self.camera_color),
                self._material(width=self.camera_width),
            )

        image_mode = self._selected_image_mode()
        if (
            image_mode != "none"
            and can_show_camera
            and self.current.image is not None
        ):
            image = compose_image_texture(
                self.current.image,
                self.current.image_size,
                self.current.prediction_2d,
                self.current.ground_truth_2d,
                self.threshold,
                self.show_2d_predictions.checked,
                self.show_2d_ground_truth.checked,
                self.prediction_2d_color,
                self.ground_truth_2d_color,
                self.prediction_2d_width,
                self.ground_truth_2d_width,
            )
            material = self._material("defaultUnlit")
            if image_mode == "texture":
                corners = image_plane_corners(
                    self.current.camera_k,
                    self.current.image_size,
                    self.image_depth,
                    transform,
                )
                mesh = _image_mesh(self.o3d, corners)
                material.albedo_img = self.o3d.geometry.Image(
                    np.ascontiguousarray(image)
                )
                self._add_geometry("image", mesh, material)
            else:
                cloud = _image_point_cloud(
                    self.o3d,
                    image,
                    self.current.camera_k,
                    self.current.image_size,
                    self.image_depth,
                    transform,
                    self.image_opacity,
                )
                material.point_size = 2.0
                self._add_geometry("image", cloud, material)

        if self.show_axes.checked:
            axes = self.o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=max(0.2, self.image_depth * 0.3)
            )
            if transform is not None:
                axes.transform(transform)
            self._add_geometry("axes", axes, self._material("defaultUnlit"))

        capabilities = [
            f"pred {displayed_predictions}",
            f"GT {len(self.current.ground_truth)}",
            f"matches {len(self.current.matching_pairs)}",
            (
                f"image {image_mode}"
                if self.current.image is not None
                else "image unavailable"
            ),
            f"pose {self.current.pose_source or 'camera-local'}",
        ]
        self.status.text = " | ".join(capabilities + self.current.notes)

    def _fit_camera(self) -> None:
        bounds = self.scene_widget.scene.bounding_box
        if bounds.is_empty():
            return
        self.scene_widget.setup_camera(60.0, bounds, bounds.get_center())

    def _focus_camera(self) -> None:
        if self.current is None:
            return
        if self.current.camera_k is None or self.current.image_size is None:
            self.status.text = (
                "Cannot focus camera: record has no intrinsics or image size"
            )
            return
        bounds = self.scene_widget.scene.bounding_box
        if bounds.is_empty():
            return
        height, width = self.current.image_size
        try:
            extrinsic = camera_extrinsic(self.current.camera_to_world)
        except np.linalg.LinAlgError:
            self.status.text = "Cannot focus camera: camera pose is singular"
            return
        self.scene_widget.setup_camera(
            self.current.camera_k,
            extrinsic,
            width,
            height,
            bounds,
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_files", nargs="+", type=Path)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--image-depth", "--image_depth", type=float, default=1.0)
    parser.add_argument(
        "--image-mode",
        "--image_mode",
        choices=("texture", "points", "none"),
        default="texture",
        help="How to render the camera image plane.",
    )
    parser.add_argument(
        "--image-opacity",
        "--image_opacity",
        type=float,
        default=0.55,
        help="Density of the see-through point-sampled image plane (0 to 1).",
    )
    parser.add_argument("--line-width", "--line_width", type=float, default=3.0)
    parser.add_argument(
        "--image-root",
        "--image_root",
        action="append",
        default=[],
        type=Path,
        help="Fallback directory for image basenames stored in records (repeatable).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be between 0 and 1")
    if not 0.0 <= args.image_opacity <= 1.0:
        raise SystemExit("--image-opacity must be between 0 and 1")
    sources = [PredictionSource(path) for path in args.prediction_files]
    references = [
        RecordRef(source, index)
        for source in sources
        for index in range(source.record_count)
    ]
    if not references:
        raise SystemExit("No prediction records found")
    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit(
            "Open3D is required for this viewer. Install it in the active environment "
            "with: pip install open3d"
        ) from exc

    app = o3d.visualization.gui.Application.instance
    app.initialize()
    viewer = PredictionViewer(
        o3d,
        references,
        [path.expanduser().resolve() for path in args.image_root],
        args.threshold,
        args.image_depth,
        args.image_opacity,
        args.image_mode,
        args.line_width,
    )
    app.run()
    del viewer
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
