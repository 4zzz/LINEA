import os
import argparse
from pathlib import Path
import warnings
import json
import zipfile
from typing import Tuple
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms.functional as F

import datasets.transforms as T

def save_json(file, data):
    f = open(file, "w")
    json.dump(data, f, indent = 6)
    f.close()

class DirectoryAnnotationSource:
    def __init__(self, annotations_dir: Path):
        self.annotations_dir = Path(annotations_dir)

    def info(self):
        if self.annotations_dir.is_symlink():
            return {
                "symlink": str(self.annotations_dir.resolve(strict=True))
            }

        return "regular_dir"

    def iter_annotations(self):
        """
        Yields:
            rel_annotation_path: Path
            load_annotation: callable returning parsed JSON
            debug_path: printable source path
        """
        for annotation_json_path in self.annotations_dir.rglob("*.json"):
            rel_annotation_path = annotation_json_path.relative_to(self.annotations_dir)

            def load_annotation(path=annotation_json_path):
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)

            yield rel_annotation_path, load_annotation, annotation_json_path


class ZipAnnotationSource:
    def __init__(self, zip_path: Path):
        self.zip_path = Path(zip_path)

    def info(self):
        return {
            "zip": str(self.zip_path.resolve(strict=True))
        }

    def iter_annotations(self):
        """
        Supports zip layouts like:

            foo.jpg.json
            subdir/foo.jpg.json

        and also:

            limap_annotations/foo.jpg.json
            limap_annotations/subdir/foo.jpg.json
        """
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            names = zf.namelist()

            for name in names:
                if name.endswith("/"):
                    continue

                if not name.endswith(".json"):
                    continue

                if name.startswith("__MACOSX/"):
                    continue

                rel_annotation_path = self._normalize_zip_member_name(name)

                if rel_annotation_path is None:
                    continue

                def load_annotation(member_name=name):
                    with zf.open(member_name, "r") as f:
                        return json.load(f)

                debug_path = f"{self.zip_path}!{name}"

                yield rel_annotation_path, load_annotation, debug_path

    def _normalize_zip_member_name(self, name: str) -> Path | None:
        path = Path(name)

        parts = path.parts

        # Handle zips that contain:
        #   limap_annotations/foo.jpg.json
        if len(parts) >= 2 and parts[0] == "limap_annotations":
            path = Path(*parts[1:])

        # Safety / cleanup: ignore weird absolute or parent-relative paths
        if path.is_absolute():
            return None

        if ".." in path.parts:
            return None

        return path

def build_args(parser, required=None):
    required_arg = {}
    if required is not None:
        required_arg['required'] = required

    parser.add_argument('--mono3d_dataset_root', type=str, **required_arg)
    parser.add_argument('--mono3d_preload_images', type=bool, default=False)
    parser.add_argument(
        '--mono3d_use_image_normalized_target_line_coords',
        action='store_true',
        default=False,
        help='Deprecated compatibility option; 2D lines are normalized once after geometric transforms.',
    )
    parser.add_argument('--mono3d_normalize_line_space', action='store_true', default=False)
    parser.add_argument('--mono3d_do_not_normalize_images', action='store_true', default=False)
    parser.add_argument('--mono3d_train2d', action='store_true', default=False)
    parser.add_argument('--mono3d_strict', action='store_true', default=False)
    parser.add_argument(
        '--mono3d_invalid_line_filter',
        choices=('none', 'lines', 'samples'),
        default='none',
        help='Keep invalid geometry, remove invalid lines, or remove their entire samples.',
    )
    parser.add_argument('--mono3d_invalid_line_min_depth', type=float, default=1e-6)
    parser.add_argument('--mono3d_invalid_line_min_length', type=float, default=1e-8)
    parser.add_argument('--mono3d_invalid_line_max_abs_coordinate', type=float, default=1e6)
    parser.add_argument('--mono3d_trim_3d_lines_to_2d', action='store_true', default=False)
    parser.add_argument('--mono3d_record_3d_line_trimming', action='store_true', default=False)

class Monolines3D(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        split,
        train2d,
        use_image_normalized_target_line_coords,
        normalize_3d_line_space,
        do_not_normalize_images,
        preload,
        strict,
        invalid_line_filter='none',
        invalid_line_min_depth=1e-6,
        invalid_line_min_length=1e-8,
        invalid_line_max_abs_coordinate=1e6,
        trim_3d_lines_to_2d=False,
        record_3d_line_trimming=False,
        transforms=None,
        experiment_dir=None,
    ):
        self.split = split
        self.train2d = train2d
        self.use_image_normalized_target_line_coords = use_image_normalized_target_line_coords
        self.normalize_3d_line_space = normalize_3d_line_space
        self.do_not_normalize_images = do_not_normalize_images
        self.preload = preload
        self.transforms = transforms
        if invalid_line_filter not in {'none', 'lines', 'samples'}:
            raise ValueError(
                "invalid_line_filter must be one of 'none', 'lines', or 'samples', "
                f"got {invalid_line_filter!r}"
            )
        self.invalid_line_filter = invalid_line_filter
        self.invalid_line_min_depth = invalid_line_min_depth
        self.invalid_line_min_length = invalid_line_min_length
        self.invalid_line_max_abs_coordinate = invalid_line_max_abs_coordinate
        self.trim_3d_lines_to_2d = trim_3d_lines_to_2d
        self.record_3d_line_trimming = record_3d_line_trimming

        dataset_root = Path(root_dir)

        print("Looking for scenes in", dataset_root, "...")
        self.scenes = self.find_scene_roots(dataset_root)

        self.used_data = []
        self.entries = []

        # Optional: set args.mono3d_strict = True if you want missing files to crash
        self.strict = strict

        sample_id_counter = 0
        for scene_root in self.scenes:
            print('Loading scene', scene_root)
            images_dir = scene_root / "images"
            annotations_dir = scene_root / "limap_annotations"

            if not images_dir.exists():
                self._problem(f"Missing images dir: {images_dir}")
                continue

            try:
                annotation_source = self._get_annotation_source(scene_root)
            except Exception as e:
                self._problem(f"Cannot load annotations for: {scene_root} ({e})")
                continue

            scene_info = {
                "scene_root": str(scene_root),
                "images": self._dir_info(images_dir),
                "annotations": annotation_source.info(),
                "used_images": [],
                "missing_images": [],
                "bad_annotations": [],
                "filtered_lines": [],
                "trimmed_lines": [],
                "line_filtering": {
                    "mode": self.invalid_line_filter,
                    "thresholds": {
                        "min_depth": self.invalid_line_min_depth,
                        "min_length": self.invalid_line_min_length,
                        "max_abs_coordinate": self.invalid_line_max_abs_coordinate,
                    },
                    "summary": {
                        "affected_views": 0,
                        "skipped_lines": 0,
                        "skipped_views": 0,
                    },
                },
                "line_trimming": {
                    "enabled": self.trim_3d_lines_to_2d,
                    "record_line_details": self.record_3d_line_trimming,
                    "summary": {
                        "affected_views": 0,
                        "trimmed_lines": 0,
                        "unchanged_lines": 0,
                        "untrimmable_lines": 0,
                    },
                },
            }

            scene_image_counter = -1
            for rel_annotation_path, load_annotation, debug_annotation_path in annotation_source.iter_annotations():
                scene_image_counter += 1
                #rel_annotation_path = annotation_json_path.relative_to(annotations_dir)

                # Example:
                #   limap_annotations/subdir/img_001.jpg.json
                # maps to:
                #   images/subdir/img_001.jpg
                #rel_image_path = rel_annotation_path.with_suffix("")
                rel_image_path = rel_annotation_path.with_suffix("")

                image_path = images_dir / rel_image_path

                if not image_path.exists():
                    msg = (
                        f"Missing image for annotation:\n"
                        f"  annotation: {debug_annotation_path}\n"
                        f"  expected image: {image_path}"
                    )
                    scene_info["missing_images"].append(str(rel_image_path))
                    self._problem(msg)
                    continue

                if split == 'train' and scene_image_counter % 5 == 0:
                    continue
                elif split == 'val' and scene_image_counter % 5 != 0:
                    continue

                try:
                    annotation = load_annotation()
                except Exception as e:
                    scene_info["bad_annotations"].append(str(rel_annotation_path))
                    self._problem(f"Could not load annotation JSON: {debug_annotation_path} ({e})")
                    continue

                # This path should be directly loadable by cv2/PIL/etc.
                # resolve() is nice here because it also resolves images_dir symlink.
                loadable_image_path = image_path.resolve()

                try:
                    lines2d, lines3d, camera_K, filtering, trimming = self.resolve_target_lines(annotation)
                except Exception as e:
                    msg = f"Could not resolve line targets for {image_path} ({e})"
                    scene_info["bad_annotations"].append(str(rel_annotation_path))
                    self._problem(msg)
                    continue

                if filtering['skipped_lines']:
                    filtered_view = {
                        "annotation": str(rel_annotation_path),
                        "image": str(rel_image_path),
                        **filtering,
                    }
                    scene_info["filtered_lines"].append(filtered_view)
                    summary = scene_info["line_filtering"]["summary"]
                    summary["affected_views"] += 1
                    summary["skipped_lines"] += filtering['removed_line_count']
                    summary["skipped_views"] += int(filtering['sample_skipped'])

                if self.trim_3d_lines_to_2d:
                    summary = scene_info["line_trimming"]["summary"]
                    summary["trimmed_lines"] += trimming['trimmed_line_count']
                    summary["unchanged_lines"] += trimming['unchanged_line_count']
                    summary["untrimmable_lines"] += trimming['untrimmable_line_count']

                if trimming['trimmed_line_count'] or trimming['untrimmable_line_count']:
                    trimmed_view = {
                        "annotation": str(rel_annotation_path),
                        "image": str(rel_image_path),
                        "trimmed_line_count": trimming['trimmed_line_count'],
                        "unchanged_line_count": trimming['unchanged_line_count'],
                        "untrimmable_line_count": trimming['untrimmable_line_count'],
                    }
                    if self.record_3d_line_trimming:
                        trimmed_view["lines"] = trimming['lines']
                    scene_info["trimmed_lines"].append(trimmed_view)
                    summary = scene_info["line_trimming"]["summary"]
                    summary["affected_views"] += 1

                if filtering['sample_skipped']:
                    continue

                entry = {
                    "sample_id": sample_id_counter,
                    "image_path": str(loadable_image_path),
                    "target_lines2d": lines2d,
                    "target_lines3d": lines3d,
                    "camera_K": camera_K,
                }

                if self.preload:
                    print('Preloading image', loadable_image_path)
                    try:
                        img = self.load_image(loadable_image_path)
                        entry['img'] = img
                    except Exception as e:
                        msg = f"Could not load image {image_path} ({e})"
                        scene_info["missing_images"].append(str(rel_image_path))
                        self._problem(msg)
                        continue

                self.entries.append(entry)
                sample_id_counter += 1

                scene_info["used_images"].append(str(rel_image_path))

            self.used_data.append(scene_info)

        if self.normalize_3d_line_space:
            print('Computing normalization constants...')
            mean, std = self.line_space_normalization_constants()
            scene_info['line_space_normalization'] = {
                "mean": mean.tolist(),
                "std": std.tolist(),
            }
            print('Normalizing lines...')
            for i in range(len(self.entries)):
                self.entries[i]['target_lines3d'] = self.normalize_lines(self.entries[i]['target_lines3d'], std, mean).numpy()

        if experiment_dir is not None and os.path.isdir(experiment_dir):
            path = os.path.join(experiment_dir, f'scene_info_{split}.json')
            save_json(path, self.used_data)


        print('Split:', split)
        print(f"\tFound {len(self.scenes)} scenes")
        print(f"\tLoaded {len(self.entries)} image/annotation pairs")

    def normalize_lines(self, lines, std, mean):
        div = torch.tensor(std.tolist() + std.tolist())
        sub = torch.tensor(mean.tolist() + mean.tolist()) / div
        #print('sub:', sub)
        #print('div:', div)
        return (lines / div) - sub

    def line_space_normalization_constants(self):
        endpoints = []
        for entry in self.entries:
            lines = entry['target_lines3d']
            for line in lines:
                #print(line.shape)
                dim = line.shape[0] // 2
                endpoints.append(line[:dim])
                endpoints.append(line[dim:])
        endpoints = np.array(endpoints)
        mean = endpoints.mean(axis=0)
        std = endpoints.std(axis=0)
        return mean, std

    def _get_annotation_source(self, scene_root: Path):
        zip_path = scene_root / "limap_annotations.zip"
        annotations_dir = scene_root / "limap_annotations"

        if zip_path.is_file():
            return ZipAnnotationSource(zip_path)

        if annotations_dir.is_dir():
            return DirectoryAnnotationSource(annotations_dir)

        raise FileNotFoundError(
            f"Scene has neither limap_annotations.zip nor limap_annotations directory: {scene_root}"
        )

    def load_image(self, image_path):
        return Image.open(str(image_path)).convert("RGB")

    def _invalid_line_reasons(self, line2d, line3d):
        reasons = []
        if line2d.shape != (2, 2) or not np.isfinite(line2d).all():
            reasons.append({
                "code": "invalid_line2d",
                "shape": list(line2d.shape),
            })
        if line3d.shape != (2, 3) or not np.isfinite(line3d).all():
            reasons.append({
                "code": "invalid_camera_line3d",
                "shape": list(line3d.shape),
            })
            return reasons

        length = float(np.linalg.norm(line3d[1] - line3d[0]))
        if length <= self.invalid_line_min_length:
            reasons.append({
                "code": "degenerate_line3d",
                "length": length,
                "min_length": self.invalid_line_min_length,
            })

        invalid_depth_indices = np.flatnonzero(line3d[:, 2] <= self.invalid_line_min_depth)
        if len(invalid_depth_indices):
            reasons.append({
                "code": "nonpositive_depth",
                "endpoint_indices": invalid_depth_indices.tolist(),
                "depths": line3d[invalid_depth_indices, 2].tolist(),
                "min_depth": self.invalid_line_min_depth,
            })

        max_abs_coordinate = float(np.abs(line3d).max())
        if max_abs_coordinate > self.invalid_line_max_abs_coordinate:
            reasons.append({
                "code": "extreme_camera_coordinate",
                "max_abs_coordinate": max_abs_coordinate,
                "limit": self.invalid_line_max_abs_coordinate,
            })
        return reasons

    def _trim_3d_line_to_2d(self, line2d, line3d, camera_K):
        projected_h = (camera_K @ line3d.T).T
        projected = projected_h[:, :2] / projected_h[:, 2:3]
        projected_delta = projected[1] - projected[0]
        projected_length_squared = float(projected_delta @ projected_delta)
        if not np.isfinite(projected_length_squared) or projected_length_squared <= 1e-12:
            return line3d, {
                "status": "untrimmable",
                "reason": "degenerate_projected_line",
            }

        segment_parameters = np.array([
            np.clip(
                float((endpoint - projected[0]) @ projected_delta) / projected_length_squared,
                0.0,
                1.0,
            )
            for endpoint in line2d
        ], dtype=np.float64)
        closest_projected = projected[0] + segment_parameters[:, None] * projected_delta

        # Image interpolation is not affine in 3D under perspective projection.
        # Convert each image-segment parameter to its matching parameter on X(t).
        z0, z1 = (float(line3d[0, 2]), float(line3d[1, 2]))
        denominators = (1.0 - segment_parameters) * z1 + segment_parameters * z0
        if not np.isfinite(denominators).all() or np.any(np.abs(denominators) <= 1e-12):
            return line3d, {
                "status": "untrimmable",
                "reason": "invalid_perspective_parameter",
                "segment_parameters": segment_parameters.tolist(),
            }

        line_parameters = segment_parameters * z0 / denominators
        trimmed = line3d[0] + line_parameters[:, None] * (line3d[1] - line3d[0])
        endpoint_errors = np.linalg.norm(closest_projected - line2d, axis=1)
        trimmed_length = float(np.linalg.norm(trimmed[1] - trimmed[0]))
        if trimmed_length <= getattr(self, 'invalid_line_min_length', 1e-8):
            return line3d, {
                "status": "untrimmable",
                "reason": "collapsed_trim",
                "segment_parameters": segment_parameters.tolist(),
                "line_parameters": line_parameters.tolist(),
                "endpoint_reprojection_distances": endpoint_errors.tolist(),
            }

        endpoint_movements = np.array([
            min(np.linalg.norm(point - line3d[0]), np.linalg.norm(point - line3d[1]))
            for point in trimmed
        ])
        changed = not np.allclose(np.sort(line_parameters), [0.0, 1.0], atol=1e-8)
        return trimmed.astype(np.float32), {
            "status": "trimmed" if changed else "unchanged",
            "segment_parameters": segment_parameters.tolist(),
            "line_parameters": line_parameters.tolist(),
            "endpoint_3d_movements": endpoint_movements.tolist(),
            "endpoint_reprojection_distances": endpoint_errors.tolist(),
        }

    def resolve_target_lines(
        self,
        annotation,
    ) -> Tuple[npt.NDArray[any], npt.NDArray[any], npt.NDArray[any], dict, dict] | None:
        camera_K = np.array(annotation['camera']['K'], dtype=np.float32)

        if 'from_limap_tracks' in annotation['lines']:
            lines = annotation['lines']['from_limap_tracks']
            lines2d = []
            lines3d = []
            skipped_lines = []
            trimming_lines = []
            trimming_counts = {
                "trimmed": 0,
                "unchanged": 0,
                "untrimmable": 0,
            }
            for line_index, line in enumerate(lines):
                line2d = np.asarray(line['line2d'], dtype=np.float32)
                line3d = np.asarray(line['camera']['track3d_trimmed'], dtype=np.float32)

                if self.invalid_line_filter != 'none':
                    reasons = self._invalid_line_reasons(line2d, line3d)
                    if reasons:
                        skipped_lines.append({
                            "line_index": line_index,
                            "reasons": reasons,
                        })
                        continue

                if self.trim_3d_lines_to_2d:
                    line3d, trimming = self._trim_3d_line_to_2d(line2d, line3d, camera_K)
                    trimming_counts[trimming['status']] += 1
                    if self.record_3d_line_trimming:
                        trimming_lines.append({
                            "line_index": line_index,
                            **trimming,
                        })

                    if trimming['status'] != 'untrimmable' and self.invalid_line_filter != 'none':
                        reasons = self._invalid_line_reasons(line2d, line3d)
                        if reasons:
                            skipped_lines.append({
                                "line_index": line_index,
                                "stage": "after_trimming",
                                "reasons": reasons,
                            })
                            continue

                if not self.trim_3d_lines_to_2d or trimming['status'] == 'untrimmable':
                    projected = (camera_K @ line3d.T).T
                    projected = projected[:, :2] / projected[:, 2:3]
                    direct_error = np.abs(projected - line2d).sum()
                    swapped_error = np.abs(projected[::-1] - line2d).sum()
                    if swapped_error < direct_error:
                        line3d = line3d[::-1].copy()

                # Keep pixel coordinates through geometric transforms. Normalize once
                # in the shared final Normalize transform.
                lines2d.append(np.concatenate([line2d[0], line2d[1]]))
                lines3d.append(np.concatenate([line3d[0], line3d[1]]))

            sample_skipped = bool(skipped_lines) and self.invalid_line_filter == 'samples'
            sample_skip_reason = 'invalid_lines' if sample_skipped else None
            if not lines2d and self.invalid_line_filter != 'none':
                sample_skipped = True
                sample_skip_reason = 'no_valid_lines_after_filtering'

            filtering = {
                "original_line_count": len(lines),
                "valid_line_count": len(lines2d),
                "kept_line_count": 0 if sample_skipped else len(lines2d),
                "removed_line_count": len(lines) if sample_skipped else len(skipped_lines),
                "skipped_lines": skipped_lines,
                "sample_skipped": sample_skipped,
                "sample_skip_reason": sample_skip_reason,
            }
            trimming = {
                "trimmed_line_count": trimming_counts['trimmed'],
                "unchanged_line_count": trimming_counts['unchanged'],
                "untrimmable_line_count": trimming_counts['untrimmable'],
                "lines": trimming_lines,
            }
            return (
                np.asarray(lines2d, dtype=np.float32).reshape(-1, 4),
                np.asarray(lines3d, dtype=np.float32).reshape(-1, 6),
                camera_K,
                filtering,
                trimming,
            )
        raise Exception('Didnt find usable lines type')

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        if self.preload:
            img = entry['img']
        else:
            img = self.load_image(entry['image_path'])

        w, h = img.size
        target = {}
        lines2d = entry['target_lines2d'].reshape(-1, 4)
        lines3d = entry['target_lines3d'].reshape(-1, 6)
        target['image_id'] = np.array([entry['sample_id']])
        target['labels'] = np.array([0 for _ in lines2d], dtype=np.int64)
        target['area'] = np.array([1 for _ in lines2d])
        target['iscrowd'] = np.array([0 for _ in lines2d])
        target['lines'] = lines2d.astype(np.float32)
        target['lines3d'] = lines3d.astype(np.float32)
        target['camera_K'] = entry['camera_K'].astype(np.float32)
        target['orig_size'] = np.array([h, w])
        target['size'] = np.array([h, w])

        target = {k: torch.from_numpy(v) for k, v in target.items()}

        target['image_path'] = entry['image_path']

        if self.transforms is not None:
            #img = Image.fromarray(img*255)
            img, target = self.transforms(img, target)

        return img, target

    def find_scene_roots(self, dataset_root: str | Path) -> list[Path]:
        dataset_root = Path(dataset_root)

        scene_roots = []

        for dirpath, dirnames, filenames in os.walk(dataset_root):
            if ".scene_root" in filenames:
                print("Found scene at:", dirpath)
                scene_roots.append(Path(dirpath))

                # Do not search inside this scene further
                dirnames.clear()

        return scene_roots

    def _dir_info(self, path: Path):
        """
        Returns either:
            "regular_dir"

        or:
            {"symlink": "/absolute/resolved/symlink/path"}
        """
        if path.is_symlink():
            return {
                "symlink": str(path.resolve(strict=True))
            }

        return "regular_dir"

    def _problem(self, msg: str):
        if self.strict:
            raise RuntimeError(msg)
        else:
            warnings.warn(msg)


def make_coco_transforms(image_set, args=None):
    ts = [
        T.ToTensor(),
    ]
    if args.mono3d_do_not_normalize_images is False:
        ts.append(T.Normalize([0.538, 0.494, 0.453], [0.257, 0.263, 0.273]))
    else:
        ts.append(T.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
    normalize = T.Compose(ts)



    # update args from config files
    scales = args.data_aug_scales
    #max_size = args.data_aug_max_size
    #scales2_resize = args.data_aug_scales2_resize
    #scales2_crop = args.data_aug_scales2_crop
    #test_size = args.eval_spatial_size

    if image_set == 'train':
        if scales is None:
            return normalize
        else:

            max_size = args.data_aug_max_size
            return T.Compose([T.RandomResize(scales, max_size=max_size), normalize])

        return T.Compose([
            T.RandomSelect(
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                ),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales2_resize),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            T.ColorJitter(),
            normalize,
        ])

    if image_set in ['val', 'test']:
        test_size = args.eval_spatial_size
        if test_size is None:
            return normalize
        else:
            max_size = args.data_aug_max_size
            return T.Compose([
                T.RandomResize([test_size], max_size=max_size),
                normalize,
            ])



    raise ValueError(f'unknown {image_set}')

def build_mono3d_from_conf(conf):
    return Monolines3D(**conf)


def build_mono3d_from_args(image_set, args, experiment_dir=None):

    transforms = make_coco_transforms(image_set, args)

    conf = {
        "root_dir": args.mono3d_dataset_root,
        "split": image_set,
        "train2d": args.mono3d_train2d,
        "use_image_normalized_target_line_coords": args.mono3d_use_image_normalized_target_line_coords,
        "normalize_3d_line_space": args.mono3d_normalize_3d_line_space,
        "preload": args.mono3d_preload_images,
        "strict": args.mono3d_strict,
        "do_not_normalize_images": args.mono3d_do_not_normalize_images,
        "invalid_line_filter": getattr(args, 'mono3d_invalid_line_filter', 'none'),
        "invalid_line_min_depth": getattr(args, 'mono3d_invalid_line_min_depth', 1e-6),
        "invalid_line_min_length": getattr(args, 'mono3d_invalid_line_min_length', 1e-8),
        "invalid_line_max_abs_coordinate": getattr(
            args,
            'mono3d_invalid_line_max_abs_coordinate',
            1e6,
        ),
        "trim_3d_lines_to_2d": getattr(args, 'mono3d_trim_3d_lines_to_2d', False),
        "record_3d_line_trimming": getattr(args, 'mono3d_record_3d_line_trimming', False),
        "transforms": transforms,
        "experiment_dir": experiment_dir,
    }
    return Monolines3D(**conf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    build_args(parser, required=True)
    args = parser.parse_args()

    print('Creating Monolines3D dataset')
    dataset = Monolines3D(args, 'train', train2d=True, preload=True)
