import os
import argparse
from pathlib import Path
import warnings
import json
import zipfile

import numpy as np
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
    parser.add_argument('--mono3d_use_image_normalized_target_line_coords', action='store_true', default=False)
    parser.add_argument('--mono3d_normalize_line_space', action='store_true', default=False)
    parser.add_argument('--mono3d_do_not_normalize_images', action='store_true', default=False)
    parser.add_argument('--mono3d_train2d', action='store_true', default=False)
    parser.add_argument('--mono3d_strict', action='store_true', default=False)

class Monolines3D(torch.utils.data.Dataset):
    def __init__(self, root_dir, split, train2d, use_image_normalized_target_line_coords, normalize_line_space, do_not_normalize_images, preload, strict, transforms=None, experiment_dir=None):
        self.split = split
        self.train2d = train2d
        self.use_image_normalized_target_line_coords = use_image_normalized_target_line_coords
        self.normalize_line_space = normalize_line_space
        self.do_not_normalize_images = do_not_normalize_images
        self.preload = preload
        self.transforms = transforms

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
                     line_target = self.resolve_target_lines(annotation)
                except Exception as e:
                    msg = f"Could not resolve line targets for {image_path} ({e})"
                    scene_info["bad_annotations"].append(str(rel_annotation_path))
                    self._problem(msg)
                    continue

                entry = {
                    "sample_id": sample_id_counter,
                    "image_path": str(loadable_image_path),
                    #"annotation": annotation,
                    "target_lines": line_target,
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

        if self.normalize_line_space:
            print('Computing normalization constants...')
            mean, std = self.line_space_normalization_constants()
            scene_info['line_space_normalization'] = {
                "mean": mean.tolist(),
                "std": std.tolist(),
            }
            print('Normalizing lines...')
            for i in range(len(self.entries)):
                self.entries[i]['target_lines'] = self.normalize_lines(self.entries[i]['target_lines'], std, mean).numpy()

        if experiment_dir is not None and os.path.isdir(experiment_dir):
            path = os.path.join(experiment_dir, f'scene_info_{split}.json')
            save_json(path, scene_info)


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
            lines = entry['target_lines']
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

    def resolve_target_lines(self, annotation) -> torch.Tensor | None:
        img_size = np.array([annotation['img_file_width'], annotation['img_file_height']])

        if 'from_limap_tracks' in annotation['lines']:
            lines = annotation['lines']['from_limap_tracks']
            target = []
            for line in lines:
                if self.train2d:
                    line2d = np.array(line['line2d'])

                    if self.use_image_normalized_target_line_coords:
                        img_size = np.array([annotation['img_file_width'], annotation['img_file_height']])
                        l = np.concatenate([line2d[0]/img_size, line2d[1]/img_size])
                    else:
                        l = np.concatenate([line2d[0], line2d[1]])
                    target.append(l)
                else:
                    line3d = np.array(line['camera']['track3d_trimmed'])
                    target.append(np.concatenate([line3d[0], line3d[1]]))
            return np.array(target)
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
        line_dim = 4 if self.train2d else 6
        lines = entry['target_lines'].reshape(-1, line_dim)#[[0, 2]]
        target['image_id'] = np.array([entry['sample_id']])
        target['labels'] = np.array([0 for _ in lines], dtype=np.int64)
        target['area'] = np.array([1 for _ in lines])
        target['iscrowd'] = np.array([0 for _ in lines])
        target['lines'] = lines.astype(np.float32)
        target['orig_size'] = np.array([h, w])
        target['size'] = np.array([h, w])

        target = {k: torch.from_numpy(v) for k, v in target.items()}

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

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.538, 0.494, 0.453], [0.257, 0.263, 0.273])
    ])

    # update args from config files
    scales = args.data_aug_scales
    max_size = args.data_aug_max_size
    scales2_resize = args.data_aug_scales2_resize
    scales2_crop = args.data_aug_scales2_crop
    test_size = args.eval_spatial_size

    if image_set == 'train':
        return T.Compose([T.RandomResize(scales, max_size=max_size), normalize])
        return normalize
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
        return T.Compose([
            T.RandomResize([test_size], max_size=max_size),
            normalize,
        ])



    raise ValueError(f'unknown {image_set}')

def build_mono3d_from_conf(conf):
    return Monolines3D(**conf)


def build_mono3d_from_args(image_set, args):

    transforms = make_coco_transforms(image_set, args)

    conf = {
        "root_dir": args.mono3d_dataset_root,
        "split": image_set,
        "train2d": args.mono3d_train2d,
        "use_image_normalized_target_line_coords": args.mono3d_use_image_normalized_target_line_coords,
        "normalize_line_space": args.mono3d_normalize_line_space,
        "preload": args.mono3d_preload_images,
        "strict": args.mono3d_strict,
        "do_not_normalize_images": args.mono3d_do_not_normalize_images,
        "transforms": transforms,
        "experiment_dir": args.output_dir,
    }
    return Monolines3D(**conf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    build_args(parser, required=True)
    args = parser.parse_args()

    print('Creating Monolines3D dataset')
    dataset = Monolines3D(args, 'train', train2d=True, preload=True)
