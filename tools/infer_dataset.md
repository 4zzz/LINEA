# Dataset Inference

`tools/infer_dataset.py` runs a checkpoint on its configured dataset. The dataset
type and loading settings come from the checkpoint; use `--dataset-arg KEY=VALUE`
to override dataset settings for inference.

## Monolines3D dataset structure

By default, the loader recursively searches `mono3d_dataset_root` for scenes
marked by a `.scene_root` file (which can be empty). Each scene contains images
and corresponding annotations:

```text
dataset_root/
|-- scene_001/
|   |-- .scene_root
|   |-- images/
|   |   |-- image_001.jpg
|   |   `-- subdirectory/
|   |       `-- image_002.jpg
|   `-- limap_annotations/
|       |-- image_001.jpg.json
|       `-- subdirectory/
|           `-- image_002.jpg.json
`-- scene_002/
    |-- .scene_root
    |-- images/
    |   `-- image_003.jpg
    `-- limap_annotations/
        `-- image_003.jpg.json
```

Annotation paths must mirror image paths, with `.json` appended to the complete
image filename. For example, `limap_annotations/subdirectory/image_002.jpg.json`
maps to `images/subdirectory/image_002.jpg`.

A scene may contain `limap_annotations.zip` instead of the `limap_annotations/`
directory. The ZIP takes precedence if both exist. ZIP members can be relative
annotation paths such as `subdirectory/image_002.jpg.json`, or include the
`limap_annotations/` prefix. Images remain outside the ZIP.

The annotation JSON contains `camera.K` (the 3 x 3 intrinsic matrix). Ground-truth
lines, when present, are loaded from `lines.from_limap_tracks`, with each line's
`line2d` and `camera.track3d_trimmed` coordinates. For images with only camera
intrinsics, see [infer_annotated_images.md](infer_annotated_images.md).

The layout can be customized with these dataset overrides:

| Setting | Default | Purpose |
| --- | --- | --- |
| `mono3d_dataset_root` | From checkpoint | Directory to search for scenes |
| `mono3d_single_scene_root` | `False` | Treat the root itself as one scene; no marker is needed |
| `mono3d_scene_images_dir` | `images` | Image directory within each scene |
| `mono3d_scene_annotations_dir` | `limap_annotations` | Annotation directory within each scene |
| `mono3d_scene_annotations_file` | `limap_annotations.zip` | Annotation ZIP within each scene |

`--split test` (the default) loads all eligible pairs. For Monolines3D, `val`
selects every fifth annotation within each scene, starting with the first;
`train` selects the remaining annotations. These splits depend on annotation
iteration order rather than separate split directories.

## Run inference

Run from the repository root using your Python environment and a Monolines3D
checkpoint:

```bash
python tools/infer_dataset.py \
  --checkpoint output/experiment/checkpoint0009.pth \
  --dataset-arg mono3d_dataset_root=/absolute/path/to/dataset_root \
  --split test \
  --prediction-record \
  --simple-json \
  --glb-model \
  --lines-2d-png \
  --output-dir output/dataset_inference \
  --output-naming seq_flat
```

Select the outputs you need:

| Option | Output |
| --- | --- |
| `--prediction-record` | One detailed prediction record per sample; default format is `.json.gz` |
| `--simple-json` | Compact JSON with scores and line arrays |
| `--glb-model` | 3D line model in GLB format; requires 3D predictions |
| `--lines-2d-png` | Image with predicted and available ground-truth 2D lines |
| `--single-prediction-record PATH` | One combined record containing all samples |

At least one output option is required. All per-sample outputs require
`--output-dir`. `--prediction-record` and `--single-prediction-record` are
mutually exclusive. Use `--prediction-record-backend json`, `json.gz`, or `h5`
to select the record format.

## Output naming

The following examples show all four per-sample outputs enabled, using the
default prediction-record backend. Only selected outputs are written.

### `seq_flat` (default)

Files are placed directly in the output directory. Each sample receives a
sequential base name, starting at `eval_000`:

```text
output/dataset_inference/
|-- eval_000.json.gz
|-- eval_000_simple.json
|-- eval_000.glb
|-- eval_000.png
|-- eval_001.json.gz
|-- eval_001_simple.json
|-- eval_001.glb
`-- eval_001.png
```

The number follows dataset iteration order, not the image filename.

### `images_structure`

Paths mirror the image paths relative to `--dataset-root`. The complete image
filename is retained and `_infer` is appended before each output suffix:

```text
output/dataset_inference/
`-- scene_001/
    `-- images/
        |-- image_001.jpg_infer.json.gz
        |-- image_001.jpg_infer_simple.json
        |-- image_001.jpg_infer.glb
        |-- image_001.jpg_infer.png
        `-- subdirectory/
            |-- image_002.jpg_infer.json.gz
            |-- image_002.jpg_infer_simple.json
            |-- image_002.jpg_infer.glb
            `-- image_002.jpg_infer.png
```

To use this mode, replace `--output-naming seq_flat` in the command above with:

```bash
--output-naming images_structure \
--dataset-root /absolute/path/to/dataset_root
```

`--dataset-root` controls output paths; it does not change the dataset loading
root. Set `mono3d_dataset_root` through `--dataset-arg` to change what is loaded.
All resolved image paths must be inside `--dataset-root`, including when image
directories are symlinks.

Both naming modes affect per-sample files. A combined record is saved at the
path supplied to `--single-prediction-record`.

## Other useful options

Use `--max-samples 10` to limit inference, `--pred-threshold 0.5` to retain
predictions with scores greater than 0.5, and `--device cpu` to run on the CPU.
The default device is `cuda`, with automatic CPU fallback when CUDA is
unavailable. `--batch-size` and `--num-workers` control data loading.

See all options with:

```bash
python tools/infer_dataset.py --help
```
