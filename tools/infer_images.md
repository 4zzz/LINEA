# Image Inference

`tools/infer_images.py` runs a checkpoint on explicitly selected images without
building or scanning the training dataset.

Infer one image:

```bash
venv314/bin/python tools/infer_images.py \
  --checkpoint output/experiment/checkpoint0009.pth \
  --image path/to/image.jpg \
  --prediction-record \
  --glb-model \
  --output-dir output/image_inference
```

For a compact JSON containing only thresholded line arrays and scores, use:

```bash
venv314/bin/python tools/infer_images.py \
  --checkpoint output/experiment/checkpoint0009.pth \
  --image path/to/image.jpg \
  --simple-json \
  --output-dir output/image_inference
```

This writes `001-image/prediction.json`. The file contains parallel `scores`
and `lines2d` arrays, plus `lines3d` when the checkpoint predicts 3D lines.

Infer several images and save PNG overlays:

```bash
venv314/bin/python tools/infer_images.py \
  --checkpoint output/experiment/checkpoint0009.pth \
  --image first.jpg second.jpg \
  --image third.jpg \
  --prediction-record \
  --lines-2d-png \
  --output-dir output/image_inference
```

Paths can also be positional, or supplied through one or more newline-delimited
files. Blank lines and lines starting with `#` are ignored:

```bash
venv314/bin/python tools/infer_images.py \
  --checkpoint output/experiment/checkpoint0009.pth \
  --image-list images.txt \
  --single-prediction-record selected_images.json.gz
```

`--prediction-record` and `--single-prediction-record` are mutually exclusive.
The former creates one directory per input image:

```text
output/image_inference/
  001-first/
    prediction.json.gz
    model.glb
    visualization.png
  002-second/
    prediction.json.gz
    model.glb
    visualization.png
```

`--prediction-record`, `--simple-json`, `--glb-model`, and `--lines-2d-png` require
`--output-dir`. GLB models are available only for LINEA3D checkpoints. They
contain thresholded predictions as native line-segment edges in the model's
camera-coordinate space.

The script uses the same model, data-loading, inference, and output options as
`infer_dataset.py`. Use `--prediction-record-backend json`, `json.gz`, or `h5`
to select a record backend. Combined HDF5 files stream records to disk.
`--prediction-record-save-exact-sample` includes the transformed and padded
input tensor; it is omitted by default. Records preserve camera metadata,
filtered line depths, and Git commit/diff metadata.

The previous spellings remain aliases: `--prediction-files`,
`--single-prediction-file` (also `--output`, `-p`, and `-o`), `--glb-models`,
`--simple-json-files`, `--save-visualizations`, and `--save-input`.
Every long option accepts both hyphens and underscores. PNG output now follows
the shared validation and requires `--output-dir`, including when a custom
`--visualization-dir` is supplied.

Per-image record names default to `prediction.json.gz`. If JSON records and
compact JSON are both selected, the compact output uses `prediction_simple.json`
to avoid overwriting the record at `prediction.json`.
Matching options are unavailable because explicit images have no ground-truth
line targets. Image coordinates in PNG and JSON outputs account for input padding.

## Camera intrinsics

An `uv_depth` LINEA3D checkpoint requires camera intrinsics. Supply one matrix
for all input images using either four parameters (`fx fy cx cy`) or all nine
row-major matrix values:

```bash
--camera-k 1000 1000 640 360
```

For varying cameras, use `--camera-k-json cameras.json`. It can contain one
matrix:

```json
[[1000, 0, 640], [0, 1000, 360], [0, 0, 1]]
```

Or a mapping keyed by absolute path or basename:

```json
{
  "first.jpg": {"K": [[1000, 0, 640], [0, 1000, 360], [0, 0, 1]]},
  "second.jpg": {"camera": {"K": [[900, 0, 640], [0, 900, 360], [0, 0, 1]]}}
}
```

As an explicit fallback, the script can try to estimate intrinsics from each
image's EXIF metadata:

```bash
--camera-k-from-exif
```

The preferred EXIF path uses `FocalLength`, `FocalPlaneXResolution`, and
`FocalPlaneResolutionUnit` to estimate sensor size. If those tags are
incomplete, `FocalLengthIn35mmFilm` is used to estimate focal length in pixels
from the image diagonal. Images without either usable metadata path are
reported, and UV-depth inference stops rather than inventing a focal length.

EXIF normally does not contain a complete calibrated intrinsic matrix. The
estimate assumes the principal point is the image center, zero skew, square
pixels where needed, and ignores lens distortion. The derivation method, source
tags, assumptions, and original estimated matrix are saved in each prediction
record under `camera_K_metadata`.

Affine fitting is intentionally unavailable because arbitrary images do not
provide matched ground-truth 3D lines from which to estimate scale and shift.
