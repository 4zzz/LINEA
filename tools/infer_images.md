# Image Inference

`tools/infer_images.py` runs a checkpoint on explicitly selected images without
building or scanning the training dataset.

Infer one image:

```bash
venv314/bin/python tools/infer_images.py \
  --checkpoint output/experiment/checkpoint0009.pth \
  --image path/to/image.jpg
```

Infer several images and save PNG overlays:

```bash
venv314/bin/python tools/infer_images.py \
  --checkpoint output/experiment/checkpoint0009.pth \
  --image first.jpg second.jpg \
  --image third.jpg \
  --save-visualizations
```

Paths can also be positional, or supplied through one or more newline-delimited
files. Blank lines and lines starting with `#` are ignored:

```bash
venv314/bin/python tools/infer_images.py \
  --checkpoint output/experiment/checkpoint0009.pth \
  --image-list images.txt \
  --output selected_images.json.gz
```

By default, output is saved to
`<checkpoint-dir>/inference/<checkpoint-name>_images/predictions.json.gz`.

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
