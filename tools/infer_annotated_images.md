# Annotated Image Inference

`tools/infer_annotated_images.py` runs a Monolines3D checkpoint on images paired
with JSON files containing camera intrinsics.

## Input directory

Pass the input directory with `--input-dir`. Subdirectories are searched
recursively:

```text
input_directory/
|-- image_001.jpg
|-- image_001.jpg.json
`-- subdirectory/
    |-- image_002.jpg
    `-- image_002.jpg.json
```

Place each JSON file next to its image and name it using the complete image
filename followed by `.json`. For example, `image_001.jpg` pairs with
`image_001.jpg.json`.

## JSON format

The JSON file only needs `camera.K`, a 3 x 3 camera intrinsic matrix:

```json
{
  "camera": {
    "K": [
      [1000.0, 0.0, 640.0],
      [0.0, 1000.0, 360.0],
      [0.0, 0.0, 1.0]
    ]
  }
}
```

Replace these example values with the intrinsics for the corresponding image.
The matrix has the form `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`, where `fx` and
`fy` are focal lengths in pixels and `(cx, cy)` is the principal point in the
original image's pixel coordinates. No line annotations are required.

## Run inference

Run from the repository root using your Python environment:

```bash
python tools/infer_annotated_images.py \
  --checkpoint output/experiment/checkpoint0009.pth \
  --input-dir input_directory \
  --simple-json \
  --lines-2d-png \
  --output-dir output/annotated_image_inference
```

This saves predicted lines as compact JSON and PNG overlays, preserving the
input's relative directory structure in the output. At least one output option
is required. Per-image outputs require `--output-dir`.

Use `--max-samples 10` to limit the number of images, `--pred-threshold 0.5` to
filter predictions by confidence, or `--device cpu` to run on the CPU (the
default is `cuda`). See all options with:

```bash
python tools/infer_annotated_images.py --help
```
