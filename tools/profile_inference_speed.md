# Inference Speed Profiling

`tools/profile_inference_speed.py` benchmarks a checkpoint directly, using the
model configuration stored inside it. It measures warm steady-state model
execution and excludes checkpoint loading, image decoding, preprocessing, file
I/O, and visualization.

Basic FP32 benchmark:

```bash
venv314/bin/python tools/profile_inference_speed.py \
  --checkpoint output/experiment/checkpoint.pth \
  --input-size 640 \
  --batch-size 1 \
  --warmup 20 \
  --iterations 100
```

Benchmark CUDA FP16 autocast and save structured results:

```bash
venv314/bin/python tools/profile_inference_speed.py \
  --checkpoint output/experiment/checkpoint.pth \
  --input-size 640 \
  --batch-size 1 \
  --warmup 20 \
  --iterations 100 \
  --amp \
  --json-output output/experiment/profiling/inference_amp.json
```

The report includes synchronized total, backbone, encoder, decoder, and
postprocessor latency; mean, median, p90, p95, minimum, and maximum timings;
throughput; and peak allocated CUDA memory. Synthetic camera intrinsics are
provided for UV-depth checkpoints so their normal 3D decoding path is measured.

For fair comparisons:

- Use the same GPU with no competing workload.
- Keep input resolution, batch size, AMP, and channels-last settings identical.
- Compare medians and p95 values, not only minimum latency.
- Run batch size 1 for interactive latency and larger batches separately for
  maximum throughput.
- Benchmark preprocessing and image loading separately when end-to-end
  application latency matters.
