# Benchmarks

Official benchmark runs are configured under `configs/benchmarks/` and launched
with `tools/run_benchmark.py`. The runner does not contain model logic; it
materializes regular YAML files and calls the public train/evaluate entrypoints.
When the benchmark YAML has a `prepare` section, `--run-all` also performs the
configured data stages before training.

Cross-dataset transfer SOTA:

```bash
python tools/run_benchmark.py \
  --config configs/benchmarks/cross_dataset_transfer_sota.yaml
```

For a non-training check:

```bash
python tools/run_benchmark.py \
  --config configs/benchmarks/cross_dataset_transfer_sota.yaml \
  --check-inputs \
  --generate \
  --include-controls \
  --dry-run
```

This benchmark trains the source checkpoint from scratch, fine-tunes each target
fold/seed from that checkpoint, evaluates the strict segment FrameAcc metric,
and writes `summary.json` under the benchmark output root.

The cross-dataset SOTA config is a full pipeline config:

1. EgoHumans source prepare: raw/extracted EgoHumans arrays -> source cache.
2. Custom target prepare: raw custom video/IMU annotations -> video manifest.
3. Custom skeleton extraction: manifest videos -> `skeleton.json`.
4. Custom segment packing: preprocessed IMU/bboxes + `skeleton.json` ->
   `custom_segments/sequences/custom_*.npz`.
5. Custom slicing: segment NPZs + custom IMU split files -> fold caches.
6. Source training, target transfer/control training, evaluation, summary.

All stage paths and scheduling defaults live in
`configs/benchmarks/cross_dataset_transfer_sota.yaml`, including GPUs,
parallelism, controls, and skip policy. If source cache, target segment NPZs,
or target fold caches already exist and `skip_existing` is true in the config,
the prepare stage reuses them.
