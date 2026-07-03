#!/usr/bin/env bash
# Experiment Note: E1-A1-build-egohumans-cache
# Build EgoHumans MoBInd cache compatible with custom w24 architecture.
# Custom w24: window_sec=0.8, patch_sec=0.2, srate=30 -> 24 frames, 4 patches, patch_size=6.
# EgoHumans:  srate=20, so we use window_sec=1.2, patch_sec=0.3 -> 24 frames, 4 patches, patch_size=6.
set -euo pipefail

REPO=/home/fzliang/Autism-project
E1_DIR="$REPO/experiments/G5:cross_dataset_transfer/E1:egohumans_dual_embedding_pretrain"
MOBIND_ROOT=/home/fzliang/MoBind

WINDOW_SEC=1.2
STRIDE_SEC=0.5
SPLIT=action

echo "=== Building EgoHumans cache: window=${WINDOW_SEC}s, stride=${STRIDE_SEC}s, split=${SPLIT} ==="
echo "This cache is architecture-compatible with custom w24 (24 frames, 4 patches, patch_size=6)."

conda run --no-capture-output -n mobind_repro python "$MOBIND_ROOT/preprocess/EgoHumans/cache.py" \
  --window_sec "$WINDOW_SEC" \
  --stride "$STRIDE_SEC" \
  --split "$SPLIT" \
  > "$E1_DIR/results/A1_build_cache.log" 2>&1

echo "=== Cache build complete. Log: $E1_DIR/results/A1_build_cache.log ==="
