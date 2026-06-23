#!/usr/bin/env bash
# Experiment Note: A1-run-full-repro
# One-click reproduction of MoBInd official evaluation on EgoHumans.
# Assumes conda env `mobind_repro` exists and symlink
# /data/lyxie/ReID/Data/egohumans/EgoHumans -> /data/lyxie/ReID/Data/egohumans is in place.

set -e

conda activate mobind_repro
cd /home/fzliang/MoBind

# Build caches
echo "[1/4] Building contrastive cache..."
python preprocess/EgoHumans/cache.py --window_sec 5 --stride 2

echo "[2/4] Building multi-person cache..."
python preprocess/EgoHumans/cache_multi_person.py --window_sec 5 --stride 2

echo "[3/4] Building sync cache..."
python preprocess/EgoHumans/cache_sync.py \
  --window_sec 20 --stride_sec 5 \
  --anno_file /home/fzliang/MoBind/data/EgoHumans/cache_sync_action_20_5/annotations.txt

# Run evaluations
echo "[4/4] Running evaluations..."
python eval_retrieval.py --exp_dir ./checkpoints/EgoHumans/stage2_repro
python eval_localization.py --exp_path ./checkpoints/EgoHumans/stage2_repro --task all
python eval_sync_egoh.py --exp_dir ./checkpoints/EgoHumans/stage2_repro --task person
python eval_sync_egoh.py --exp_dir ./checkpoints/EgoHumans/stage2_repro --task video
