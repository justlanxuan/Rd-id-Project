#!/bin/bash
# Experiment Note: A2-train-ours
# Train our 4-IMU pipeline with a single right-wrist IMU repeated 4 times
# on the MoBInd-aligned 24-frame-window split.
set -e

cd /home/fzliang/Autism-project

conda run -n mobind_repro python -m src.pipelines \
  --config experiments/G_egohumans/E6:fair_single_imu_same_window/config/egohumans_mobind_aligned_single_imu_24.yaml \
  --stages train
