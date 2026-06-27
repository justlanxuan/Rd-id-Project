#!/bin/bash
# 运行自定义数据集的 4-fold 训练和评估管道

set -e

cd /home/fzliang/Autism-project
export PYTHONPATH=/home/fzliang/Autism-project:$PYTHONPATH

PYTHON=/home/fzliang/miniconda3/envs/autism_test/bin/python

# 配置
DATA_ROOT="data/interim/custom_complete/slice"
FOLDS_DIR="data/interim/custom_complete/slice/folds_3train_valtest"

echo "=========================================="
echo "Starting 4-fold IMU-guided Legacy Training"
echo "=========================================="

# 对每个fold运行训练和评估
FOLD_NUM=0
for FOLD_DIR in "$FOLDS_DIR"/fold*/; do
  FOLD_NUM=$((FOLD_NUM + 1))
  FOLD_NAME=$(basename "$FOLD_DIR")
  echo ""
  echo "=========================================="
  echo "Processing Fold $FOLD_NUM: $FOLD_NAME"
  echo "=========================================="
  
  RUN_NAME="custom_imu_guided_legacy_${FOLD_NAME}_e15"
  
  echo "[1/3] Training..."
  $PYTHON experiments/imu_guided_video_encoder/run_training.py \
    --data_root "$DATA_ROOT" \
    --train_csv "$FOLD_DIR/windows_train.csv" \
    --val_csv "$FOLD_DIR/windows_val.csv" \
    --motionbert_root /home/fzliang/origin/MotionBERT \
    --motionbert_config configs/pose3d/MB_ft_h36m_global_lite.yaml \
    --motionbert_ckpt checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin \
    --rep_dim 512 \
    --temporal_layers 2 \
    --pred_hidden_dim 256 \
    --embed_dim 128 \
    --epochs 15 \
    --batch_size 64 \
    --lr_heads 1e-4 \
    --run_name "$RUN_NAME" || {
      echo "Training failed for $FOLD_NAME"
      exit 1
    }
  
  CKPT_PATH="artifacts/imu_guided_video/${RUN_NAME}/best.pt"
  
  if [ ! -f "$CKPT_PATH" ]; then
    echo "Checkpoint not found: $CKPT_PATH"
    exit 1
  fi
  
  echo "[2/3] Evaluating (non-temporal)..."
  $PYTHON experiments/imu_guided_video_encoder/run_eval.py \
    --checkpoint "$CKPT_PATH" \
    --test_csv "$FOLD_DIR/windows_test.csv" \
    --data_root "$DATA_ROOT" \
    --save_json "artifacts/imu_guided_video/${RUN_NAME}/eval_grouped.json" || {
      echo "Non-temporal evaluation failed for $FOLD_NAME"
      exit 1
    }
  
  echo "[3/3] Evaluating (temporal)..."
  $PYTHON experiments/imu_guided_video_encoder/run_eval_temporal.py \
    --checkpoint "$CKPT_PATH" \
    --test_csv "$FOLD_DIR/windows_test.csv" \
    --data_root "$DATA_ROOT" \
    --temporal_confidence_threshold 0.8 \
    --temporal_alpha 0.7 \
    --save_json "artifacts/imu_guided_video/${RUN_NAME}/eval_temporal.json" || {
      echo "Temporal evaluation failed for $FOLD_NAME"
      exit 1
    }
  
  echo "✓ Fold $FOLD_NUM complete!"
  echo ""
done

echo "=========================================="
echo "All folds completed successfully!"
echo "=========================================="
echo "Results saved in: artifacts/imu_guided_video/"
