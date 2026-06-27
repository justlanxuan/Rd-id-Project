#!/bin/bash

# 脚本：在 custom 数据上跑完整的 temporal matcher 训练和评估 pipeline

set -e  # Exit on error

PROJECT_ROOT="/home/fzliang/Autism-project"
cd "$PROJECT_ROOT"

# 设置 PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

CONDA_ENV="/home/fzliang/miniconda3/envs/autism_test"
PYTHON_BIN="${CONDA_ENV}/bin/python"

DATA_ROOT="data/interim/custom_complete/slice"
FOLDS_DIR="${DATA_ROOT}/folds_3train_valtest"

echo "======================================"
echo "Temporal Matcher Pipeline"
echo "======================================"
echo "Project root: ${PROJECT_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Data root: ${DATA_ROOT}"
echo ""

# Check fold directories exist
if [ ! -d "$FOLDS_DIR" ]; then
    echo "ERROR: Folds directory not found: $FOLDS_DIR"
    exit 1
fi

echo "Found fold directories:"
ls -d "${FOLDS_DIR}"/fold* 2>/dev/null | head -3

echo ""
echo "======================================"
echo "STEP 1: Training Legacy IMU-Guided Encoder (3 folds)"
echo "======================================"

for FOLD_DIR in "$FOLDS_DIR"/fold*; do
    if [ ! -d "$FOLD_DIR" ]; then
        continue
    fi
    
    FOLD_NAME=$(basename "$FOLD_DIR")
    RUN_NAME="custom_imu_guided_legacy_${FOLD_NAME}_e15"
    
    echo ""
    echo "Training fold: $FOLD_NAME"
    echo "Run name: $RUN_NAME"
    
    if [ -f "$FOLD_DIR/windows_train.csv" ] && [ -f "$FOLD_DIR/windows_val.csv" ]; then
        echo "Found train and val CSVs, starting training..."
        
        "$PYTHON_BIN" experiments/imu_guided_video_encoder/run_training.py \
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
            --run_name "$RUN_NAME"
        
        echo "✓ Training completed for $FOLD_NAME"
    else
        echo "  Skipping $FOLD_NAME: missing train/val CSV files"
    fi
done

echo ""
echo "======================================"
echo "STEP 2: Evaluation with Temporal Matcher"
echo "======================================"

for FOLD_DIR in "$FOLDS_DIR"/fold*; do
    if [ ! -d "$FOLD_DIR" ]; then
        continue
    fi
    
    FOLD_NAME=$(basename "$FOLD_DIR")
    RUN_NAME="custom_imu_guided_legacy_${FOLD_NAME}_e15"
    CHECKPOINT="artifacts/imu_guided_video/${RUN_NAME}/best.pt"
    TEST_CSV="$FOLD_DIR/windows_test.csv"
    
    echo ""
    echo "Evaluating fold: $FOLD_NAME"
    
    if [ ! -f "$CHECKPOINT" ]; then
        echo "  Skipping: checkpoint not found at $CHECKPOINT"
        continue
    fi
    
    if [ ! -f "$TEST_CSV" ]; then
        echo "  Skipping: test CSV not found at $TEST_CSV"
        continue
    fi
    
    echo "Checkpoint: $CHECKPOINT"
    echo "Test CSV: $TEST_CSV"
    
    "$PYTHON_BIN" experiments/imu_guided_video_encoder/run_eval_temporal.py \
        --checkpoint "$CHECKPOINT" \
        --test_csv "$TEST_CSV" \
        --data_root "$DATA_ROOT" \
        --confidence_threshold 0.8 \
        --temporal_alpha 0.7 \
        --group_sizes "2,4,6" \
        --num_trials 30 \
        --save_json "artifacts/imu_guided_video/${RUN_NAME}/eval_temporal.json"
    
    echo "✓ Evaluation completed for $FOLD_NAME"
done

echo ""
echo "======================================"
echo "STEP 3: Summary"
echo "======================================"
echo "All training and evaluation completed!"
echo ""
echo "Checkpoint locations:"
find artifacts/imu_guided_video -name "best.pt" -type f -newer /tmp 2>/dev/null | head -5

echo ""
echo "Evaluation results:"
find artifacts/imu_guided_video -name "eval_temporal.json" -type f -newer /tmp 2>/dev/null | head -5

echo ""
echo "✓ Pipeline completed successfully!"
