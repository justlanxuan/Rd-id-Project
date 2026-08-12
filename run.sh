#!/bin/bash
# Unified pipeline runner

set -e

export PYTHONPATH="${PWD}:${PWD}/src"

CONFIG="${1:-configs/totalcapture_vicon_test.yaml}"
STAGE="${2:-all}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "Config: $CONFIG"
echo "Stages: $STAGE"
echo "Python: $PYTHON_BIN"

"$PYTHON_BIN" "$PWD/run_pipeline.py" --config "$CONFIG" --stages "$STAGE"

echo "=== Done ==="
