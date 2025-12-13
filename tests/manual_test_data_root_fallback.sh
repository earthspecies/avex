#!/usr/bin/env bash
# Manual test script to verify data_root fallback behavior during evaluation
#
# This script demonstrates how to test the data_root fallback gracefully:
# 1. Run a small evaluation on a dataset with a valid local data_root
# 2. Delete/move the data_root directory
# 3. Run evaluation again - should fall back to cloud paths with a warning
#
# Usage:
#   bash tests/manual_test_data_root_fallback.sh

set -e

echo "=================================================="
echo "Manual Test: data_root Fallback to Cloud Paths"
echo "=================================================="

# Create a minimal test evaluation config with a data_root that will disappear
TEST_EVAL_CONFIG="configs/evaluation_configs/test_data_root_fallback.yml"
mkdir -p "$(dirname "$TEST_EVAL_CONFIG")"

cat > "$TEST_EVAL_CONFIG" << 'EOF'
dataset_config: configs/data_configs/benchmark.yml

training_params:
  train_epochs: 1
  lr: 0.001
  batch_size: 32
  optimizer: adamw
  weight_decay: 0.01
  amp: false
  amp_dtype: bf16

experiments:
  - run_name: test_data_root_fallback
    run_config: configs/run_configs/pretrained/beats_ft.yml
    probe_config:
      probe_type: "linear"
      aggregation: "mean"
      input_processing: "pooled"
      target_layers: ["last_layer"]
      freeze_backbone: true
      online_training: true
    pretrained: false
    checkpoint_path: gs://representation-learning/models/sl_beats_all.pt
EOF

echo ""
echo "Test Configuration"
echo "=================="
echo "Created minimal evaluation config: $TEST_EVAL_CONFIG"
echo ""
echo "Expected behavior:"
echo "1. First run: Uses local data_root if available"
echo "2. If data_root path doesn't exist: Falls back to cloud paths with warning"
echo ""
echo "To test manually:"
echo ""
echo "  # Run evaluation"
echo "  uv run repr-learn evaluate --config $TEST_EVAL_CONFIG --save_dir ./test_results"
echo ""
echo "Watch the logs for:"
echo "  - If local path exists: Files load quickly from disk"
echo "  - If local path missing: Warning message about missing data_root"
echo "    'data_root .../path... not found for dataset ...'"
echo "    'Clearing data_root to use default cloud paths (may be slower due to resampling)'"
echo ""
echo "The evaluation should continue successfully in both cases."
echo ""
echo "Test config saved to: $TEST_EVAL_CONFIG"
echo "=================================================="
