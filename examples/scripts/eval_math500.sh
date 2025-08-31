#!/usr/bin/env bash
set -euo pipefail

# 评测配置（按需修改）
DATASET_PATH="/home/yanzhe/OpenRLHF/test_datasets/HuggingFaceH4_MATH-500_test"
DATASET_SPLIT="test"

# 两个待对比的预测结果（JSONL）
PRED1="/home/yanzhe/OpenRLHF/outputs/Qwen2.5-3B-base_SFT_math500-2048.jsonl"
PRED2="/home/yanzhe/OpenRLHF/outputs/Qwen2.5-3B-base_mataSFT_math500.jsonl"

# 预测文件里的字段名
PRED_INPUT_KEY="input"
PRED_OUTPUT_KEY="output"

PYTHON_BIN="${PYTHON:-python}"
EVAL_PY="/home/yanzhe/OpenRLHF/examples/scripts/eval_math500.py"

echo "[Config] DATASET_PATH=$DATASET_PATH (split=$DATASET_SPLIT)"
echo "[Config] PRED1=$PRED1"
echo "[Config] PRED2=$PRED2"
echo "[Config] KEYS: input=$PRED_INPUT_KEY, output=$PRED_OUTPUT_KEY"

if [ ! -f "$EVAL_PY" ]; then
  echo "[Error] Eval script not found: $EVAL_PY" >&2
  exit 1
fi

if [ ! -e "$PRED1" ]; then
  echo "[Warn] PRED1 not found: $PRED1" >&2
fi
if [ ! -e "$PRED2" ]; then
  echo "[Warn] PRED2 not found: $PRED2" >&2
fi

"$PYTHON_BIN" "$EVAL_PY" \
  --dataset_path "$DATASET_PATH" \
  --dataset_split "$DATASET_SPLIT" \
  --pred_paths "$PRED1" "$PRED2" \
  --pred_input_key "$PRED_INPUT_KEY" \
  --pred_output_key "$PRED_OUTPUT_KEY"

echo "[Done] Evaluation finished."


