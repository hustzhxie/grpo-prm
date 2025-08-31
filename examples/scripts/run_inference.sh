#!/usr/bin/env bash
set -euo pipefail

# =========================
# 配置区（仅需修改这里）
# =========================

# 模型与数据集
MODEL="Qwen/Qwen2.5-Math-1.5B-Instruct"           # HF 模型名或本地权重路径
DATASET="HuggingFaceH4/MATH-500"             # HF 数据集名或本地 JSON/JSONL 路径
INPUT_KEY="problem"                          # 数据集中问题字段名（如 question/prompt/query）
DATASET_SPLIT="test"                         # 数据集分割：train/validation/test
EXPORT_SPLIT_TO_DISK=true                     # 将指定 split 预导出到本地，避免 DatasetDict 报错

# 输出
OUTPUT_PATH="./outputs/Qwen2.5-1.5B-math_math500.jsonl"  # 结果 JSONL 路径

# 生成参数
MAX_NEW_TOKENS=2048
PROMPT_MAX_LEN=1024
TOP_P=0.9
TEMPERATURE=0.7
MICRO_BATCH_SIZE=16
EVAL_TASK="generate_vllm"
TP_SIZE=1                 # vLLM 张量并行大小（等于你要用的 GPU 数）
MAX_NUM_SEQS=8            # vLLM 单次并发序列数，调小可省显存

# 性能/显存相关
USE_BF16=true            # true/false -> 是否启用 bf16
USE_FLASH_ATTN=true      # true/false -> 是否启用 FlashAttention2
ZERO_STAGE=0             # 推理使用的 DeepSpeed ZeRO Stage（0 表示关闭 ZeRO）
DISABLE_FAST_TOKENIZER=false  # 部分模型需关闭 fast tokenizer 可设为 true

# 可选：显存碎片配置（建议开启）
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True,max_split_size_mb:64"}

# 可选：固定 HF 缓存（如你的环境里路径不同请改）
# export HF_HUB_CACHE="/home/cache/huggingface/hub"
# export HF_DATASETS_CACHE="/home/cache/huggingface/datasets"

# =========================
# 运行区（无需改动）
# =========================

mkdir -p "$(dirname "$OUTPUT_PATH")"

BF16_FLAG=""; $USE_BF16 && BF16_FLAG="--bf16" || true
FLASH_FLAG=""; $USE_FLASH_ATTN && FLASH_FLAG="--flash_attn" || true
DFT_FLAG=""; $DISABLE_FAST_TOKENIZER && DFT_FLAG="--disable_fast_tokenizer" || true

echo "[Config] MODEL=$MODEL"
echo "[Config] DATASET=$DATASET (split=$DATASET_SPLIT, input_key=$INPUT_KEY, export_split=$EXPORT_SPLIT_TO_DISK)"
echo "[Config] OUTPUT_PATH=$OUTPUT_PATH"
echo "[Config] MAX_NEW_TOKENS=$MAX_NEW_TOKENS, PROMPT_MAX_LEN=$PROMPT_MAX_LEN, TOP_P=$TOP_P, TEMPERATURE=$TEMPERATURE"
echo "[Config] MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE, ZERO_STAGE=$ZERO_STAGE, BF16=$USE_BF16, FLASH_ATTN=$USE_FLASH_ATTN"
echo "[Config] EVAL_TASK=$EVAL_TASK, TP_SIZE=$TP_SIZE, MAX_NUM_SEQS=$MAX_NUM_SEQS"

# 若需要，将 HF 数据集的指定 split 预导出为本地 Dataset，避免 DatasetDict 无法 select 的问题
DATASET_ARG="$DATASET"
if $EXPORT_SPLIT_TO_DISK; then
  case "$DATASET" in
    */*)
      SAFE_NAME=$(echo "$DATASET" | tr '/:' '__')
      OUT_DIR="./test_datasets/${SAFE_NAME}_${DATASET_SPLIT}"
      mkdir -p "$OUT_DIR"
      echo "[Prepare] Export HF dataset split to disk -> $OUT_DIR"
      HF_DATASET_NAME="$DATASET" HF_DATASET_SPLIT="$DATASET_SPLIT" HF_DATASET_OUTDIR="$OUT_DIR" \
      python - <<'PY'
import os
from datasets import load_dataset
name = os.environ['HF_DATASET_NAME']
split = os.environ['HF_DATASET_SPLIT']
outdir = os.environ['HF_DATASET_OUTDIR']
try:
    need_export = not os.path.isdir(outdir) or len(os.listdir(outdir)) == 0
except Exception:
    need_export = True
if need_export:
    print(f"Exporting {name} split={split} to {outdir} ...")
    ds = load_dataset(name, split=split)
    ds.save_to_disk(outdir)
else:
    print(f"Use cached dataset at {outdir}")
PY
      DATASET_ARG="$OUT_DIR"
      ;;
    *)
      echo "[Prepare] Skip export (DATASET looks like local file/directory)"
      ;;
  esac
fi

# === 根据任务选择运行方式 ===
if [ "$EVAL_TASK" = "generate_vllm" ]; then
  # vLLM 不要用 deepspeed 包裹；按需设置 GPU 列表
  export CUDA_VISIBLE_DEVICES=1,2,3      #GPU数量
  export VLLM_WORKER_MULTIPROC_METHOD=spawn

  python -m openrlhf.cli.batch_inference \
    --eval_task generate_vllm \
    --pretrain "$MODEL" \
    --dataset "$DATASET_ARG" \
    --input_key "$INPUT_KEY" \
    --prompt_max_len "$PROMPT_MAX_LEN" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --top_p "$TOP_P" \
    --temperature "$TEMPERATURE" \
    --tp_size "$TP_SIZE" \
    --max_num_seqs "$MAX_NUM_SEQS" \
    --output_path "$OUTPUT_PATH" \
    --input_template "User: {}Please answer step by step.\nAssistant: "
else
  deepspeed --module openrlhf.cli.batch_inference \
    --eval_task generate \
    --zero_stage "$ZERO_STAGE" \
    $BF16_FLAG \
    $FLASH_FLAG \
    $DFT_FLAG \
    --pretrain "$MODEL" \
    --dataset "$DATASET_ARG" \
    --input_key "$INPUT_KEY" \
    --prompt_max_len "$PROMPT_MAX_LEN" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --top_p "$TOP_P" \
    --temperature "$TEMPERATURE" \
    --micro_batch_size "$MICRO_BATCH_SIZE" \
    --output_path "$OUTPUT_PATH"
fi

echo "Inference done -> $OUTPUT_PATH"


