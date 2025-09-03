# 基础的sft训练脚本
set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset /home/yanzhe/OpenRLHF/datasets/math-short-prompt.jsonl \
   --input_key problem \
   --output_key solution \
   --train_batch_size 16 \
   --micro_train_batch_size 4 \
   --max_samples 500000 \
   --pretrain /home/yanzhe/Qwen2.5-Math-1.5B \
   --save_path ./checkpoint/SFT/Qwen2.5-1.5B-base-mathmix \
   --save_steps 1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --max_epochs 3 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --packing_samples \
   --gradient_checkpointing
EOF
    # --wandb [WANDB_TOKENS]
    # --packing_samples

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi