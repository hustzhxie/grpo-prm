set -x


export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_ENDPOINT=https://hf-mirror.com
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV



python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 5 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --vllm_gpu_memory_utilization 0.6 \
   --init_kl_coef 1e-3 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k3 \
   --advantage_estimator group_norm \
   --pretrain Qwen/Qwen2.5-Math-1.5B \
   --reward_pretrain Qwen/Qwen2.5-Math-PRM-7B \
   --remote_rm_url "Qwen/Qwen2.5-Math-PRM-7B" \
   --save_path /home/zhxie/OpenRLHF/checkpoint/GRPO/Qwen2.5-Math-1.5B-Instruct_lora \
   --ckpt_path /home/zhxie/OpenRLHF/checkpoint/GRPO/Qwen2.5-Math-1.5B-Instruct_lora/ckpt \
   --save_hf_ckpt \
   --save_steps 1 \
   --micro_train_batch_size 2 \
   --train_batch_size 4 \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size 4 \
   --n_samples_per_prompt 6 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 100 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --prompt_data /home/zhxie/OpenRLHF/datasets/math_level3to5_data_processed_with_qwen_prompt.json \
   --input_key question \
   --label_key answer \
   --apply_chat_template \
   --normalize_reward \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep

# You could also try
#   --kl_estimator k2 \

