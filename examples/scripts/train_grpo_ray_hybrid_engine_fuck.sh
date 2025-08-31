set -x


export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_ENDPOINT=https://hf-mirror.com


python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --colocate_actor_ref \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --vllm_gpu_memory_utilization 0.6 \
   --init_kl_coef 1e-3 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k3 \
   --advantage_estimator group_norm \
   --pretrain Qwen/Qwen2.5-Math-1.5B-Instruct \
   --reward_pretrain Qwen/Qwen2.5-Math-PRM-7B \
   --remote_rm_url "Qwen/Qwen2.5-Math-PRM-7B" \
   --save_path /home/yanzhe/OpenRLHF/checkpoint/GRPO/Qwen2.5-Math-1.5B-Instruct \
   --ckpt_path /home/yanzhe/OpenRLHF/checkpoint/GRPO/Qwen2.5-Math-1.5B-Instruct/ckpt \
   --save_hf_ckpt \
   --micro_train_batch_size 1 \
   --train_batch_size 4 \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size 4 \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 100000 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --prompt_data /home/yanzhe/OpenRLHF/datasets/math_level3to5_data_processed_with_qwen_prompt.json \
   --input_key question \
   --apply_chat_template \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep

# You could also try
#   --kl_estimator k2 \
