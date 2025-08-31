"""
本文件用于进行 SFT（监督微调，Supervised Fine-Tuning）训练入口脚本。

主要流程概览：
1) 初始化分布式训练策略（DeepSpeed 等），并据此创建/包装模型、优化器、数据加载器；
2) 加载预训练模型作为 Actor（可选 LoRA/4bit/FlashAttention 等加速与内存优化）；
3) 根据传入的数据集路径或 HF 数据集名称，融合/采样训练数据，构造 `SFTDataset`；
4) 计算每轮的训练步数，构建学习率调度器，调用 `SFTTrainer` 执行训练循环；
5) 支持从检查点恢复训练进度，训练完成后保存模型与 Tokenizer。

重要参数说明（部分）：
- micro_train_batch_size：每张 GPU 的微批大小（用于 DataLoader 的 batch_size）；
- train_batch_size：全局训练 batch 大小（通常等于 micro_batch * 累积步数 * GPU 数）；
- max_len：单条样本的最大 token 数；
- dataset：训练数据集路径或 HF 数据集名称；
- pretrain：要加载的 HuggingFace 预训练模型名称或本地路径；
- zero_stage/flash_attn/bf16 等：与加速、显存占用、数值精度相关的策略参数；
- packing_samples：样本打包以提高吞吐（不使用 CrossAttention，依赖 FlashAttention）；
- apply_chat_template/tokenizer_chat_template：是否启用 HF 的 chat 模板来拼接多轮对话样本。

注意：本文件仅组织训练流程与参数解析，具体的模型封装与训练细节在 `openrlhf.models` 与
`openrlhf.trainer.sft_trainer` 中实现，数据处理细节在 `openrlhf.datasets` 中实现。
"""

import argparse
import math
import os
from datetime import datetime

from transformers.trainer import get_scheduler

from openrlhf.datasets import SFTDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.models import Actor
from openrlhf.trainer.sft_trainer import SFTTrainer
from openrlhf.utils import get_strategy, get_tokenizer


def train(args):
    """
    训练入口函数。

    主要职责：
    - 初始化并配置分布式训练策略（如 DeepSpeed、数据并行、张量并行等）。
    - 构建/加载预训练模型（Actor），并按需启用 FlashAttention、bf16、LoRA、4bit 等特性。
    - 准备训练/验证数据集（融合/采样），并构建 `SFTDataset` 与相应 DataLoader。
    - 计算训练步数、配置优化器与学习率调度器，封装至策略对象（strategy.prepare）。
    - 支持从检查点恢复（加载已消耗样本数），并启动训练循环（`SFTTrainer.fit`）。
    - 按 rank0 保存最终权重（可选保存 HF 权重格式）。

    参数
    ------
    args : argparse.Namespace
        由命令行解析得到的所有训练超参数与路径配置。
    """
    # ========== Step 1. 配置分布式/加速策略 ==========
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model
    # 通过 `Actor` 封装底层 HF 模型，统一接入 FlashAttention、bf16、LoRA、4bit、DeepSpeed 等特性。
    # 当启用 packing_samples/ring-attention 等特性时，模型内部会按需适配。
    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        packing_samples=args.packing_samples,
        use_liger_kernel=args.use_liger_kernel,
    )
    # configure tokenizer
    # 注意：`use_fast` 可由参数控制，因为某些模型的 fast tokenizer 存在兼容性问题。
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)
    strategy.print(model)

    # gradient_checkpointing
    # 可选启用梯度检查点以降低显存占用；是否使用 reentrant 模式由参数控制。
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # configure optimizer
    # 基于策略创建优化器（可能包含 ZeRO-offload/参数切分等），支持权重衰减与自定义 betas。
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)

    # prepare for data and dataset
    # ========== Step 2. 加载/融合训练数据集 ==========
    # 支持：
    # - 单一路径/HF 数据集
    # - 多数据集按概率混合（dataset_probs）
    # - 指定 split（train/validation/test 等）
    train_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        dataset_split=args.dataset_split,
    )
    # 进一步裁剪最大样本数，确保不会超过设定的 `max_samples`。
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    # 构建 SFT 数据集：
    # - 支持 pretrain_mode：预训练损失（无监督）或 SFT 损失（监督形式）
    # - 支持自定义输入模板（input_template）与多轮对话（multiturn）
    train_dataset = SFTDataset(
        train_data,
        tokenizer,
        args.max_len,
        strategy,
        pretrain_mode=args.pretrain_mode,
        input_template=args.input_template,
        multiturn=args.multiturn,
    )
    # prepare dataloader
    # 通过策略封装 DataLoader，确保在分布式场景下 sampler、pin_memory、num_workers 等配置一致。
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.collate_fn,
    )

    eval_dataloader = None
    # ========== Step 3. 可选：构建验证集与 DataLoader ==========
    if getattr(args, "eval_dataset", None):   #默认不存在
        eval_data = blending_datasets(
            args.eval_dataset,
            None,
            strategy,
            dataset_split=args.eval_split,
        )
        eval_dataset = SFTDataset(
            eval_data,
            tokenizer,
            args.max_len,
            strategy,
            pretrain_mode=args.pretrain_mode,
            input_template=args.input_template,
            multiturn=args.multiturn,
        )
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset,
            args.micro_train_batch_size,
            True,
            False,
            eval_dataset.collate_fn,
        )

    # scheduler
    # ========== Step 4. 配置学习率调度器 ==========
    # 计算每轮更新步数：数据集大小 // 全局训练 batch（train_batch_size）
    # 注意：train_batch_size 通常与 micro_batch * 累积步数 * GPU 数相关，应确保配置一致性。
    num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size   #数据集总数除以一次梯度更新用到的数量
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        args.lr_scheduler,
        optim,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )

    # prepare models
    # ========== Step 5. 交由策略封装模型/优化器/调度器 ==========
    # 在 DeepSpeed 等策略下，这一步会完成模型并行/参数切分/ZeRO 优化器初始化等工作。
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))

    # load checkpoint
    # ========== Step 6. 可选：从检查点恢复 ==========
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(args.ckpt_path):    #检查是否有已经训练了一部分的检查点
        _, states = strategy.load_ckpt(model.model, args.ckpt_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    os.makedirs(args.save_path, exist_ok=True)

    # configure Trainer
    # ========== Step 7. 构建并启动 Trainer ==========
    trainer = SFTTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        pretrain_mode=args.pretrain_mode,
        batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        tokenizer=tokenizer,
        save_hf_ckpt=args.save_hf_ckpt,
        disable_ds_ckpt=args.disable_ds_ckpt,
    )

    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)

    # save model checkpoint after fitting on only rank0
    # ========== Step 8. 仅在 rank0 保存模型与 tokenizer ==========
    strategy.save_model(model, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./ckpt")    #脚本传入的参数
    parser.add_argument("--save_steps", type=int, default=-1)    #脚本传入的参数
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--logging_steps", type=int, default=1)    #脚本传入的参数
    parser.add_argument("--eval_steps", type=int, default=-1)    #脚本传入的参数
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_sft")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)    #脚本传入的参数
    parser.add_argument("--use_ds_universal_ckpt", action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--micro_train_batch_size", type=int, default=8, help="batch size per GPU")    #脚本传入的参数
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")    #脚本传入的参数
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)    #脚本传入的参数
    parser.add_argument("--deepcompile", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full_determinism",
        action="store_true",
        default=False,
        help="Enable reproducible behavior during distributed training",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")    #脚本传入的参数
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")    #脚本传入的参数
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")    #脚本传入的参数
    parser.add_argument("--use_liger_kernel", action="store_true", default=False, help="Enable Liger Kernel")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--ds_tensor_parallel_size", type=int, default=1, help="DeepSpeed Tensor parallel size")

    # SFT
    parser.add_argument("--max_epochs", type=int, default=2)    #脚本传入的参数
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--pretrain", type=str, default=None)    #脚本传入的参数
    parser.add_argument("--learning_rate", type=float, default=5e-6)    #脚本传入的参数
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--pretrain_mode", action="store_true", default=False, help="Use pretrain loss")
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--l2", type=float, default=0, help="weight decay loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")

    # ring-attention
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument(
        "--ring_head_stride",
        type=int,
        default=1,
        help="the number of heads to do ring attention each time. "
        "It should be a divisor of the number of heads. "
        "A larger value may results in faster training but will consume more memory.",
    )

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # packing SFT samples without CrossAttention
    parser.add_argument("--packing_samples", action="store_true", default=False)    #脚本传入的参数

    # custom dataset
    parser.add_argument("--dataset", type=str, default=None, help="Path to the training dataset")    #脚本传入的参数
    parser.add_argument("--dataset_probs", type=str, default=None, help="Sampling probabilities for training datasets")
    parser.add_argument("--eval_dataset", type=str, default=None, help="Path to the evaluation dataset")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="train")
    parser.add_argument("--max_samples", type=int, default=1000000, help="Maximum number of samples to use")    #脚本传入的参数
    parser.add_argument("--train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--multiturn", action="store_true", default=False, help="Use compacted multiturn dataset")

    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")    #脚本传入的参数
    parser.add_argument("--output_key", type=str, default=None, help="JSON dataset key")    #脚本传入的参数
    parser.add_argument("--input_template", type=str, default="User: {}\nAssistant: ")
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=2048, help="Max tokens for the samples")    #脚本传入的参数

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_sft")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="sft_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    args = parser.parse_args()

    if args.multiturn:
        assert args.apply_chat_template, "apply_chat_template must be enabled when using multiturn format"

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.packing_samples and not args.flash_attn:
        print("[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
        args.flash_attn = True

    if args.ring_attn_size > 1:
        assert args.packing_samples, "packing_samples must be enabled when using ring attention"

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope to speed up.
        patch_hub()

    train(args)
