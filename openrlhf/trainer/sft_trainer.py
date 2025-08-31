"""
SFTTrainer：监督微调（Supervised Fine-Tuning, SFT）训练器。

功能要点：
- 组织 SFT 的训练/评测/日志与检查点保存流程；
- 兼容分布式训练策略（strategy），如 DeepSpeed、数据并行、张量并行等；
- 可选日志后端（Weights & Biases / TensorBoard）；
- 通过 `SFTLoss` 计算自回归语言建模损失，并可选叠加 MoE 额外正则项（aux_loss）。

用法：由 `openrlhf/cli/train_sft.py` 调用本训练器，在 `fit()` 中完成训练循环，在 `evaluate()` 中执行评测，
日志与检查点保存集中在 `save_logs_and_checkpoints()`。
"""

import os
from abc import ABC

import torch
from torch.optim import Optimizer
from tqdm import tqdm

from openrlhf.models import SFTLoss
from openrlhf.utils.distributed_sampler import DistributedSampler


# 该类封装了 SFT 的常规训练范式：前向 -> 损失 -> 反向 -> 优化器步进 -> 日志/评测/保存
class SFTTrainer(ABC):
    """
    Trainer for supervised fine-tuning (SFT).

    Args:
        model (torch.nn.Module): The model to be trained.
        strategy (Strategy): The training strategy to be applied.
        optim (Optimizer): The optimizer for model training.
        train_dataloader (DataLoader): The dataloader for the training dataset.
        eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
        scheduler (Scheduler): The learning rate scheduler to adjust training rates.
        max_norm (float, defaults to 1): Maximum gradient norm for clipping to prevent exploding gradients.
        pretrain_mode (bool, defaults to False): Flag to indicate if the trainer is in pre-training mode.
        batch_size (int, defaults to 1): Batch size for training.
        max_epochs (int, defaults to 2): The maximum number of training epochs.
        tokenizer (Tokenizer, optional): The tokenizer for processing input data.
        save_hf_ckpt (bool): Whether to save huggingface-format model weight.
        disable_ds_ckpt (bool): Whether not to save deepspeed-format model weight. (Deepspeed model weight is used for training recovery)
    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm: float = 1,
        pretrain_mode: bool = False,
        batch_size: int = 1,
        max_epochs: int = 2,
        tokenizer=None,
        save_hf_ckpt: bool = False,
        disable_ds_ckpt: bool = False,
    ) -> None:
        super().__init__()
        # ========== 训练核心对象与超参数 ==========
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.pretrain_mode = pretrain_mode
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.args = strategy.args
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt

        # 标准 SFT 损失（自回归语言建模损失）
        self.loss_fn = SFTLoss()

        # Mixtral 8*7B 等 MoE 模型的额外正则项（aux loss），由超参数开关控制
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # 是否启用样本打包（packing），通常与 FlashAttention 配合提升吞吐
        self.packing_samples = strategy.args.packing_samples

        # ========== 日志后端：W&B / TensorBoard ==========
        # 仅在 rank0 初始化日志实例，避免多卡重复写入
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # 若未使用 W&B，则可选使用 TensorBoard
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        """
        执行完整的 SFT 训练流程，包括：
        - 配置评测/保存步频
        - 断点恢复（根据 consumed_samples 恢复起始 step/epoch）
        - 多轮 epoch 的训练：前向、损失、反向、优化器步进
        - 周期性日志/评测/保存
        - 训练结束后关闭日志后端
        """
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        # Restore step and start_epoch
        # 基于已消耗样本数推算起始 step 与 epoch，支持从检查点恢复训练
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)  # 更新当前epoch中的进度，默认设置epoch是0，所以不用理会

        epoch_bar = tqdm(
            range(start_epoch, self.epochs), # 迭代的范围：从 start_epoch 到 总 epoch 数
            desc="Train epoch",  
            disable=not self.strategy.is_rank_0(),  # 只在 rank=0 的进程显示进度条
        )
        loss_sum = 0
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                # 分布式采样器需在每个 epoch 设置随机种子相关状态，确保各进程采样一致性
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # train
            self.model.train()  #启动训练模式
            for inputs, attention_masks, loss_masks in self.train_dataloader:
                # 将批数据搬运至当前 GPU；多数情况下形状为 [B, 1, L]，使用 squeeze(1) 去掉冗余维度
                inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)
                loss_mask = loss_masks.to(torch.cuda.current_device()).squeeze(1)
                # 前向：返回逐 token 对数概率与可选的底层输出（用于 MoE aux_loss）
                per_token_log_probs, output = self.model(
                    inputs,
                    attention_mask=attention_mask,
                    return_output=True,
                    return_logprobs=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                )

                # mixtral
                if self.aux_loss:
                    aux_loss = output.aux_loss
                else:
                    aux_loss = 0
                # 语言模型损失：基于 per_token_log_probs 与标签（由 loss_mask 对齐）
                gpt_loss = self.loss_fn(per_token_log_probs, loss_mask[:, :-1])
                # 总损失 = 主损失 + MoE 正则项（按系数缩放）
                loss = gpt_loss + aux_loss * self.args.aux_loss_coef
                # 反向传播（由 strategy 封装，内部可能处理梯度累积/混合精度/ZeRO 等）
                self.strategy.backward(loss, self.model, self.optimizer)
                # 优化器与调度器步进（同样由 strategy 统一处理）
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                loss_sum += gpt_loss.item()
                logs_dict = {
                    "gpt_loss": gpt_loss.item(),
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()
                # step bar
                # 多卡场景下先 all-reduce 指标，再在 rank0 侧更新 tqdm 展示
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    # “有效步”结束时（考虑梯度累积），记录平均损失并按频率触发日志/评测/保存
                    logs_dict["loss_mean"] = loss_sum / self.strategy.accumulated_gradient
                    loss_sum = 0
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1

            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    # logs/checkpoints/evaluation
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        """
        统一处理日志记录、评测触发与检查点保存。

        参数：
            args: 训练参数集合。
            global_step: 已完成的有效训练步（考虑梯度累积后的步数）。
            step_bar: tqdm 进度条对象，用于展示当前状态。
            logs_dict: 需要记录/展示的指标字典（在多卡场景下先做 all-reduce）。
            client_states: 自定义状态字典（例如 consumed_samples），将写入检查点用于断点恢复。
        """
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        if global_step % args.eval_steps == 0:
            # do eval when eval_dataloader is not None and len(dataloader) > 0, avoid zero division in eval.
            if self.eval_dataloader is not None and len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)

        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            if not self.disable_ds_ckpt:
                self.strategy.save_ckpt(
                    self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
                )
            if self.save_hf_ckpt:
                save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
                self.strategy.save_model(self.model, self.tokenizer, save_path)

    def evaluate(self, eval_dataloader, steps=0):
        """
        在验证集上评测模型：
        - 关闭梯度计算，逐步累积损失并做 all-reduce；
        - 在 rank0 写入日志；
        - 评测结束后恢复 `train()` 状态。
        """
        times = 0
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            for inputs, attention_masks, loss_masks in eval_dataloader:
                # 将批数据搬运至当前 GPU，并去掉冗余维度
                inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)
                loss_mask = loss_masks.to(torch.cuda.current_device()).squeeze(1)
                # 评测阶段仅需逐 token 对数概率（不返回底层输出）
                per_token_log_probs = self.model(
                    inputs,
                    attention_mask=attention_mask,
                    return_logprobs=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                )

                loss = self.loss_fn(per_token_log_probs, loss_mask[:, :-1])

                times += 1
                loss_sum += loss.item()
                bar_dict = {"eval gpt_loss": loss_sum / times}
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()  # reset model state
