import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import timedelta
from typing import Any, List, Tuple, Union

from numpy import append

import ray
import torch
import time

from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.seqlen_balancing import get_minimum_num_micro_batch_size, get_seqlen_balanced_partitions
from openrlhf.utils.utils import remove_pad_token, zero_pad_sequences

logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Experience:
    """Experience is a batch of data for RLHF training.

    Shapes of each tensor:
    index: (B,)
    sequences: (B, S)
    attention_mask: (B, S)
    action_mask: (B, A)
    action_log_probs: (B, S)
    base_action_log_probs: (B, S)
    values: (B, S)
    returns: (B, S)
    advantages: (B, S)
    kl: (B, S)
    info: dict[str, list]
    """

    index: list[int] = None
    sequences: torch.Tensor = None
    attention_mask: torch.LongTensor = None
    action_mask: torch.BoolTensor = None

    action_log_probs: torch.Tensor = None
    base_action_log_probs: torch.Tensor = None
    rollout_log_probs: torch.Tensor = None
    values: torch.Tensor = None
    returns: torch.Tensor = None
    advantages: torch.Tensor = None
    kl: torch.Tensor = None

    prompts: list[str] = None
    labels: list[str] = None
    rewards: torch.Tensor = None  # used for advantage calculation
    scores: torch.Tensor = None  # 0-1 reward used for dynamic sampling

    # the info field is used to store additional information
    # all the fields in the info will be logged to the tensorboard/wandb
    info: dict[str, torch.Tensor] = None

    def __init__(
        self,
        index=None,
        sequences=None,
        action_log_probs=None,
        base_action_log_probs=None,
        rollout_log_probs=None,
        values=None,
        returns=None,
        advantages=None,
        attention_mask=None,
        action_mask=None,
        kl=None,
        prompts=None,
        labels=None,
        rewards=None,
        scores=None,
        info=None,
    ):
        self.index = index
        self.sequences = sequences
        self.action_log_probs = action_log_probs
        self.base_action_log_probs = base_action_log_probs
        self.rollout_log_probs = rollout_log_probs
        self.values = values
        self.returns = returns
        self.advantages = advantages
        self.attention_mask = attention_mask
        self.action_mask = action_mask
        self.kl = kl
        self.prompts = prompts or []
        self.labels = labels or []
        self.rewards = rewards
        self.scores = scores
        self.info = info or []

    @torch.no_grad()
    def to_device(self, device: torch.device):
        """Move all tensor fields to the specified device."""
        for field, value in self.__dict__.items():
            if isinstance(value, dict):
                setattr(self, field, {key: to(val, device) for key, val in value.items()})
            else:
                setattr(self, field, to(value, device))

        return self

    def pin_memory(self):
        """Pin memory for all tensor fields."""
        for field, value in self.__dict__.items():
            if isinstance(value, dict):
                setattr(self, field, {key: pin_memory(val) for key, val in value.items()})
            else:
                setattr(self, field, pin_memory(value))

        return self

    @staticmethod
    def select(experiences: List["Experience"], fields: List[str]) -> List["Experience"]:
        """Select specific fields from a list of Experience instances to create new Experience instances.

        Args:
            experiences: List of Experience instances
            fields: List of field names to select

        Returns:
            A list of new Experience instances containing only the selected fields
        """
        new_experiences = []
        for exp in experiences:
            new_exp = Experience()
            for field in fields:
                if hasattr(exp, field):
                    setattr(new_exp, field, getattr(exp, field))
            new_experiences.append(new_exp)
        return new_experiences

    @staticmethod
    def _merge_item(items: List, pad_value: int = 0) -> Union[torch.Tensor, list, dict, Any]:
        """Merge a list of items into a single item.
        Recursively merge tensors, lists and dicts.
        For tensors, use zero_pad_sequences to merge sequences of different lengths.

        Args:
            items: List of items to merge
            pad_value: Value used for padding tensors
        """
        if isinstance(items[0], torch.Tensor):
            return zero_pad_sequences(items, side="right", value=pad_value)
        elif isinstance(items[0], list):
            return sum(items, [])
        elif isinstance(items[0], dict):
            result = {}
            # Collect all values for each key
            for d in items:
                for key, value in d.items():
                    if key not in result:
                        result[key] = []
                    result[key].append(value)
            # Merge all values for each key at once
            return {key: Experience._merge_item(values, pad_value) for key, values in result.items()}
        elif items[0] is None:
            return None
        else:
            raise ValueError(f"Unsupported type: {type(items[0])}")

    @staticmethod
    def concat_experiences(experiences_list: List["Experience"], pad_token_id) -> "Experience":
        """Concatenate multiple experiences into one large experience.

        Args:
            experiences_list: List of Experience to concatenate
            pad_token_id: Token id used for padding sequences

        Returns:
            A new Experience instance containing all the concatenated data
        """
        if not experiences_list:
            return Experience()

        # Get all field names from the dataclass
        field_names = [f.name for f in fields(Experience)]

        # Create result dictionary
        result = {}

        # Merge all fields
        for field in field_names:
            values = [getattr(e, field) for e in experiences_list]
            # Use pad_token_id for sequences field, 0 for others
            pad_value = pad_token_id if field == "sequences" else 0
            result[field] = Experience._merge_item(values, pad_value)

        return Experience(**result)


def update_samples_with_rewards(rewards_info, samples_list):
    """Process rewards info and update samples with rewards, scores and extra logs.

    Args:
        rewards_info: List of reward information dictionaries
        samples_list: List of Experience objects to update
    """
    # Process rewards and scores
    samples_len = [len(sample.sequences) for sample in samples_list]   # 获取每个experience中的样本数量
    rewards_list = torch.cat([info["rewards"] for info in rewards_info], dim=0).split(samples_len)
    if "scores" in rewards_info[0]:
        scores_list = torch.cat([info["scores"] for info in rewards_info], dim=0).split(samples_len)
    else:
        scores_list = rewards_list

    # Process extra_logs if present
    if "extra_logs" in rewards_info[0]:
        # Merge all extra_logs tensors first
        merged_logs = {
            key: torch.cat([logs[key] for logs in [info["extra_logs"] for info in rewards_info]], dim=0).split(
                samples_len
            )
            for key in rewards_info[0]["extra_logs"].keys()
        }

    # Update samples with rewards, scores and extra logs
    for i, samples in enumerate(samples_list):
        samples.rewards = rewards_list[i]  # 既可能是token-level的奖励，也可能是sample-level的奖励
        samples.scores = scores_list[i]
        samples.info["score"] = scores_list[i]
        samples.info["reward"] = rewards_list[i]
        if "extra_logs" in rewards_info[0]:
            for key, values in merged_logs.items():
                samples.info[key] = values[i]

    return samples_list


class SamplesGenerator:
    def __init__(self, vllm_engines, strategy, tokenizer, prompt_max_len):
        self.strategy = strategy
        self.args = strategy.args
        self.vllm_engines = vllm_engines
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_labels, **generate_kwargs) -> List[Experience]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        rollout_samples = self._generate_vllm(all_prompts, all_labels, **generate_kwargs)

        # vLLM offload when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        return rollout_samples

        # tokenizer

    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    def _generate_vllm(self, all_prompts: List[str], all_labels, **kwargs) -> List[Experience]:
        """Generate samples using vLLM engine.

        Args:
            all_prompts: List of prompts to generate from
            all_labels: List of labels corresponding to prompts
            **kwargs: Additional arguments for generation

        Returns:
            List of Experience objects containing generated samples
        """
        from vllm import SamplingParams
        # 说明：SamplingParams 是 vLLM 的采样配置对象，用于指定解码时的采样/截断策略
        # 包括温度、top-p、top-k、最大/最小生成长度、是否跳过特殊符号、是否把停止词包含进输出等。

        llms = self.vllm_engines
        args = self.strategy.args
        # llms：一组 vLLM 引擎（通常是 Ray Actor 封装的远端推理服务）
        # args：训练/采样的全局参数对象，从 strategy 中继承

        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
            logprobs=1 if self.strategy.args.enable_vllm_is_correction else None,
        )
        # 关键参数释义：
        # - temperature：温度采样，越小越保守，越大越发散
        # - top_p / top_k：核采样/Top-K 采样的截断阈值
        # - max_tokens / min_tokens：控制生成回复的 token 上限/下限
        # - skip_special_tokens：是否在解码文本时丢弃特殊 token（如 EOS）
        # - include_stop_str_in_output：命中停止字符串时是否保留到输出里
        # - logprobs：当启用 vLLM IS-correction 时，请求返回每个生成 token 的对数概率
        max_response_length = kwargs.get("max_new_tokens", 1024)
        truncate_length = self.prompt_max_len + max_response_length
        # truncate_length：总序列最大长度上限 = prompt 最大长度 + 回复最大长度
        # 用于后续将拼接后的序列统一裁剪，避免过长导致显存或后处理开销过大

        # Expand prompt list based on the number of samples per prompt
        n_samples_per_prompt = kwargs.pop("n_samples_per_prompt", args.n_samples_per_prompt)
        all_prompts = sum([[prompt] * n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * n_samples_per_prompt for label in all_labels], [])
        all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]
        # 这里会把每条 prompt 复制 n 次（做多样性采样），并在不 padding 的前提下拿到每条 prompt 的 token 序列
        # 注意：不 padding 可以减少无效计算，便于把 prompt 按长度分发到不同引擎

        # Distribute requests to engines and collect responses
        refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            refs.append(llm.add_requests.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids))
        ray.get(refs)
        # 将请求按近似均匀的批大小切分并分发给各个 vLLM 引擎；
        # ray.get(refs) 用于确保所有请求已成功提交（而非等待生成完成）。

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote())
        print("get responses")
        # 收集所有 vLLM 引擎的响应结果
        # all_outputs 是一个列表，每个元素都是一个 output 对象
        # output 对象包含 prompt_token_ids 和 outputs 列表
        all_outputs = sum(ray.get(all_output_refs), [])
        # 这里再统一收集各引擎返回的生成结果，并按顺序合并为一个列表。

        # Process outputs into Experience objects
        # 遍历每个 vLLM 引擎返回的 output，将其转换为 Experience 对象
        samples_list = []
        for i in range(len(all_outputs)):
            # all_outputs[i] 是一个 output 对象，包含：
            # - prompt_token_ids: 输入的 prompt 的 token IDs
            # - outputs: 包含生成响应的列表（通常只有一个元素）
            output = all_outputs[i]
            prompt = all_prompts[i]
            label = all_labels[i]
            # 将第 i 个输出与对应的 prompt/label 对齐，保证后续监督信号与采样结果一一对应。
            
            # output 结构说明（来自 vLLM 引擎返回）：
            # output = {
            #     'prompt_token_ids': [1, 2, 3, ...],  # 输入的 prompt 的 token IDs
            #     'outputs': [                          # 这是一个列表，包含生成的响应
            #         {
            #             'token_ids': [100, 101, 102, ...],  # 生成的 response 的 token IDs
            #             'logprobs': [                        # 每个生成位置的概率分布字典列表
            #                 {100: LogprobInfo(logprob=-2.1), 101: LogprobInfo(logprob=-1.8), ...},  # 第0个位置
            #                 {200: LogprobInfo(logprob=-1.5), 201: LogprobInfo(logprob=-2.3), ...},  # 第1个位置
            #                 # ... 每个位置包含所有可能token的概率分布
            #             ]
            #             # 其他可能的字段...
            #         }
            #         # 理论上可能有多个输出，但通常只有一个
            #     ]
            # }
            # 注意：outputs[0] 表示第一个（通常也是唯一的）生成响应

            # Concatenate prompt and output tokens
            # output.outputs[0] 获取第一个生成响应（通常也是唯一的）
            # output.outputs[0].token_ids 包含生成的 response 的 token IDs
            input_ids = list(output.prompt_token_ids) + list(output.outputs[0].token_ids)   #拼接输入和输出序列

            attention_mask = [1] * len(input_ids)
            # attention_mask：与 input_ids 等长，1 表示有效 token，0 表示 padding；这里无 padding，故全为 1。

            sequences = torch.tensor(input_ids)

            attention_mask = torch.tensor(attention_mask)

            # Create action mask based on output token positions
            action_mask = torch.zeros_like(attention_mask)
            # 获取生成响应的长度，用于后续创建 action_mask
            response_length = len(output.outputs[0].token_ids)
            # 将 response 部分（从 prompt 结束位置开始）的 mask 设为 1，prompt 部分保持为 0
            action_mask[len(output.prompt_token_ids) : len(output.prompt_token_ids) + response_length] = 1   #把response部分mask为1，prompt为0
            # action_mask：用于区分「动作」(response) 与「上下文」(prompt) 的位置，
            # 只有 response 部分为 1，会参与策略梯度/优势计算，prompt 部分为 0。

            # Calculate rollout log probs
            rollout_log_probs = None
            if self.strategy.args.enable_vllm_is_correction:   # 默认不使用
                rollout_log_probs = []
                # 获取生成响应的 token IDs，用于后续计算对数概率
                response_ids = list(output.outputs[0].token_ids)
                # 遍历每个生成 token 的对数概率
                # output.outputs[0].logprobs 是一个列表，每个元素包含该位置所有可能 token 的概率分布
                for i, logprob in enumerate(output.outputs[0].logprobs):
                    # 从概率分布中提取实际生成 token 的对数概率
                    rollout_log_probs.append(logprob[response_ids[i]].logprob)

                rollout_log_probs = torch.tensor([0.0] * len(list(output.prompt_token_ids)) + rollout_log_probs)
                rollout_log_probs = rollout_log_probs[1:truncate_length].to("cpu")
                # 当启用 IS-correction：
                # - vLLM 会返回每个生成 token 的对数概率，这里将其与 prompt 的长度对齐（在前面补零）
                # - 然后裁剪到 truncate_length，并放到 CPU，便于后续经验拼接与节省显存

            sequences = sequences[:truncate_length].to("cpu")
            attention_mask = attention_mask[:truncate_length].to("cpu")
            action_mask = action_mask[1:truncate_length].to("cpu")
            total_length = attention_mask.float().sum()
            is_clipped = response_length >= max_response_length
            # 注意：action_mask 从索引 1 开始裁剪，以与后续右移对齐的行为保持一致（常见于自回归训练场景）。
            # total_length：当前序列总 token 数（用于动态 batch 切分等），is_clipped：回复是否命中最大长度上限。

            info = {
                "response_length": torch.tensor([response_length]),
                "total_length": torch.tensor([total_length]),
                "response_clip_ratio": torch.tensor([is_clipped]),
            }
            # info 中记录每条样本的统计量：
            # - response_length：回复 token 数
            # - total_length：总序列长度（prompt+response）
            # - response_clip_ratio：是否发生了最大长度裁剪（布尔值）

            rollout_samples = Experience(
                sequences=sequences.unsqueeze(0),      #完整的token序列
                attention_mask=attention_mask.unsqueeze(0),   #掩码
                action_mask=action_mask.unsqueeze(0),     #区分动作(response)和prompt
                rollout_log_probs=rollout_log_probs.unsqueeze(0) if rollout_log_probs is not None else None,
                prompts=[prompt],
                labels=[label],
                info=info,
            )
            samples_list.append(rollout_samples)
            # 每个 output 构造成一个 Experience（批维度为 1）：
            # - sequences/attention_mask/action_mask：张量形状统一
            # - prompts/labels：以列表形式保存原文本与标签
            # - rollout_log_probs：仅在启用 IS-correction 时可用

        # Get rewards from remote reward models if needed
        # This is required by dynamic sampling
        remote_reward_model = kwargs.get("remote_reward_model", None)
        print(remote_reward_model)
        if remote_reward_model:
            print("get rewards")
            all_queries = sum(
                [
                    self.tokenizer.batch_decode(
                        remove_pad_token(s.sequences, s.action_mask), skip_special_tokens=False   #原先是attention_mask
                    )
                    for s in samples_list
                ],
                [],
            )   
            all_prompts = sum([s.prompts for s in samples_list], [])
            all_labels = sum([s.labels for s in samples_list], [])
            # 如果传入了远端奖励模型（如环境、HTTP 服务或自定义函数），
            # 我们需要把当前生成的 query（prompt+response）与 prompt/label 一并发送以获取奖励信息。
            # 这里 decode 时选择 skip_special_tokens=False，确保用于奖励模型的输入更贴近真实模型输出。

            # Get rewards info from remote model
            rewards_info = ray.get(remote_reward_model.get_rewards.remote(all_queries, all_prompts, all_labels))
            # Process rewards and scores
            update_samples_with_rewards(rewards_info, samples_list)
            # 将远端返回的奖励/评分/额外日志写回到每条 Experience，供后续优势估计与日志记录使用。
        else:
            print("no rewards in rollout")

        return samples_list


class RemoteExperienceMaker(ABC):
    def __init__(
        self,
        actor_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        reward_model_group: RayActorGroup,
        initial_model_group: RayActorGroup,
        kl_controller,
        strategy=None,
        tokenizer=None,
        remote_reward_model=None,
        **kwargs,
    ):
        super().__init__()

        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_group = reward_model_group
        self.initial_model_group = initial_model_group
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.advantage_estimator = strategy.args.advantage_estimator
        self.args = strategy.args

        # remote_rm_url indicates that the remote reward model is agent enviroment, remote http server or custom reward func
        self.remote_rm_url = self.args.remote_rm_url
        self.remote_reward_model = remote_reward_model
        self.tokenizer = tokenizer

    def split_rollout_samples(self, rollout_samples):
        for i, sample in enumerate(rollout_samples):
            sample.index = [i]    # 标记每个sample的位置

        samples_list = []
        if self.args.use_dynamic_batch:
            total_lengths = [int(s.info["total_length"].item()) for s in rollout_samples]
            effective_actor_num = (
                self.args.actor_num_nodes
                * self.args.actor_num_gpus_per_node
                // self.args.ring_attn_size
                // self.args.ds_tensor_parallel_size
            )
            minimum_batch_num = get_minimum_num_micro_batch_size(
                total_lengths,
                self.args.rollout_max_tokens_per_gpu,
                self.args.ring_attn_size,
                self.args.ds_tensor_parallel_size,
            )
            minimum_batch_num = minimum_batch_num // effective_actor_num * effective_actor_num
            num_batch = max(minimum_batch_num, effective_actor_num)
            batch_indexes = get_seqlen_balanced_partitions(total_lengths, num_batch, False)
            for micro_index in batch_indexes:
                micro_batch = [rollout_samples[idx] for idx in micro_index]
                concat_samples = Experience.concat_experiences(micro_batch, self.tokenizer.pad_token_id)
                samples_list.append(concat_samples)
        else:
            batch_size = self.args.micro_rollout_batch_size     # 按照 micro_rollout_batch_size 的大小分组
            for i in range(0, len(rollout_samples), batch_size):
                concat_samples = Experience.concat_experiences(
                    rollout_samples[i : i + batch_size], self.tokenizer.pad_token_id
                )
                samples_list.append(concat_samples)
        return samples_list

    @torch.no_grad()
    def make_experience_batch(self, rollout_samples) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        # Each batch of samples will be scheduled to a effective Ray Actor (i.e, a DP rank)
        samples_list = self.split_rollout_samples(rollout_samples)   # list里 micro_rollout_batch_size个experience打包成一个experience
        print("finish split samples")
        print(f"sample list里的样本数量是{len(samples_list)}")
        print(samples_list[0])
        # Make experiences (models forward: logprobs, values, rewards, and kl divergence)
        experiences = self.make_experience(samples_list)

        # Process experiences (reward shaping, etc.)
        experiences = self.compute_advantages_and_returns(experiences)
        return experiences

    @torch.no_grad()
    def make_experience(self, samples_list: List[Experience]) -> List[Experience]:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        start_time = time.time()
        logger.info(f"🚀 Starting experience making with {sum([len(s.sequences) for s in samples_list])} samples")

        args = self.strategy.args
        device = "cpu"

        # Extract all information from samples in one pass
        # Convert samples into lists of tensors and metadata for batch processing
        sequences_list = [s.sequences for s in samples_list]
        attention_mask_list = [s.attention_mask for s in samples_list]
        action_mask_list = [s.action_mask for s in samples_list] 
        reward_action = []  # 恢复为减少前的 action_mask（在前面补一个 0），输出形状统一为 [bs, L+1]
        for am in action_mask_list:
            # 规范形状为 [bs, L]
            print(f"am的形状是{am.shape}")
            if am.dim() == 1:
                am_2d = am.unsqueeze(0)
            elif am.dim() == 2:
                am_2d = am
            else:
                am_2d = am.view(am.size(0), -1)

            bs = am_2d.size(0)
            leading_zero_col = torch.zeros(bs, 1, dtype=am_2d.dtype, device=am_2d.device)
            full_mask_2d = torch.cat([leading_zero_col, am_2d], dim=1)
            print(f"full_mask_2d的形状是{full_mask_2d.shape}")
            reward_action.append(full_mask_2d)

        # The rewards are already filled in the samples_list, such as the agent's environment rewards
        if samples_list[0].rewards is not None:
            pass
        elif self.remote_rm_url:
            print("no rewards and now get")
            queries_list = sum(
                [
                    self.tokenizer.batch_decode(remove_pad_token(seq, action_mask), skip_special_tokens=False)
                    for seq, action_mask in zip(sequences_list, reward_action)
                ],
                [],
            )
            # print(queries_list[0])
            print("finish decode response for rewards")
            prompts_list = sum([s.prompts for s in samples_list], [])
            labels_list = sum([s.labels for s in samples_list], [])
            # Keep the remote call asynchronous
            r_refs = self.remote_reward_model.get_rewards.remote(queries_list, prompts_list, labels_list)
            print("wait for getting rewards")
        else:
            # Batch call reward model
            r_refs = self.reward_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                attention_mask=attention_mask_list,
                pad_sequence=[True] * len(samples_list),
            )

        # Sync to avoid GPU OOM when colocate models
        if args.colocate_all_models and not self.remote_rm_url:
            ray.get(r_refs)
            ray.get(self.reward_model_group.async_run_method(method_name="empty_cache"))

        # Batch call actor model
        action_log_probs_ref = self.actor_model_group.async_run_method_batch(
            method_name="forward",
            sequences=sequences_list,
            action_mask=action_mask_list,
            attention_mask=attention_mask_list,
        )     # old策略输出的概率
        print("wait for getting old logit_probs")

        # Sync to avoid GPU OOM when colocate models
        if args.colocate_all_models or args.colocate_actor_ref:
            ray.get(action_log_probs_ref)
            ray.get(self.actor_model_group.async_run_method(method_name="empty_cache"))

        # Batch call critic model
        if self.critic_model_group is not None:
            if args.colocate_critic_reward and not self.remote_rm_url:
                ray.get(r_refs)
                ray.get(self.reward_model_group.async_run_method(method_name="empty_cache"))

            value_ref = self.critic_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                action_mask=action_mask_list,
                attention_mask=attention_mask_list,
            )
            if args.colocate_all_models or args.colocate_critic_reward:
                ray.get(value_ref)
                ray.get(self.critic_model_group.async_run_method(method_name="empty_cache"))
        else:
            value_ref = ray.put([[None]] * (len(samples_list) * args.ring_attn_size * args.ds_tensor_parallel_size))

        # Batch call initial model
        if self.initial_model_group is not None:
            base_action_log_probs_ref = self.initial_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                action_mask=action_mask_list,
                attention_mask=attention_mask_list,
            )     # ref策略输出的概率
            print("wait for getting ref logit_probs")

            if args.colocate_all_models or args.colocate_actor_ref:
                ray.get(base_action_log_probs_ref)
                ray.get(self.initial_model_group.async_run_method(method_name="empty_cache"))
        else:
            base_action_log_probs_ref = ray.put(
                [[None]] * (len(samples_list) * args.ring_attn_size * args.ds_tensor_parallel_size)
            )

        # Wait for all remote calls to complete and flatten the results
        # Note: the results duplicated ring_attn_size * ds_tensor_parallel_size times
        # This is because the actors in ring group and tp group will return the same output
        duplicate_factor = args.ring_attn_size * args.ds_tensor_parallel_size
        print("get old logit_probs")
        action_log_probs_list = sum(ray.get(action_log_probs_ref)[::duplicate_factor], [])
        print("get ref logit_probs")
        base_action_log_probs_list = sum(ray.get(base_action_log_probs_ref)[::duplicate_factor], [])
        print("finish logit_probs")
        value_list = sum(ray.get(value_ref)[::duplicate_factor], [])

        # Process rewards based on source
        if samples_list[0].rewards is not None:
            pass
        elif self.remote_rm_url:
            # Get rewards info from remote model
            print("get rewards now")
            rewards_info = ray.get(r_refs)
            # Process rewards and scores
            print("update rewards")
            update_samples_with_rewards(rewards_info, samples_list)
        else:
            # Reward Model
            rewards_list = sum(ray.get(r_refs)[::duplicate_factor], [])
            for i, samples in enumerate(samples_list):
                samples.rewards = rewards_list[i]
                samples.info["reward"] = rewards_list[i]

        assert (
            len(samples_list) == len(action_log_probs_list) == len(base_action_log_probs_list) == len(value_list)
        ), f"len(samples_list): {len(samples_list)}, len(action_log_probs_list): {len(action_log_probs_list)}, len(base_action_log_probs_list): {len(base_action_log_probs_list)}, len(value_list): {len(value_list)}"

        # Process results for each sample
        for i, (samples, action_log_probs, base_action_log_probs, value) in enumerate(
            zip(samples_list, action_log_probs_list, base_action_log_probs_list, value_list)
        ):
            if (self.initial_model_group is not None) and (not args.use_kl_loss):
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    kl_estimator=self.strategy.args.kl_estimator,
                )
            else:
                kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)
            kl_mean = masked_mean(kl, samples.action_mask, dim=-1)

            if not args.use_kl_loss:
                base_action_log_probs = None

            # Update experience with new information
            samples.action_log_probs = action_log_probs    # old的action_log_probs
            samples.base_action_log_probs = base_action_log_probs     
            samples.values = value
            samples.kl = kl
            samples.info["kl"] = kl_mean

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ Experience making completed in {time_str}")
        return samples_list

    @torch.no_grad()
    def compute_advantages_and_returns(
        self, experiences: List[Experience], **kwargs
    ) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.
        Example, use_dynamic_batch
            >>> rewards: [0, 1, 0.5, 1], indices: [1, 2, 0, 3], n_samples_per_prompt: 2
            >>> sorted rewards: [0,5, 0, 1, 1], reward shaping: [0.25, 0.25, 1, 1]
            >>> map back: [0.25, 1, 0.25, 1]
        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args

        # DAPO reward shaping with optional overlong penalty - Apply BEFORE dynamic indices processing
        if args.overlong_buffer_len is not None:
            assert (
                args.generate_max_len >= args.overlong_buffer_len
            ), "generate_max_len must be larger than overlong_buffer_len"
            overlong_buffer_len = args.overlong_buffer_len
            expected_len = args.generate_max_len - overlong_buffer_len
            overlong_penalty_factor = args.overlong_penalty_factor

            # Apply penalty to each experience's rewards based on response length
            for experience in experiences:
                response_lengths = experience.info["response_length"]
                batch_size = len(response_lengths)
                for j in range(batch_size):
                    valid_response_length = response_lengths[j].item()
                    # Cap the exceed_len to overlong_buffer_len to prevent excessive penalty
                    exceed_len = min(valid_response_length - expected_len, overlong_buffer_len)
                    if exceed_len > 0:
                        overlong_penalty = -exceed_len / overlong_buffer_len * overlong_penalty_factor
                        # Apply penalty to the j-th reward in this experience
                        experience.rewards[j] += overlong_penalty

        # 获取每个 experience 中样本的数量（用于后续分割）
        # exp_len[i] 表示第 i 个 experience 包含多少个样本
        exp_len = [len(experience.index) for experience in experiences]    
        
        # 构建原始位置索引映射
        # indices 的作用：
        # - 当不使用动态批处理时：indices = [0, 1, 2, 3, ...]（身份映射）
        # - 当使用动态批处理时：indices 记录了每个样本在原始 rollout_samples 中的位置
        # 例如：如果样本被重新排列，indices 可能是 [2, 0, 4, 1, 3, ...]
        indices = torch.tensor(sum([experience.index for experience in experiences], []))
        
        # 将所有 experience 的奖励拼接成一个大的张量
        # raw_rewards 的形状：(total_samples,)，包含所有样本的原始奖励
        # 注意：这里的顺序可能与原始生成顺序不同（如果使用了动态批处理）
        raw_rewards = torch.cat([experience.rewards for experience in experiences], dim=0)
         
        # 创建一个空的奖励张量，用于存储重排序后的奖励
        rewards = torch.empty_like(raw_rewards)
        
        # 使用 indices 将原始奖励重新排序到正确位置
        # rewards[indices] = raw_rewards 的含义：
        # - 将 raw_rewards[0] 放到 rewards[indices[0]] 位置
        # - 将 raw_rewards[1] 放到 rewards[indices[1]] 位置
        # - 以此类推，恢复原始的顺序
        rewards[indices] = raw_rewards  # sorted
        
        # 将一维奖励重塑为二维，便于组内处理
        # 重塑后的形状：(num_prompts, n_samples_per_prompt)
        # 例如：如果有 3 个 prompt，每个 prompt 生成 2 个样本，则形状为 (3, 2)
        # 这样便于后续进行 GRPO 的组内奖励整形（如减去组内均值、计算组内标准差等）
        rewards = rewards.reshape(-1, args.n_samples_per_prompt)

        # log group reward std
        if args.n_samples_per_prompt > 1:
            group_reward_stds = (
                rewards.std(-1, keepdim=True).repeat(1, args.n_samples_per_prompt).reshape(-1)[indices].split(exp_len)
            )
            for experience, group_reward_std in zip(experiences, group_reward_stds):
                experience.info["group_reward_std"] = group_reward_std

        # reward shaping
        if args.advantage_estimator == "rloo":
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
        elif args.advantage_estimator in ["reinforce_baseline", "dr_grpo"]:
            # REINFORCE++-baseline and Dr. GRPO removed the `/std` in GRPO as `/ std` is not needed in RL variance reduction theory.
            # And `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
            rewards = rewards - rewards.mean(-1, keepdim=True)
        elif args.advantage_estimator == "group_norm":
            rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)

        rewards = rewards.reshape(-1)[indices].split(exp_len)  # 展平为一维，然后根据exp_len分割成元组，元组中有多个[bs]大小的张量，此时的reward已经是归一化的优势
        # print(f"此时归一化后的奖励是：{rewards}")
        # for experience, reward in zip(experiences, rewards):
        #     experience.advantages = reward


        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                reward_clip_range=args.reward_clip_range,
            )
            # print(f"此时reward计算完是：{reward},形状是{reward.shape}")
            # time.sleep(10)

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    args.gamma,
                    args.lambd,
                )
            elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo"]:
                if args.gamma != 1.0 and self.advantage_estimator in [
                    "rloo",
                    "reinforce_baseline",
                    "group_norm",
                    "dr_grpo",
                ]:
                    logger.warning("gamma is set to 1.0 for rloo, reinforce_baseline, and group_norm")
                    args.gamma = 1.0

                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    args.gamma,
                )
                experience.advantages = deepcopy(experience.returns)
                # print(f"此时返回是：{experience.returns},形状是{experience.returns.shape}")
                # print(f"此时优势是：{experience.advantages},形状是{experience.advantages.shape}")
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            return_sums = reward.sum(dim=-1)
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None

        # Normalize advantages across all experiences for GAE, REINFORCE, and REINFORCE-baseline
        if self.args.advantage_estimator in ["gae", "reinforce", "reinforce_baseline"]:
            all_advantages = []
            all_action_masks = []
            for exp in experiences:
                all_advantages.append(exp.advantages.flatten())
                all_action_masks.append(exp.action_mask.flatten())

            advantages_vector = torch.cat(all_advantages, dim=0).float()
            action_masks_vector = torch.cat(all_action_masks, dim=0)
            num_actions = action_masks_vector.sum()

            # mean
            mean = (advantages_vector * action_masks_vector).sum() / num_actions
            # std
            if not self.args.no_advantage_std_norm:
                var = ((advantages_vector - mean).pow(2) * action_masks_vector).sum() / num_actions
                rstd = var.clamp(min=1e-8).rsqrt()
            else:
                rstd = 1

            # Apply normalization to each experience
            for exp in experiences:
                exp.advantages = (exp.advantages - mean) * rstd

        return experiences

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """
        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns
