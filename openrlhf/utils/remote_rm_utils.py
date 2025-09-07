# import time
# import ray
# import requests

# from openrlhf.utils.logging_utils import init_logger

# logger = init_logger(__name__)


# def request_api_wrapper(url, data, try_max_times=5):
#     """Synchronous request API wrapper"""
#     headers = {
#         "Content-Type": "application/json",
#     }
#     for _ in range(try_max_times):
#         try:
#             response = requests.post(url=url, json=data, headers=headers, timeout=180)
#             response.raise_for_status()  # Raise an HTTPError for bad responses
#             response = response.json()
#             return response
#         except requests.RequestException as e:
#             logger.info(f"Request error, please check: {e}")
#         except Exception as e:
#             logger.info(f"Unexpected error, please check: {e}")
#         time.sleep(1)

#     raise Exception(f"Request error for {try_max_times} times, returning None. Please check the API server.")


# @ray.remote
# def remote_rm_fn_ray(api_url, queries, prompts, labels):
#     return request_api_wrapper(api_url, {"query": queries, "prompts": prompts, "labels": labels})


# @ray.remote
# class RemoteRewardModel:
#     def __init__(self, args, remote_rm_url):
#         self.args = args
#         self.remote_rm_url = [remote_rm_url] if isinstance(remote_rm_url, str) else remote_rm_url
#         self.custom_reward_func = None

#         if self.remote_rm_url and self.remote_rm_url[0].endswith(".py"):
#             print(f"Loading custom `reward_func(queries, prompts, labels)` from {self.remote_rm_url[0]}")
#             import importlib.util

#             spec = importlib.util.spec_from_file_location("reward_func", self.remote_rm_url[0])
#             reward_module = importlib.util.module_from_spec(spec)
#             spec.loader.exec_module(reward_module)
#             self.custom_reward_func = ray.remote(reward_module.reward_func)

#     def get_rewards(self, queries_list, prompts_list, labels_list):
#         if self.custom_reward_func:
#             # Let Ray automatically distribute the workload across available resources
#             batch_size = self.args.micro_rollout_batch_size
#             num_chunks = (len(queries_list) + batch_size - 1) // batch_size
#             r_refs = []
#             for i in range(num_chunks):
#                 start_idx = i * batch_size
#                 end_idx = min((i + 1) * batch_size, len(queries_list))
#                 r = self.custom_reward_func.remote(
#                     queries_list[start_idx:end_idx],
#                     prompts_list[start_idx:end_idx],
#                     labels_list[start_idx:end_idx],
#                 )
#                 r_refs.append(r)
#         else:
#             # Distribute data across different remote reward function servers
#             num_servers = len(self.remote_rm_url)
#             batch_size = (len(queries_list) + num_servers - 1) // num_servers
#             r_refs = []
#             for i in range(num_servers):
#                 start_idx = i * batch_size
#                 end_idx = min((i + 1) * batch_size, len(queries_list))
#                 rm = self.remote_rm_url[i]
#                 r = remote_rm_fn_ray.remote(
#                     rm,
#                     queries=queries_list[start_idx:end_idx],
#                     prompts=prompts_list[start_idx:end_idx],
#                     labels=labels_list[start_idx:end_idx],
#                 )
#                 r_refs.append(r)

#         return ray.get(r_refs)


# import torch
# import ray
# from transformers import AutoModelForTokenClassification, AutoTokenizer
# import os

# class ProcessRewardModel:
#     """专门的过程奖励模型"""
    
#     def __init__(self, remote_rm_url, device_map="auto"):
#         self.model_path = remote_rm_url
#         self.device_map = device_map
#         self.model = None
#         self.tokenizer = None
#         self.load_model()
    
#     def load_model(self):
#         """加载PRM模型"""
#         try:
#             print(f"Loading PRM model from: {self.model_path}")
#             # 明确选择设备，避免 device_map=auto 将模型放在 CPU 导致 GPU 空闲
#             if torch.cuda.is_available():
#                 self.device = torch.device("cuda")
#                 preferred_dtype = torch.float16
#             else:
#                 self.device = torch.device("cpu")
#                 preferred_dtype = torch.float32
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 self.model_path, 
#                 trust_remote_code=True,
#             )
#             self.model = AutoModelForTokenClassification.from_pretrained(
#                 self.model_path,
#                 torch_dtype=preferred_dtype,
#                 trust_remote_code=True,
#             ).to(self.device).eval()
#             # 打印设备信息，确认是否在 GPU 上
#             first_param_device = next(self.model.parameters()).device
#             print(f"PRM model loaded successfully on device: {first_param_device}, dtype: {preferred_dtype}")
#         except Exception as e:
#             print(f"Failed to load PRM model: {e}")
#             raise
    
#     def compute_rewards(self, queries, prompts, labels):
#         """计算过程奖励"""
#         rewards_info = []

#         print("begin rewards")
#         try:
#             # 批量计算每个样本的 step 奖励
#             batch_step_rewards = self.compute_step_rewards_batch(queries, max_batch_size=32)
#             for step_rewards in batch_step_rewards:
#                 step_tensor = torch.tensor(step_rewards)
#                 scores = step_tensor.sum()
#                 scores_tensor = scores.unsqueeze(0)
#                 rewards_info.append({
#             "rewards": scores_tensor,    # 此处有修改
#             "scores": scores_tensor,
#             "extra_logs": {"dummy_scores": scores_tensor}
#         })
#         except Exception as e:
#             print(f"Error computing batch rewards: {e}")
#             # 兜底：逐个计算，避免整批失败
#             # for query in queries:
#             #     try:
#             #         step_rewards = self.compute_step_rewards(query)
#             #         step_tensor = torch.tensor(step_rewards)
#             #         rewards.append(step_tensor)
#             #         scores.append(step_tensor.sum())
#             #         scores_tensor = torch.tensor(scores)
#             #     except Exception as ie:
#             #         print(f"Error computing reward for query: {ie}")
#             #         rewards.append(torch.tensor([0.5]))
#             # scores.append(torch.tensor([0.5]))
#         print("finish rewards")
#         print(scores_tensor)
#         return rewards_info
#     def make_step_rewards(self, logits, token_masks):
#         """计算每个step的奖励分数"""
#         all_scores_res = []
#         for sample, token_mask in zip(logits, token_masks):
#             # sample: (seq_len, num_labels)
#             probs = sample[token_mask].softmax(dim=-1)  # (num_steps, 2)
#             process_reward = probs[:, 1] - probs[:, 0]  # (num_steps,)
#             # weighted sum to approx. min, highly recommend when BoN eval and Fine-tuning LLM
#             # weight = torch.softmax(
#             #     -process_reward / 0.1, 
#             #     dim=-1,
#             # )
#             # process_reward = weight * process_reward
#             all_scores_res.append(process_reward.cpu().tolist())
#         return all_scores_res

#     def compute_step_rewards(self, query):
#         """计算每个step的奖励"""
#         try:
#             step_separator = "\n\n"  # 分隔步骤的标识符（使用///分隔）

#             # 分割query获取问题部分和各个步骤
#             parts = query.split(step_separator)
            
#             if len(parts) < 2:
#                 # 如果没有步骤，返回默认奖励
#                 return [0.5]
            
#             # 第一部分通常是问题，剩余部分是步骤
#             question = parts[0].strip()
#             steps = [step.strip() for step in parts[1:] if step.strip()]
            
#             if not steps:
#                 return [0.5]
            
#             # 设置step分隔符token（仍然使用换行符作为模型输入的分隔符）
#             step_separator_token = self.tokenizer(
#                 "@@", 
#                 add_special_tokens=False, 
#                 return_tensors='pt',
#             )['input_ids']
            
#             # 按照示例脚本的逻辑构建输入序列
#             input_ids = self.tokenizer(
#                 question, 
#                 add_special_tokens=False, 
#                 return_tensors='pt',
#             )['input_ids']
            
#             score_ids = []
#             for step in steps:
#                 step_ids = self.tokenizer(
#                     step, 
#                     add_special_tokens=False, 
#                     return_tensors='pt',
#                 )['input_ids']
#                 input_ids = torch.cat(
#                     [input_ids, step_ids, step_separator_token], 
#                     dim=-1,
#                 )
#                 score_ids.append(input_ids.size(-1) - 1)  # 记录分隔符的最后一个token位置
            
#             # 移动到模型设备
#             input_ids = input_ids.to(self.model.device)
            
#             # 创建token_masks，标记分隔符位置（与示例脚本一致）
#             token_masks = torch.zeros_like(input_ids, dtype=torch.bool)
#             token_masks[0, score_ids] = True
            
#             # 验证分隔符位置（与示例脚本一致）
#             assert torch.all(input_ids[token_masks].to("cpu") == step_separator_token)
            
#             print("begin step rewards")
#             # 获取模型输出并计算奖励
#             with torch.inference_mode():
#                 if self.model.device.type == "cuda":
#                     with torch.cuda.amp.autocast(dtype=torch.float16):
#                         logits = self.model(input_ids).logits
#                 else:
#                     logits = self.model(input_ids).logits
#                 print("compute step rewards")
#                 step_rewards = self.make_step_rewards(logits, token_masks)
            
#             # 返回第一个样本的奖励（因为这里只处理一个query）
#             return step_rewards[0] if step_rewards else [0.5]
            
#         except Exception as e:
#             print(f"Error in compute_step_rewards: {e}")
#             # 返回默认奖励
#             return [0.5]

#     def _prepare_query_tensors(self, query):
#         """为单个样本构建 input_ids 与 token_mask 张量，返回 (ids[1,L], mask[1,L])"""
#         step_separator = "\n\n"
#         parts = query.split(step_separator)
#         if len(parts) < 2:
#             return None

#         question = parts[0].strip()
#         steps = [step.strip() for step in parts[1:] if step.strip()]
#         if not steps:
#             return None

#         step_separator_token = self.tokenizer(
#             "@@",
#             add_special_tokens=False,
#             return_tensors='pt',
#         )['input_ids']

#         input_ids = self.tokenizer(
#             question,
#             add_special_tokens=False,
#             return_tensors='pt',
#         )['input_ids']

#         score_ids = []
#         for step in steps:
#             step_ids = self.tokenizer(
#                 step,
#                 add_special_tokens=False,
#                 return_tensors='pt',
#             )['input_ids']
#             input_ids = torch.cat([input_ids, step_ids, step_separator_token], dim=-1)
#             score_ids.append(input_ids.size(-1) - 1)

#         token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
#         token_mask[0, score_ids] = True

#         # 验证分隔符位置
#         assert torch.all(input_ids[token_mask].to("cpu") == step_separator_token)
#         return input_ids, token_mask

#     def compute_step_rewards_batch(self, queries, max_batch_size=8):
#         """批量计算每个step的奖励，提升 GPU 利用率"""
#         all_scores = []
#         pad_id = self.tokenizer.pad_token_id
#         if pad_id is None:
#             pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

#         for start in range(0, len(queries), max_batch_size):
#             print("begin a batch")
#             batch_queries = queries[start:start + max_batch_size]

#             prepped = [self._prepare_query_tensors(q) for q in batch_queries]
#             # 对于无法解析的样本，用默认分数占位
#             valid_indices = [i for i, x in enumerate(prepped) if x is not None]
#             if not valid_indices:
#                 all_scores.extend([[0.5] for _ in batch_queries])
#                 continue

#             ids_list = [prepped[i][0].squeeze(0) for i in valid_indices]  # [L]
#             mask_list = [prepped[i][1].squeeze(0) for i in valid_indices]  # [L]

#             ids_padded = torch.nn.utils.rnn.pad_sequence(ids_list, batch_first=True, padding_value=pad_id)
#             masks_padded = torch.nn.utils.rnn.pad_sequence(mask_list, batch_first=True, padding_value=0)

#             ids_padded = ids_padded.to(self.model.device)
#             masks_padded = masks_padded.to(self.model.device)

#             with torch.inference_mode():
#                 if self.model.device.type == "cuda":
#                     with torch.cuda.amp.autocast(dtype=torch.float16):
#                         logits = self.model(ids_padded).logits
#                 else:
#                     logits = self.model(ids_padded).logits

#             batch_scores = self.make_step_rewards(logits, masks_padded)

#             # 写回到对应位置，无法解析的样本用默认分数
#             out_ptr = 0
#             for i in range(len(batch_queries)):
#                 if i in valid_indices:
#                     all_scores.append(batch_scores[out_ptr])
#                     out_ptr += 1
#                 else:
#                     all_scores.append([0.5])

#         return all_scores

# @ray.remote
# class RemoteRewardModel:
#     """远程过程奖励模型"""
    
#     def __init__(self, args, remote_rm_url):
#         self.process_rm = ProcessRewardModel(remote_rm_url)
    
#     def get_rewards(self, queries, prompts, labels):
#         return self.process_rm.compute_rewards(queries, prompts, labels)



import torch
import ray
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import os
from tqdm import tqdm
import math
import re

class ProcessRewardModel:
    """专门的过程奖励模型"""
    
    def __init__(self, remote_rm_url, device_map="auto"):
        self.model_path = remote_rm_url
        self.device_map = device_map
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """加载PRM模型"""
        try:
            print(f"Loading PRM model from: {self.model_path}")
            # 明确选择设备，避免 device_map=auto 将模型放在 CPU 导致 GPU 空闲
            # if torch.cuda.is_available():
                # self.device = torch.device("cuda")
                # preferred_dtype = torch.bfloat16
            # else:
            #     self.device = torch.device("cpu")
            #     preferred_dtype = torch.float32
            self.device = torch.device("cuda")
            preferred_dtype = torch.bfloat16
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True,
            )
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=preferred_dtype,
                trust_remote_code=True,
            ).to(self.device).eval()
            # 记录 step 分隔 token id（<extra_0>）
            self.step_sep_id = self.tokenizer.encode("<extra_0>")[0]
            # 打印设备信息，确认是否在 GPU 上
            first_param_device = next(self.model.parameters()).device
            print(f"PRM model loaded successfully on device: {first_param_device}, dtype: {preferred_dtype}")
        except Exception as e:
            print(f"Failed to load PRM model: {e}")
            raise

    def process_reward(self, outcome_reward, step_rewards, k_method):    # step_rewards是步骤奖励的列表
        def compute_entropy(step_reward):
            # 计算熵
            epsilon = 1e-8  # 添加一个小的正数避免对 0 或负数取对数
            step_reward = max(epsilon, min(1 - epsilon, step_reward))
            entropy = -((step_reward) * math.log(step_reward) + (1 - step_reward) * math.log(1 - step_reward))
            return entropy

        lamda = 0.5     # 阈值
        k = 2       # 衰减系数
        scores = 1.0
        N = len(step_rewards)    # 步骤的数量
        a = 1     # k计算时用 p2 方法的系数
        b = 1     # k计算时用 p3 方法的系数

        if outcome_reward == 1.0:
            # 结果正确的情况
            for i, step_reward in enumerate(step_rewards):
                if step_reward < lamda:
                    step_entropy = compute_entropy(step_reward)
                    if k_method == "p1":
                        k_d = k * (1 - step_entropy/(2*compute_entropy(0.5)))      # 这里衰减系数k可以用别的方法确定
                    elif k_method == "p2":
                        k_d = k * math.epx(-a * step_entropy)
                    elif k_method == "p3":
                        k_d = k / (1 + math.exp(b * (step_entropy - compute_entropy(0.5))))      # 这里衰减系数k可以用别的方法确定
                    scores = scores * math.exp(-k_d * (N - i - 1)/N)
                    return scores
            return scores
        else:
            # 结果错误的情况
            for i, step_reward in enumerate(step_rewards):
                if step_reward < lamda: 
                    if i == 0:
                        return 0.0
                    else:
                        step_entropy = compute_entropy(step_rewards[i-1])
                        if k_method == "p1":
                            k_d = k * (1 - step_entropy/(2*compute_entropy(0.5)))      # 这里衰减系数k可以用别的方法确定
                        elif k_method == "p2":
                            k_d = k * math.epx(-a * step_entropy)
                        elif k_method == "p3":
                            k_d = k / (1 + math.exp(b * (step_entropy - compute_entropy(0.5))))      # 这里衰减系数k可以用别的方法确定
                        scores = scores * (1 - math.exp(-k_d * i/N))
                        return scores
            return scores * (1 - math.exp(-k))

    def process_reward_test(self, outcome_reward, step_rewards, k_method):    # step_rewards是步骤奖励的列表
        def compute_entropy(step_reward):
            # 计算熵
            epsilon = 1e-8  # 添加一个小的正数避免对 0 或负数取对数
            step_reward = max(epsilon, min(1 - epsilon, step_reward))
            entropy = -((step_reward) * math.log(step_reward) + (1 - step_reward) * math.log(1 - step_reward))
            return entropy

        lamda = 0.5     # 阈值
        k = 2       # 衰减系数
        scores = 1.0
        N = len(step_rewards)    # 步骤的数量
        a = 1     # k计算时用 p2 方法的系数
        b = 1     # k计算时用 p3 方法的系数
        x = 0

        if outcome_reward == 1.0:
            # 结果正确的情况
            for i, step_reward in enumerate(step_rewards):
                if step_reward < 0.2:  # 对于这样的步骤得分我们认为该步骤一定错误，后面的步骤无效，惩罚最高
                    print(f"步骤{i+1}获得最高惩罚")
                    k_d = k
                    scores = scores * math.exp(-k_d * (N - i - 1)/N)
                    return scores
                elif (step_reward >= 0.2) and (step_reward < 0.4):
                    print(f"步骤{i+1}获得中等惩罚")
                    step_entropy = compute_entropy(step_reward)
                    if k_method == "p1":
                        k_d = k * (1 - step_entropy/(2*compute_entropy(0.5)))      # 这里衰减系数k可以用别的方法确定
                    elif k_method == "p2":
                        k_d = k * math.exp(-a * step_entropy)
                    elif k_method == "p3":
                        k_d = k / (1 + math.exp(b * (step_entropy - compute_entropy(0.5))))      # 这里衰减系数k可以用别的方法确定
                    scores = scores * math.exp(-k_d * (N - i - 1)/N)
                    return scores
                elif (step_reward >= 0.4) and (step_reward < 0.6):
                    print(f"步骤{i+1}获得不自信惩罚")
                    step_entropy = compute_entropy(step_reward)
                    scores -= (step_entropy/compute_entropy(0.5))/N 
                    continue
            return scores
        else:
            # 结果错误的情况
            for i, step_reward in enumerate(step_rewards):
                if step_reward < lamda: 
                    if i == 0:
                        return 0.0
                    else:
                        j=i
                        for m, step_reward in enumerate(step_rewards): 
                            if(m<=j):
                                scores += step_reward*(0.9**m)   # 这里的0.9是一个衰减系数，可以调整
                        #scores=scores/(j+1)/1.5
                        scores=scores/ sum (0.9**m for m in range(j+1))/5
                        for m, step_reward in enumerate(step_rewards): 
                            if(m>j):
                                x+=step_reward
                        x=x/(N-j)
                        if(x>0.5):
                            scores+=0.1
                        return scores
            return scores * (1 - math.exp(-k))


    def get_outcome_reward(self, response: str, gt: str) -> float:
        """
        response: 模型输出（包含推理步骤 + 最终答案）
        gt: ground truth 正确答案
        """
        # 匹配 \boxed{ ... } 中的内容
        match = re.search(r"\\boxed\{([^}]*)\}", response)
        
        if not match:
            # 没找到 boxed 答案 → reward = 0
            return 0.0

        pred = match.group(1).strip()   # 提取预测答案
        
        # 比较预测答案和gt
        if pred == gt.strip():
            return 1.0
        else:
            return 0.0


    def extract_steps_from_response(self, response_text: str):
        # 按 \n\n 切分，去掉首尾空白，并过滤空串
        return [s.strip() for s in response_text.split("\n\n") if s.strip()]
    
    def compute_rewards(self, queries, prompts, labels):
        """计算过程奖励"""
        rewards_info = []
        print("begin rewards")
        k_method = "p1"     # 给出三个计算 k 的 method p1, p2, p3
        for query, prompt, label in tqdm(zip(queries, prompts, labels), total=len(queries), desc="Computing rewards"):
            try:
                outcome_reward = self.get_outcome_reward(query, label)
                step_rewards = self.compute_step_rewards(query, prompt)
                print(f"结果奖励是{outcome_reward}")
                print(step_rewards[0])
                print(len(step_rewards))
            except Exception as e:
                print(f"Error computing reward for one query: {e}")
                step_rewards = [0.5]
            # step_tensor = torch.tensor(step_rewards)
            # scores = step_tensor.sum()
            # scores_tensor = scores.unsqueeze(0)
            # scores_tensor = self.process_reward(step_rewards)
            scores = self.process_reward_test(outcome_reward, step_rewards, k_method)
            scores_tensor = torch.tensor([scores])
            rewards_info.append({
                "rewards": scores_tensor,
                "scores": scores_tensor,
                "extra_logs": {"dummy_scores": scores_tensor},
            })
        print("finish rewards")
        for reward_info in rewards_info:
            print(reward_info["rewards"])
        return rewards_info
    def make_step_rewards(self, logits, token_masks):
        """根据新 PRM 机制（softmax + mask 提取 <extra_0> 位置的正类概率）计算每个 step 的分数"""
        # logits: [bs, seq_len, num_labels]
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)

        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]  # [seq_len, num_labels]
            # 仅保留 mask 位置的元素，并重塑为 [num_steps, 2]，取正类概率
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
            all_scores_res.append(positive_probs.detach().cpu().tolist())
        return all_scores_res

    def compute_step_rewards(self, query, prompt):
        """计算每个step的奖励"""
        try:
            # 从回答中抽取步骤，并使用 <extra_0> 连接，构造 Chat 模板输入
            response = self.extract_steps_from_response(query)
            data = {
            "system": "Please reason step by step, and put your final answer within \\boxed{}.",
            "query": prompt[0],
            "response": response}
            # messages = [
            # {"role": "system", "content": data['system']},
            # {"role": "user", "content": data['query']},
            # {"role": "assistant", "content": "<extra_0>".join(data['response']) + "<extra_0>"},]
            # conversation_str = self.tokenizer.apply_chat_template(
            # messages, 
            # tokenize=False, 
            # add_generation_prompt=False)
            conversation_str = prompt + "\n" + "<extra_0>".join(data['response']) + "<extra_0><|im_end|>"

            input_ids = self.tokenizer.encode(
            conversation_str, 
            return_tensors="pt", 
            ).to(self.device)
            # 生成 mask：等于 <extra_0> 的位置为 True
            token_masks = (input_ids == self.step_sep_id)
            
            print("begin step rewards")
            # 获取模型输出并计算奖励
            with torch.inference_mode():
                if self.model.device.type == "cuda":
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                else:
                    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                # 新 PRM 模型的 logits 位于 outputs[0]
                logits = outputs[0]
                print("compute step rewards")
                step_rewards = self.make_step_rewards(logits, token_masks)
            
            # 返回第一个样本的奖励（因为这里只处理一个query）
            return step_rewards[0] if step_rewards else [0.5]
            
        except Exception as e:
            print(f"Error in compute_step_rewards: {e}")
            # 返回默认奖励
            return [0.5]

    def _prepare_query_tensors(self, query):
        """为单个样本构建 input_ids 与 token_mask 张量，返回 (ids[1,L], mask[1,L])"""
        # 直接对传入的 query 进行编码，假设其中的步骤由 <extra_0> 分隔
        input_ids = self.tokenizer.encode(
            query,
            return_tensors="pt",
        )

        # 构造 token mask：等于 <extra_0> 的位置为 True
        token_mask = (input_ids == self.step_sep_id)
        if not token_mask.any():
            return None
        return input_ids, token_mask

@ray.remote(num_gpus=1)
class RemoteRewardModel:
    """远程过程奖励模型"""
    
    def __init__(self, args, remote_rm_url):
        self.process_rm = ProcessRewardModel(remote_rm_url)
    
    def get_rewards(self, queries, prompts, labels):
        return self.process_rm.compute_rewards(queries, prompts, labels)

