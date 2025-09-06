import json
from ..utils.remote_rm_utils import ProcessRewardModel
from datasets import load_dataset, load_from_disk

def extract_steps_from_response(response_text: str):
        # 按 \n\n 切分，去掉首尾空白，并过滤空串
        return [s.strip() for s in response_text.split("\n\n") if s.strip()]

if __name__ == "__main__":
    data = []
    rm_path = "Qwen/Qwen2.5-Math-PRM-7B"
    eval_path = "/home/yanzhe/OpenRLHF/test_datasets/HuggingFaceH4_MATH-500_test"
    with open("outputs/math_sft_grpo.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))
    print(f"response total: {len(data)}")
    print("load reward model")
    reward_model = ProcessRewardModel(rm_path)
    print("load eval dataset")
    dataset = load_from_disk(eval_path)
    answers = []
    for d in dataset:
        answers.append(d["answer"])
    print(f"eval dataset total: {len(answers)}")
    step_rewards = []
    outcome_reward = []
    step_test = []
    count_5_lsit = []
    count_6_lsit = []
    count_7_lsit = []
    count_8_lsit = []
    count_9_lsit = []
    count_9_5_lsit = []
    step_length = []
    for a, ans in zip(data, answers):
        query = a["input"]
        response = a["output"]
        step_reward = reward_model.compute_step_rewards(response, query)
        step_rewards.append(step_reward)
        outcome_reward.append(reward_model.get_outcome_reward(response, ans))
        count_5 = sum([1 for r in step_reward if r > 0.5])
        count_6 = sum([1 for r in step_reward if r > 0.6])
        count_7 = sum([1 for r in step_reward if r > 0.7])
        count_8 = sum([1 for r in step_reward if r > 0.8])
        count_9 = sum([1 for r in step_reward if r > 0.9])
        count_9_5 = sum([1 for r in step_reward if r > 0.95])
        count_5_lsit.append(count_5)
        count_6_lsit.append(count_6)
        count_7_lsit.append(count_7)
        count_8_lsit.append(count_8)
        count_9_lsit.append(count_9)
        count_9_5_lsit.append(count_9_5)
        step_length.append(len(extract_steps_from_response(response)))
        step_test.append(sum(step_reward)/len(step_reward))

    print(f"整体的准确率: {sum(outcome_reward)/len(outcome_reward)}")
    print(f"整体的步骤: 0.5:{sum(count_5_lsit)}  0.6:{sum(count_6_lsit)}    0.7:{sum(count_7_lsit)}   0.8:{sum(count_8_lsit)}    0.9:{sum(count_9_lsit)}    0.95:{sum(count_9_5_lsit)}")
    print(f"整体的步骤: 总步数:{sum(step_length)}   0.5:{sum(count_5_lsit)/sum(step_length)}   0.6:{sum(count_6_lsit)/sum(step_length)}   0.7:{sum(count_7_lsit)/sum(step_length)}   0.8:{sum(count_8_lsit)/sum(step_length)}   0.9:{sum(count_9_lsit)/sum(step_length)}   0.95:{sum(count_9_5_lsit)/sum(step_length)}")
    print(f"整体的步骤质量: {sum(step_test)/len(step_test)}")

