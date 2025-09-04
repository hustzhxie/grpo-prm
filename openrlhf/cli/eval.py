import json
from openrlhf.utils.remote_rm_utils import RemoteRewardModel
from dataset import load_dataset


if __name__ == "__main__":
    data = []
    rm_path = "Qwen/Qwen2.5-Math-PRM-7B"
    eval_path = "/home/yanzhe/OpenRLHF/test_datasets/HuggingFaceH4_MATH-500_test"
    with open("outputs/11111.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))
    print(f"response total: {len(data)}")
    print("load reward model")
    reward_model = RemoteRewardModel(rm_path)
    print("load eval dataset")
    dataset = load_dataset(eval_path)
    answers = []
    for d in dataset["train"]:
        answers.append(d["answer"])
    print(f"eval dataset total: {len(answers)}")
    step_rewards = []
    outcome_reward = []
    for a, ans in zip(data, answers):
        query = a["input"]
        response = a["output"]
        step_reward = reward_model.process_rm.compute_step_rewards(response, query)
        step_rewards.append(step_reward)
        outcome_reward.append(reward_model.process_rm.get_outcome_reward(response, query, ans))
        count_5 = sum([1 for r in step_reward if r > 0.5])
        count_8 = sum([1 for r in step_reward if r > 0.8])
        print(f"step rewards的平均值: {sum(step_reward)/len(step_reward)}, 正确可能性大于0.5的个数: {count_5}, 正确可能性大于0.6的个数: {count_8}")

    print(f"整体的准确率: {sum(outcome_reward)/len(outcome_reward)}")

