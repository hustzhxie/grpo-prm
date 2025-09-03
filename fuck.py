import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F


def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res


model_name = "Qwen/Qwen2.5-Math-PRM-7B"
device = "auto"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, 
    device_map=device, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()

data = {
    "system": "Please reason step by step, and put your final answer within \\boxed{}.",
    "query": "Compute: $9 \\cdot \\frac{1}{13} \\cdot 26.$",
    "response": [
      "To solve the given problem, we follow the steps below:\n\nFirst, we have the expression:\n\\[9 \\cdot \\frac{1}{13} \\cdot 26.\\]\n\nWe can rearrange the multiplication to make it easier to compute:\n\\[9 \\cdot 26 \\cdot \\frac{1}{13}.\\]\n\nNext, we can simplify the multiplication of $26$ and $\\frac{1}{13}$:\n\\[26 \\cdot \\frac{1}{13} = 2.\\]\n\nNow, we multiply the result by $9$:\n\\[9 \\cdot 2 = 18.\\]\n\nTherefore, the final answer is:\n\\[Final\\ Answer: \\boxed{18}.\\]\nAnswer: \\boxed{18}"]
}

messages = [
    {"role": "system", "content": data['system']},
    {"role": "user", "content": data['query']},
    {"role": "assistant", "content": "<extra_0>".join(data['response']) + "<extra_0>"},
]
conversation_str = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=False
)

input_ids = tokenizer.encode(
    conversation_str, 
    return_tensors="pt", 
).to(model.device)

outputs = model(input_ids=input_ids)

step_sep_id = tokenizer.encode("<extra_0>")[0]
token_masks = (input_ids == step_sep_id)
step_reward = make_step_rewards(outputs[0], token_masks)
print(step_reward)  # [[1.0, 0.1904296875, 0.9765625, 1.0]]

