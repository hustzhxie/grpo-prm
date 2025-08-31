#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
评测脚本：对 MATH-500（或同结构数据）上的模型输出 JSONL 进行准确率评估。

功能：
- 读取参考数据集（本地 save_to_disk 目录或 HF 名称），使用字段 `problem` 与 `answer`；
- 读取一个或多个预测 JSONL（字段默认 `input`/`output`），按 `input` 精确匹配到参考 `problem`；
- 对预测答案与参考答案做轻量归一化后 Exact Match；
- 输出每个预测文件的准确率与样本覆盖情况；
- 若传入多个预测，额外输出对比表。

用法示例：
python examples/scripts/eval_math500.py \
  --dataset_path ./outputs/_datasets/HuggingFaceH4_MATH-500_test \
  --pred_paths ./outputs/qwen_sft_math500.jsonl ./outputs/other.jsonl
"""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple, Optional

import jsonlines


def load_reference_dataset(dataset_path: str, split: str = "test") -> Tuple[Dict[str, str], int]:
    """
    加载参考数据。优先尝试从 `datasets` 的 save_to_disk 目录加载；
    若失败，则尝试当作 HF 数据集名称加载。

    返回：
        problem_to_answer: 映射 {problem_text: answer_text}
        num_examples: 参考集样本总数
    """
    try:
        from datasets import load_from_disk, load_dataset
        if os.path.isdir(dataset_path):
            ds = load_from_disk(dataset_path)
            if split in ds:
                ds = ds[split]
        else:
            ds = load_dataset(dataset_path, split=split)
    except Exception as e:
        raise RuntimeError(f"加载参考数据失败：{e}")

    problem_to_answer: Dict[str, str] = {}
    for row in ds:
        problem = row.get("problem", None)
        answer = row.get("answer", None)
        if problem is None or answer is None:
            # 跳过不完整样本
            continue
        problem_to_answer[str(problem)] = str(answer)
    return problem_to_answer, len(ds)


_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def _extract_boxed(text: str) -> str:
    """若文本中包含 \boxed{...}，优先取其中内容。"""
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1)
    return text


def _strip_latex(text: str) -> str:
    """去除部分常见 LaTeX 标记（轻量处理）。"""
    # 去美元符
    text = text.replace("$", "")
    # 去 \text{...}
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    # 去 \mathrm{...} 等通用命令包装
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    return text


# def normalize_answer(text: str) -> str:
#     """
#     对答案做轻量归一化：
#     1) 提取 \boxed{...} 若存在；
#     2) 去除常见 LaTeX 包裹、前后缀提示词（如 'Answer:' 等）；
#     3) 统一大小写、去空白与部分标点；
#     4) 替换常见 Unicode 符号（如长横）。

#     注：MATH 答案形式多样，此归一化并不保证覆盖所有情况，但足以做快速评估。
#     """
#     if text is None:
#         return ""

#     text = str(text).strip()
#     text = _extract_boxed(text)
#     text = _strip_latex(text)

#     # 去除常见提示前缀
#     prefixes = [
#         r"^answer\s*:\s*",
#         r"^final\s*answer\s*:\s*",
#         r"^答案\s*[:：]\s*",
#     ]
#     for p in prefixes:
#         text = re.sub(p, "", text, flags=re.IGNORECASE)

#     # 统一符号与空白
#     text = text.replace("\u2212", "-")  # Unicode minus
#     text = text.replace(",", "")
#     text = text.strip()

#     # 去掉行尾句号/空白
#     text = text.rstrip(".。 ")
#     # 小写（对大小写不敏感的答案）
#     text = text.lower()

#     # 去除多余空白
#     text = re.sub(r"\s+", "", text)
#     return text
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_ANSWER_BOXED_RE = re.compile(r"Answer: \\boxed\{([^}]*)\}")

def normalize_answer(text: str) -> str:
    """
    对答案做轻量归一化：
    1) 提取 \boxed{...} 若存在；
    2) 去除常见 LaTeX 包裹、前后缀提示词（如 'Answer:' 等）；
    3) 统一大小写、去空白与部分标点；
    4) 替换常见 Unicode 符号（如长横）。

    注：MATH 答案形式多样，此归一化并不保证覆盖所有情况，但足以做快速评估。
    """
    if text is None:
        return ""

    text = str(text).strip()
    
    # 提取 Answer: \boxed{...} 内的内容
    m = _ANSWER_BOXED_RE.search(text)
    if m:
        return m.group(1)

    # 提取 \boxed{...} 内的内容
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1)

    # 去除常见 LaTeX 包裹
    text = _strip_latex(text)

    # 去除常见提示前缀
    prefixes = [
        r"^answer\s*:\s*",
        r"^final\s*answer\s*:\s*",
        r"^答案\s*[:：]\s*",
    ]
    for p in prefixes:
        text = re.sub(p, "", text, flags=re.IGNORECASE)

    # 统一符号与空白
    text = text.replace("\u2212", "-")  # Unicode minus
    text = text.replace(",", "")
    text = text.strip()

    # 去掉行尾句号/空白
    text = text.rstrip(".。 ")
    # 小写（对大小写不敏感的答案）
    text = text.lower()

    # 去除多余空白
    text = re.sub(r"\s+", "", text)
    return text

def read_predictions(jsonl_path: str, input_key: str = "input", output_key: str = "output") -> List[dict]:
    """读取预测 JSONL 列表。"""
    data = []
    with jsonlines.open(jsonl_path, mode="r") as reader:
        for obj in reader:
            if not isinstance(obj, dict):
                continue
            if input_key not in obj or output_key not in obj:
                continue
            data.append({"input": str(obj[input_key]), "output": str(obj[output_key])})
    return data


def _find_ref_by_regex(pattern: str, ref_map: Dict[str, str]) -> Optional[str]:
    """使用正则在参考 problem 中查找第一个匹配项，返回匹配到的 problem 文本。"""
    try:
        regex = re.compile(pattern, flags=re.DOTALL)
    except re.error:
        # 若正则编译失败，退化为转义后的精确匹配
        safe = re.escape(pattern)
        try:
            regex = re.compile(safe, flags=re.DOTALL)
        except re.error:
            return None

    for problem in ref_map.keys():
        if regex.search(problem) is not None:
            return problem
    return None


def evaluate_one(pred_path: str,
                 ref_map: Dict[str, str],
                 pred_input_key: str = "input",
                 pred_output_key: str = "output",
                 match_regex: bool = False) -> dict:
    """
    评测单个预测文件。
    返回：
        {
          'file': ..., 'total_preds': N, 'matched': M, 'correct': C, 'accuracy': C/M,
          'unmatched': N-M
        }
    """
    preds = read_predictions(pred_path, pred_input_key, pred_output_key)

    matched = 0
    correct = 0
    pred_regex_hits = 0  # 统计预测输出中可被正则（Answer:\boxed{...} 或 \boxed{...}）直接提取的数量
    for item in preds:
        inp = item["input"]
        pred = item["output"]
        # 统计正则可提取数量（无论是否能对齐参考题目）
        if _ANSWER_BOXED_RE.search(pred) or _BOXED_RE.search(pred):
            pred_regex_hits += 1
        # 对齐参考样本：支持精确匹配或正则匹配
        if match_regex:
            ref_problem = _find_ref_by_regex(inp, ref_map)
            if ref_problem is None:
                continue
            gold = ref_map[ref_problem]
        else:
            if inp not in ref_map:
                # 无法匹配到参考题目，跳过计分
                continue
            gold = ref_map[inp]
        matched += 1
        print("111")
        print(normalize_answer(pred))
        print("222")
        print(normalize_answer(gold))
        if normalize_answer(pred) == normalize_answer(gold):
            correct += 1

    res = {
        "file": pred_path,
        "total_preds": len(preds),
        "matched": matched,
        "unmatched": len(preds) - matched,
        "correct": correct,
        "accuracy": (correct / matched) if matched > 0 else 0.0,
        "pred_regex_hits": pred_regex_hits,
        "pred_regex_ratio": (pred_regex_hits / len(preds)) if len(preds) > 0 else 0.0,
    }
    return res


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str,
                        default="./outputs/_datasets/HuggingFaceH4_MATH-500_test",
                        help="参考数据集的本地 save_to_disk 目录或 HF 名称")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--pred_paths", type=str, nargs="+", required=True,
                        help="一个或多个预测 JSONL 路径")
    parser.add_argument("--pred_input_key", type=str, default="input")
    parser.add_argument("--pred_output_key", type=str, default="output")
    parser.add_argument("--match_regex", action="store_true", default=False,
                        help="使用预测 input 作为正则匹配参考 problem 进行对齐")

    args = parser.parse_args()

    ref_map, ref_total = load_reference_dataset(args.dataset_path, args.dataset_split)
    print(f"Loaded reference: total={ref_total}, unique_problems={len(ref_map)}")

    results = []
    for path in args.pred_paths:
        if not os.path.exists(path):
            print(f"[Warn] prediction file not found: {path}")
            continue
        r = evaluate_one(
            path,
            ref_map,
            pred_input_key=args.pred_input_key,
            pred_output_key=args.pred_output_key,
            match_regex=args.match_regex,
        )
        results.append(r)

    if not results:
        print("No valid prediction results.")
        return

    # 输出每个文件的结果
    for r in results:
        print(json.dumps(r, ensure_ascii=False))

    # 若有多个，输出对比摘要
    if len(results) > 1:
        print("\nSummary (accuracy):")
        for r in results:
            print(f"- {os.path.basename(r['file'])}: {r['accuracy']:.4f} (correct={r['correct']}, matched={r['matched']}, total_preds={r['total_preds']})")


if __name__ == "__main__":
    main()


