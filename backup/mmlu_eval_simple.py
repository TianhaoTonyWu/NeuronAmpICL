import argparse
import json
import os
from utils import evaluate_mmlu
from vllm import LLM, SamplingParams
import pickle
from few_vs_zero.src.globals_vars import today, MAX_TOKEN_FOR_MMLU

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="/home/wth/model")
parser.add_argument("-k", "--shot", type=int, default=5) 
parser.add_argument("-sj", "--subject", type=str, default="professional_psychology") # 57 subcategories
args = parser.parse_args()

# Set system variables
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

# 加载模型
# Do not use tensor parallelism for consistency
model = LLM(model=args.model, enforce_eager=True)

subject =  args.subject
nshot = args.shot

token_path = f"/home/wth/few_vs_zero/data/mmlu/token0/{subject}/test_{nshot}.pkl"
if os.path.exists(token_path):
    with open(token_path, 'rb') as f:
        data = pickle.load(f)
        test_token, labels = data["inputs"], data["labels"]


# 生成结果
outputs = model.generate(prompt_token_ids=test_token, sampling_params=SamplingParams(max_tokens=MAX_TOKEN_FOR_MMLU,temperature=0,top_p=1, stop= ["[INST]"]))

output_folder = f"/home/wth/few_vs_zero/results/mmlu/{subject}/"
os.makedirs(output_folder, exist_ok=True)
output_file = f"{output_folder}/{subject}-{nshot}-shot.jsonl"

# evalaute and save results
num, correct, accuracy = evaluate_mmlu(output_file=output_file, tokenizer_name=args.model, labels=labels, outputs=outputs)

# Record summary
with open(f"new_result_{today}.jsonl", "a") as f:
    f.write(json.dumps(
        {
            "subject": subject, 
            "shot": nshot, 
            "num": num, 
            "correct": correct, 
            "acc": accuracy,  
        }
        , ensure_ascii=False) + "\n")
