import argparse
import json
import torch
import os
from utils import evaluate_gsm
from vllm import LLM, SamplingParams
import pickle
from few_vs_zero.src.globals_vars import today, MAX_TOKEN_FOR_GSM

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="/home/wth/model")
parser.add_argument("-k", "--shot", type=int, default=5)
parser.add_argument("-d", "--device", type=str, default="0,1,2,3")
parser.add_argument("-dt","--dataset",type=str, default="gsm")
args = parser.parse_args()

# Set system variables
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 加载模型
model = LLM(model=args.model, enforce_eager=True)

nshot = args.shot

token_path = f"/home/wth/few_vs_zero/data/{args.dataset}/token0/test_{nshot}.pkl"
if os.path.exists(token_path):
    with open(token_path, 'rb') as f:
        data = pickle.load(f)
        test_token, labels = data["inputs"], data["labels"]


# 生成结果
outputs = model.generate(prompt_token_ids=test_token, sampling_params=SamplingParams(max_tokens=MAX_TOKEN_FOR_GSM,temperature=0,top_p=1, stop= ["["]))

output_folder = f"/home/wth/few_vs_zero/results/{args.dataset}/"
os.makedirs(output_folder, exist_ok=True)
output_file = f"{output_folder}/{nshot}-shot.jsonl"

# evalaute and save results
num, correct, accuracy = evaluate_gsm(output_file=output_file, tokenizer_name=args.model, labels=labels, outputs=outputs)

# Record summary
with open(f"new_result_{today}.jsonl", "a") as f:
    f.write(json.dumps(
        { 
            "dataset": args.dataset,
            "shot": nshot, 
            "num": num, 
            "correct": correct, 
            "acc": accuracy,  
        }
        , ensure_ascii=False) + "\n")
