import argparse
import json
import os
from utils import process_and_save_tokens, evaluate_sni
from vllm import LLM, SamplingParams
from globals_vars import MAX_TOKEN_FOR_SNI
from filelock import FileLock
# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="/home/wth/model")
parser.add_argument("-k", "--shot", type=int, default=5)
parser.add_argument("-t", "--task", type=str, default="task274_overruling_legal_classification.json")
parser.add_argument("-d", "--device", type=str, default="6")
args = parser.parse_args()

# Set system variables
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
os.environ["TOKENIZERS_PARALLELISM"] = "false"

task = args.task.split("_")[0]

test_token, labels = process_and_save_tokens(
    task=args.task, # .json file name in the dataset
    shot=args.shot,
    tokenizer_name=args.model,
    percentage=0.5,
    isTest=True,
    temp=0
)

# 加载模型
model = LLM(model=args.model, enforce_eager=True)

# 生成结果
outputs = model.generate(prompt_token_ids=test_token, sampling_params=SamplingParams(max_tokens=MAX_TOKEN_FOR_SNI,temperature=0,top_p=1,stop="[INST]"))

output_folder = f"/home/wth/few_vs_zero/results/sni/{task}/"
os.makedirs(output_folder, exist_ok=True)
output_file = f"{output_folder}/{task}-{args.shot}-shot.jsonl"

# evalaute and save results
num, correct, accuracy= evaluate_sni(output_file=output_file, tokenizer_name=args.model, labels=labels, outputs=outputs)

# Record summary
lock_file_name = "/home/wth/few_vs_zero/mass_search.lock"
with FileLock(lock_file_name):
    with open(f"/home/wth/few_vs_zero/mass_search.jsonl", "a") as f:
        f.write(json.dumps(
            {
                "task": args.task, 
                "shot": args.shot, 
                "num": num, 
                "correct": correct, 
                "acc": accuracy
            }
            , ensure_ascii=False) + "\n")
