import argparse
import json
import os
from utils import evaluate_results1
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from globals_vars import ct, today, MAX_TOKEN_FOR_SNI

N_Q = 10

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="/home/wth/model")
parser.add_argument("-t", "--task", type=str, default="task274_overruling_legal_classification.json")
parser.add_argument("-d", "--device", type=str, default="7")
args = parser.parse_args()

def construct_unsupervised_ICL_prompts(train_data, test_data, instruction, n_questions, n_demos):
    prompts = []
    labels = []

    for i in test_data:
        prompt = ["".join(instruction) + ". " + "You will be provided Problems similar to the ones below:"]

        # first 
        for j in train_data[:n_questions]:
            prompt += ["Problem: " + "".join(j['input'])]
        
        prompt += ["Now, I am going to give you a series of demonstrations of Problems and Answers. When you respond, respond only with the Answer of the final Problem."]
        
        # last n in train as demo
        for k in train_data[-n_demos:]:
            
            prompt +=  ["Problem: " + "".join(k['input'])]
            prompt +=  ["Answer: " + "".join(k['output'])]

        prompt += ["Problem: " + "".join(i['input'])]


        prompts.append("\n".join(prompt))
        labels.append("".join(i['output']))

    return prompts, labels

def construct_ICL_prompts(train_data, test_data, instruction, n_demos):
    prompts = []
    labels = []
    n_dev = 10 + n_demos

    for i in test_data:
        
        prompt = ["".join(instruction)  + ". " + "Now, I am going to give you a series of demonstrations of Problems and Answers. When you respond, respond only with the Answer of the final Problem."]
        
        for k in train_data[-n_dev:]:
            
            prompt +=  ["Problem: " + "".join(k['input'])]
            prompt +=  ["Answer: " + "".join(k['output'])]

        prompt += ["Problem: " + "".join(i['input'])]


        prompts.append("\n".join(prompt))
        labels.append("".join(i['output']))

    return prompts, labels



# Set system variables
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
os.environ["TOKENIZERS_PARALLELISM"] = "false"

task_file = args.task
task = task_file.split("_")[0]

DATA_PATH = "/home/wth/few_vs_zero/datasets/SNI"

tokenizer = AutoTokenizer.from_pretrained("/home/wth/model")
tokenizer.chat_template = ct

# 加载模型
model = LLM(model=args.model, enforce_eager=True)

# Load data and split into training and test sets
data_file = os.path.join(DATA_PATH, task_file)
with open(data_file, "r") as f:
    data = json.load(f)
    instruction = data["Definition"]
    instances = data["Instances"][:6500]
    total_length = len(instances)
    sub_len = int(total_length * 0.5)
    train, test = instances[:sub_len], instances[sub_len:]

prompts, labels = construct_unsupervised_ICL_prompts(train_data=train, test_data=test, instruction=instruction, n_questions=N_Q, n_demos=5)

test_token = []
for p in prompts:
    test_token.append(tokenizer.encode(p))

# 生成结果
outputs = model.generate(prompt_token_ids=test_token, sampling_params=SamplingParams(max_tokens=MAX_TOKEN_FOR_SNI,temperature=0,top_p=1,stop="Problem"))

output_folder = f"/home/wth/few_vs_zero/results/sni/{task}/"
os.makedirs(output_folder, exist_ok=True)
output_file = f"{output_folder}/{task}_unsupervised_ICL.jsonl"

# evalaute and save results
num, correct, accuracy= evaluate_results1(output_file=output_file, tokenizer_name=args.model, labels=labels, outputs=outputs)

# Record summary
with open(f"/home/wth/few_vs_zero/ICL_result_{today}.jsonl", "a") as f:
    f.write(json.dumps(
        {   
            "setting" : "unsupervised ICL",
            "n_questions" : N_Q,
            "task": args.task, 
            "num": num, 
            "correct": correct, 
            "acc": accuracy
        }
        , ensure_ascii=False) + "\n")


########

prompts, labels = construct_ICL_prompts(train_data=train, test_data=test, instruction=instruction, n_demos=5)

test_token = []
for p in prompts:
    test_token.append(tokenizer.encode(p))

# 生成结果
outputs = model.generate(prompt_token_ids=test_token, sampling_params=SamplingParams(max_tokens=MAX_TOKEN_FOR_SNI,temperature=0,top_p=1,stop="Problem"))

output_folder = f"/home/wth/few_vs_zero/results/sni/{task}/"
os.makedirs(output_folder, exist_ok=True)
output_file = f"{output_folder}/{task}_ICL.jsonl"

# evalaute and save results
num, correct, accuracy= evaluate_results1(output_file=output_file, tokenizer_name=args.model, labels=labels, outputs=outputs)

# Record summary
with open(f"/home/wth/few_vs_zero/ICL_result_{today}.jsonl", "a") as f:
    f.write(json.dumps(
        {
            "setting" : "baseline ICL",
         
            "task": args.task, 
            "num": num, 
            "correct": correct, 
            "acc": accuracy
        }
        , ensure_ascii=False) + "\n")
