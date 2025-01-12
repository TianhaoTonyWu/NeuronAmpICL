import argparse
import os
import json
from types import MethodType
import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams
from globals_vars import today, MAX_TOKEN_FOR_SNI, MAX_TOKEN_FOR_GSM, MAX_TOKEN_FOR_MMLU
from utils import evaluate_sni, evaluate_gsm, evaluate_mmlu
import pickle


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="/home/wth/model")

parser.add_argument("-x", "--multiplier", type=float, default=1.0)
parser.add_argument("-mask", "--mask_file", type=str, default="/home/wth/few_vs_zero/data/gsm/activation_mask/GV/0.1p.pth")
parser.add_argument("-in", "--token_file", type=str, default="/home/wth/few_vs_zero/data/gsm/token0/test_5.pkl")
parser.add_argument("-d", "--device", type=str, default="0")
args = parser.parse_args()

# Set system variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

multiplier = args.multiplier 
hasMask = multiplier != 1.0
token_path = args.token_file
mask_path = args.mask_file if hasMask else None

dataset = token_path.split("/")[5]
assert dataset in ["gsm", "mmlu", "sni"]
# /home/wth/few_vs_zero/data/sni/token/task242/test_0.pkl
task_name = token_path.split("/")[7] if dataset in ["sni", "mmlu"] else "ALL"
mod =  mask_path.split("/")[-2] if hasMask else "None"
assert mod in ["GV", "GV_last", "LAPE", "GV_final", "Random", "None"]

if mod != "Random":
    percent = float(mask_path.split("/")[-1].split("p")[0]) if hasMask else "None"
else:
    percent = 0.05

nshot = int(token_path.split(".")[0][-1])

# Match dataset specific parameters
if dataset == "gsm":
    max_toks = MAX_TOKEN_FOR_GSM
    evaluate_func = evaluate_gsm
elif dataset == "sni":
    max_toks = MAX_TOKEN_FOR_SNI
    evaluate_func = evaluate_sni
elif dataset == "mmlu":
    max_toks = MAX_TOKEN_FOR_MMLU
    evaluate_func = evaluate_mmlu
else:
    raise ValueError("Unseen dataset")



if os.path.exists(token_path):
    with open(token_path, 'rb') as f:
        data = pickle.load(f)
        test_token, labels = data["inputs"], data['labels']
else:
    raise FileNotFoundError("No test token found")


# Load model on a Single GPU
model = LLM(model=args.model, enforce_eager=True)

if hasMask:

    # Load activation mask

    if os.path.exists(mask_path):
        activation_mask_path = mask_path
    else:
        raise FileNotFoundError(f"The file does not exist: {mask_path}")
    # Load or generate activation masks
    activation_masks = torch.load(activation_mask_path, weights_only=True)

    # Custom forward function with mask application
    def custom_llama_forward(mask):
        def llama_forward(self, x):
            gate_up, _ = self.gate_up_proj(x)
            i = gate_up.size(-1)
            activation = F.silu(gate_up[:, :i // 2])
            mask_tensor = torch.ones(activation.size(-1), device=activation.device)
            mask_tensor[mask] = multiplier
            activation *= mask_tensor
            x = activation * gate_up[:, i // 2:]
            x, _ = self.down_proj(x)
            return x
        return llama_forward

    # Apply custom forward to model layers
    for activation_mask in activation_masks:
        if activation_mask:
            for i, layer_mask in enumerate(activation_mask):
                obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp
                obj.forward = MethodType(custom_llama_forward(layer_mask.to('cuda').type(torch.int64)), obj)

outputs = model.generate(prompt_token_ids=test_token, sampling_params=SamplingParams(max_tokens=max_toks,temperature=0,top_p=1,stop="[INST]"))


output_folder = f"/home/wth/few_vs_zero/results/{dataset}/{task_name}/"
os.makedirs(output_folder, exist_ok=True)

if hasMask:
    output_file = f"eval_{mod}_x{multiplier}_{percent}percent.jsonl"
else:
    output_file = f"eval_{nshot}shot.jsonl"


output_file = output_folder + output_file

# evalaute and save results
num, correct, accuracy= evaluate_func(output_file=output_file, tokenizer_name=args.model, labels=labels, outputs=outputs)
   
# record summary
with open(f"/home/wth/few_vs_zero/new_result_{today}.jsonl", "a") as f:
    f.write(json.dumps(
        {
            "mod": mod,
            "n_shot" : nshot,
            "dataset" : dataset,
            "task": task_name, 
            "multiplier": multiplier, 
            "percentage" : percent,
            "num": num, 
            "correct": correct, 
            "acc": accuracy
        }, ensure_ascii=False) + "\n")
