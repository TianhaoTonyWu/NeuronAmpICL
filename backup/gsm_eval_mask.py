import argparse
import os
import json
from types import MethodType
import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams
import pickle
from utils import evaluate_gsm
from few_vs_zero.src.globals_vars import today, MAX_TOKEN_FOR_GSM


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="/home/wth/model")
parser.add_argument("-ms", "--mask_shot", type=int, default=5)
parser.add_argument("-ts", "--task_shot", type=int, default=0)
parser.add_argument("-x", "--multiplier", type=float, default=1.0)
parser.add_argument("-pt", "--percent", type=float, default=0.1)
parser.add_argument("-md", "--mod", type=str, default="GV_intersect")
parser.add_argument("-dt","--dataset",type=str, default="gsm")
parser.add_argument("-d", "--device", type=str, default="0,1,2,3")
args = parser.parse_args()

# Set system variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

# Load activation mask
mask_path = f"/home/wth/few_vs_zero/data/{args.dataset}/activation_mask/activation_{args.mod}_{args.mask_shot}shot_{args.percent}percent_pth"
if os.path.exists(mask_path):
    activation_mask_path = mask_path
else:
    raise FileNotFoundError(f"The file does not exist: {mask_path}")


# Load model on a Single GPU
model = LLM(model=args.model, enforce_eager=True)

# Load or generate activation masks
activation_masks = torch.load(activation_mask_path, weights_only=True)

# Custom forward function with mask application
def custom_llama_forward(mask):
    def llama_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        i = gate_up.size(-1)
        activation = F.silu(gate_up[:, :i // 2])
        mask_tensor = torch.ones(activation.size(-1), device=activation.device)
        mask_tensor[mask] = args.multiplier
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




token_path = f"/home/wth/few_vs_zero/data/{args.dataset}/token0/test_{args.task_shot}.pkl"
if os.path.exists(token_path):
    with open(token_path, 'rb') as f:
        data = pickle.load(f)
        test_token, labels = data["inputs"], data["labels"]

# Generate outputs
outputs = model.generate(prompt_token_ids=test_token, sampling_params=SamplingParams(max_tokens=MAX_TOKEN_FOR_GSM,temperature=0,top_p=1,stop="["))


output_folder = f"/home/wth/few_vs_zero/results/{args.dataset}/"
os.makedirs(output_folder, exist_ok=True)
output_file = f"{output_folder}/masked_{args.mod}_{args.task_shot}_{args.multiplier}_{args.percent}percent.jsonl"

num, correct, accuracy = evaluate_gsm(output_file=output_file, tokenizer_name=args.model, labels=labels, outputs=outputs)

# record summary
with open(f"new_result_{today}.jsonl", "a") as f:
    f.write(json.dumps(
        {
            "mode": args.mod,
            "dataset": args.dataset,
            "multiplier": args.multiplier, 
            "percentage" : args.percent,
            "num": num, "correct": correct, 
            "acc": accuracy,  
        }, ensure_ascii=False) + "\n")
