import argparse
import os
import json
from types import MethodType
import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams
from globals_vars import today, MAX_TOKEN_FOR_SNI
from utils import process_and_save_tokens, evaluate_results


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="/home/wth/model")
parser.add_argument("-ms", "--mask_shot", type=int, default=5)
parser.add_argument("-ts", "--task_shot", type=int, default=0)
parser.add_argument("-t", "--task", type=str, default="task242_tweetqa_classification.json")
parser.add_argument("-x", "--multiplier", type=float, default=1.1)
parser.add_argument("-pt", "--percent", type=float, default=0.1)
parser.add_argument("-md", "--mod", type=str, default="GV_trace")
parser.add_argument("-d", "--device", type=str, default="0")
args = parser.parse_args()

# Set system variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
# Derive task name and model base name
task = args.task.split("_")[0]

# Load activation mask
mask_path = f"/home/wth/few_vs_zero/data/sni/activation_mask/{task}/activation_{args.mod}_{args.mask_shot}shot_{args.percent}percent_pth"

if os.path.exists(mask_path):
    activation_mask_path = mask_path
else:
    raise FileNotFoundError(f"The file does not exist: {mask_path}")


# process tokens
test_token, labels = process_and_save_tokens(
    task=args.task,
    shot=args.task_shot,
    tokenizer_name=args.model,
    percentage=0.5,
    isTest=True,
    temp=None

)
###############

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

# Generate outputs
test_token = test_token



outputs = model.generate(prompt_token_ids=test_token, sampling_params=SamplingParams(max_tokens=MAX_TOKEN_FOR_SNI,temperature=0,top_p=1,stop="[INST]"))


output_folder = f"/home/wth/few_vs_zero/results/sni/{task}/"
os.makedirs(output_folder, exist_ok=True)
output_file = f"{output_folder}/masked_{args.mod}_{task}_{args.task_shot}_{args.multiplier}_{args.percent}percent.jsonl"

# evalaute and save results
num, correct, accuracy= evaluate_results(output_file=output_file, tokenizer_name=args.model, labels=labels, outputs=outputs)
   
# record summary
with open(f"new_result_{today}.jsonl", "a") as f:
    f.write(json.dumps(
        {
            "mod": args.mod,
            "task": task, 
            "multiplier": args.multiplier, 
            "percentage" : args.percent,
            "num": num, "correct": correct, 
            "acc": accuracy
        }, ensure_ascii=False) + "\n")
