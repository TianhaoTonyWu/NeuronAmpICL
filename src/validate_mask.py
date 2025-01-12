import argparse
import os
import json
from types import MethodType
import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams
import pickle
from utils import evaluate_sni, evaluate_gsm, evaluate_mmlu
from globals_vars import MAX_TOKEN_FOR_GSM, MAX_TOKEN_FOR_MMLU, MAX_TOKEN_FOR_SNI
from filelock import FileLock


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="/root/model")
parser.add_argument("-in", "--input_file", type=str, default="/root/few_vs_zero/data/gsm/token0/val_0.pkl")
parser.add_argument("-mp", "--mask_dir", type=str, default="/root/few_vs_zero/data/gsm/activation_mask/GV")
parser.add_argument("-d", "--device", type=str, default="0,1,2,3")
args = parser.parse_args()

# Set system variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

token_path = args.input_file
mask_dir = args.mask_dir
assert os.path.isdir(mask_dir) 

dataset = token_path.split("/")[5]
assert dataset in ["gsm", "mmlu", "sni"]

task_name = token_path.split("/")[7] if dataset in ["sni", "mmlu"] else "ALL"

mod =  mask_dir.split("/")[-1]
assert mod in ["GV", "GV_last", "GV_final", "LAPE"]
    


# Load model on a Single GPU
model = LLM(model=args.model, enforce_eager=True)

max_acc = -1.0
best_p = None
best_x = None



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
        val_token, labels = data["inputs"], data["labels"]

else:
    raise FileNotFoundError("No such validation token file!")

for percentage in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4 ,0.45, 0.5]:
# For GSM, run smaller 
# for percentage in [0.01, 0.02, 0.03, 0.04]:
    # Load activation mask

    mask_path = mask_dir + f"/{percentage}p.pth"
    if os.path.exists(mask_path):
        activation_mask_path = mask_path
    else:
        raise FileNotFoundError(f"The file does not exist: {mask_path}")

    for multiplier in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]:

        # Load or generate activation masks
        activation_masks = torch.load(activation_mask_path, weights_only=True)
        # Custom forward function with mask application
        def custom_llama_forward(mask):
            def llama_forward(self, x):
                gate_up, _ = self.gate_up_proj(x)
                i = gate_up.size(-1)
                activation = F.silu(gate_up[:, :i // 2])
                mask_tensor = torch.ones(activation.size(-1), device=activation.device)
                multiplier_tensor = torch.tensor(multiplier, device=activation.device)
                # fix to illegal memory access: add everything to the same devcie
                mask_tensor[mask.to(activation.device)] = multiplier_tensor           
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
                    # obj.forward = MethodType(custom_llama_forward(layer_mask.to('cuda').type(torch.int64)), obj)
                    obj.forward = MethodType(custom_llama_forward(layer_mask.type(torch.int64)), obj)
        
        
        outputs = model.generate(prompt_token_ids=val_token, sampling_params=SamplingParams(max_tokens=max_toks, temperature=0, top_p=1, stop=["[INST]"]))


        output_folder = f"/root/few_vs_zero/results/{dataset}/{task_name}/"
        os.makedirs(output_folder, exist_ok=True)
        output_file = f"{output_folder}/masked_val_{mod}_{percentage}percent.jsonl"

        # evalaute and save results
        num, correct, accuracy = evaluate_func(output_file=output_file, tokenizer_name=args.model, labels=labels, outputs=outputs)
        

        print(round(multiplier,2), round(percentage,2), round(accuracy, 3))

        if accuracy > max_acc:
            max_acc = accuracy
            best_x = float(multiplier)
            best_p = percentage
       

print("Best Parameters:", round(best_x, 1), best_p, max_acc)

final_out_file = f"/root/few_vs_zero/best_params.jsonl"
lock_file_name = f"{final_out_file}.lock"
with FileLock(lock_file_name):
    with open(final_out_file, "a", encoding="utf8") as f:
        f.write(json.dumps({
                            "Mode" : mod,
                            "Dataset" : dataset,
                            "Task" : task_name,
                            "Multiplier": best_x,
                            "Percentage": best_p,
                            "acc": max_acc},
                            ensure_ascii=False) + "\n")
