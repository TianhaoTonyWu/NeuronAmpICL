import argparse
import os
import json
from types import MethodType
import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams
import pickle
from utils import evaluate_results
from few_vs_zero.src.globals_vars import MAX_TOKEN_FOR_SNI
from filelock import FileLock
from tqdm import tqdm
from jinja2 import Template
from transformers import AutoTokenizer
from utils import data_construct
import few_vs_zero.src.globals_vars as globals_vars

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="/home/wth/model")
parser.add_argument("-msj", "--mask_subject", type=str, default="")
parser.add_argument("-tsj", "--task_subject", type=str, default="")
parser.add_argument("-md", "--mod", type=str, default="GV_trace")
parser.add_argument("-d", "--device", type=str, default="0")
args = parser.parse_args()

# Set system variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

task_subject = args.task_subject.split("_")[0]
mask_subject = args.mask_subject.split("_")[0]

# Load model on a Single GPU
model = LLM(model=args.model, enforce_eager=True)


max_acc = -1.0
best_p = None
best_x = None


token_path = f"/home/wth/few_vs_zero/data/sni/token/{task_subject}/val_0.pkl"
if os.path.exists(token_path):
    with open(token_path, 'rb') as f:
        data = pickle.load(f)
        val_token, labels = data["inputs"], data["labels"]

else:
    DATA_PATH = "/home/wth/few_vs_zero/datasets/SNI"
    base_path='/home/wth/few_vs_zero/data/sni/token/'
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.chat_template = globals_vars.ct

    # Load data and split into training and val sets
    data_file = os.path.join(DATA_PATH, args.task_subject)
    with open(data_file, "r") as f:
        data = json.load(f)
        instruction = data["Definition"]
        instances = data["Instances"][:6500]
        total_length = len(instances)
        sub_len = total_length // 4
        val = instances[:sub_len]
        test_message = data_construct(val, instruction, shot=0)
        
    # Process and save val tokens and labels
    test_file = os.path.join(base_path, task_subject, f'val_{0}.pkl')
    if os.path.exists(test_file):
        with open(test_file, 'rb') as f:
            data = pickle.load(f)
            val_token, labels = data["inputs"], data["labels"]
    else:
        results, val_token, labels = [], [], []
        progress_bar = tqdm(total=len(test_message), desc='val Processing data')
        for message in test_message:
            progress_bar.update(1)
            prompt, output = message[:-1], message[-1]
            template_str = tokenizer.chat_template
            template = Template(template_str)
            result = template.render(messages=prompt, bos_token="", eos_token="")
            results.append(result)
            val_token.append(tokenizer.encode(result))
            labels.append(output["content"])
        progress_bar.close()

        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        with open(test_file, 'wb') as f:
            pickle.dump({"inputs": val_token, "labels": labels}, f)

for percentage in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4 ,0.45, 0.5]:
    # Load activation mask
    mask_path = f"/home/wth/few_vs_zero/data/sni/activation_mask/{mask_subject}/activation_{args.mod}_{mask_subject}_5shot_{percentage}percent_pth"
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
        
        
        outputs = model.generate(prompt_token_ids=val_token, sampling_params=SamplingParams(max_tokens=MAX_TOKEN_FOR_SNI, temperature=0, top_p=1, stop=["[INST]"]))


        output_folder = f"/home/wth/few_vs_zero/results/sni/{task_subject}/"
        os.makedirs(output_folder, exist_ok=True)
        output_file = f"{output_folder}/masked_val_{args.mod}_{task_subject}_{percentage}percent.jsonl"

        # evalaute and save results
        num, correct, accuracy = evaluate_results(output_file=output_file, tokenizer_name=args.model, labels=labels, outputs=outputs)
        

        print(round(multiplier,2), round(percentage,2), round(accuracy, 3))

        if accuracy > max_acc:
            max_acc = accuracy
            best_x = float(multiplier)
            best_p = percentage
       

print("Best Parameters:", round(best_x, 1), best_p, max_acc)

final_out_file = f"/home/wth/few_vs_zero/SNI_best_param.jsonl"
lock_file_name = f"{final_out_file}.lock"
with FileLock(lock_file_name):
    with open(final_out_file, "a", encoding="utf8") as f:
        f.write(json.dumps({
                            "Mode": args.mod,
                            "Subject": task_subject,
                            "Multiplier": best_x,
                            "Percentage": best_p,
                            "acc": max_acc},
                            ensure_ascii=False) + "\n")
