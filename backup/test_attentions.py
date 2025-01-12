from transformers import LlamaForCausalLM, AutoTokenizer
import numpy as np
from utils import find_all_sublists, clean_text
from globals_vars import sub_sequence_list, MAX_TOKEN_FOR_SNI
import os
import torch
from tqdm import tqdm 
import json
import argparse
import random
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", type=str, default="5,6,7")
parser.add_argument("-w", "--weight", type=float, default=1.0)
parser.add_argument("-s", "--seed", type=int, default=42) 
parser.add_argument("-in", "--input_path", type=str, default="/home/wth/few_vs_zero/data/sni/token/task242/test_5.pkl")
args = parser.parse_args()

if args.seed != None:
    RAND_SEED = args.seed
    random.seed(RAND_SEED)
    DO_RANDOM = True

token_path = args.input_path
#/home/wth/few_vs_zero/data/sni/token/task242/test_5.pkl
dataset = token_path.split("/")[5]
task_name = token_path.split("/")[7]

if dataset == "sni":
    max_toks = MAX_TOKEN_FOR_SNI
else:
    raise ValueError("Unseen Dataset")



# Set environment variable for GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_mask(input_ids, doRandom : bool):
    """
    Generate a custom attention mask for a given prompt's input IDs.
    
    Parameters:
    - input_ids (list[int]): The tokenized input IDs of the prompt.

    Returns:
    - torch.Tensor: The custom attention mask for the input prompt.
    """

    # Identify the sub-sequences
    sp0 = sub_sequence_list[0]  # [INST]
    sp1 = sub_sequence_list[1]  # [/INST]

    # Find indices of [INST] and [/INST] tokens
    sp0_indices = [x[0] - 1 for x in find_all_sublists(input_ids, sp0)[1:]]
    sp1_indices = [x[-1] - 1 for x in find_all_sublists(input_ids, sp1)[:-1]]

    # Get ranges between [INST] and [/INST]
    def find_integers_between_inclusive(list1, list2):
        result = []
        for a, b in zip(list1, list2):
            start, end = min(a, b), max(a, b)
            result.append(list(range(start, end + 1)))  # Include both start and end
        return np.concatenate(result)

    all_sp_indices = find_integers_between_inclusive(sp1_indices, sp0_indices)

    n_positions = len(all_sp_indices)

    mask_length = len(input_ids)

    # Create a custom attention mask
    custom_mask = torch.ones(mask_length, dtype=torch.float)  # Initialize with all ones

    if doRandom:
        random_indices = random.sample(range(0, mask_length), n_positions)
        custom_mask[random_indices] += args.weight 
    else:  
        custom_mask[all_sp_indices] += args.weight  # Increment mask values for specified ranges

    return custom_mask

# Load the model and tokenizer
model = LlamaForCausalLM.from_pretrained("/home/wth/model", attn_implementation="eager", device_map ='auto')

tokenizer = AutoTokenizer.from_pretrained("/home/wth/model")

if os.path.exists(token_path):
    with open(token_path, 'rb') as f:
        data = pickle.load(f)
        test_token, labels = data["inputs"], data['labels']
else:
    raise FileNotFoundError("No test token found")

correct = 0
N = len(test_token)

output_file = f"/home/wth/few_vs_zero/results/{dataset}/{task_name}/attn_+{args.weight}.jsonl"
os.makedirs(os.path.dirname(output_file), exist_ok=True)


for input_ids, label in tqdm(zip(test_token, labels), desc="Generating", total= N):

    prompt_length = len(input_ids)
    custom_mask = generate_mask(input_ids, DO_RANDOM)

    outputs = model.generate(input_ids=torch.tensor(input_ids).unsqueeze(0).to(device),
                             attention_mask=custom_mask.unsqueeze(0).to(device),
                             top_p = 1,
                             do_sample=False,
                             temperature = None,
                             max_new_tokens = max_toks
                             )
    

    ## Evaluate
    prompt = tokenizer.decode(outputs[0][:prompt_length], skip_special_tokens=True)
    pred = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)

    clean_label = clean_text(label)
    clean_pred = clean_text(pred)

    if clean_pred == clean_label:
        correct+=1

    with open(output_file, "a", encoding="utf8") as f:
        f.write(json.dumps({"prompt": prompt, "pred" : pred, "label" : label},ensure_ascii=False) + "\n")

acc = correct / N 

with open(f"/home/wth/few_vs_zero/attn_new_results.jsonl", "a") as f:
     f.write(json.dumps(
        {
            "setting" : "random positions" if DO_RANDOM else "label positions",
            "seed" : args.seed,
            "dataset" : dataset,
            "task": task_name, 
            "added_weigth" : args.weight,
            "attn_accuracy": acc
        }
        , ensure_ascii=False) + "\n")