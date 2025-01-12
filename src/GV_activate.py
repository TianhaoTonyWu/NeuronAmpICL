from transformers import LlamaForCausalLM
import torch
import os
import pickle
from tqdm import tqdm
import argparse
import torch.nn as nn
import json


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="/home/wth/model")
parser.add_argument("-in", "--input_file", type=str)
parser.add_argument("-out", "--output_file", type=str)
parser.add_argument("-dv", "--device", type=str, default="4,5,6,7")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


######################## For different datasets ###############

# Need to process the data first 

token_path = args.input_file
matrix_path = args.output_file


# Load or process tokenized data
if os.path.exists(token_path):
    with open(token_path, 'rb') as f:
        data = pickle.load(f)
        train_token, indexs = data["inputs"], data['indices']

else: 
        
    raise FileNotFoundError(f"Tokens not found")

##################################################################

# Load model
model = LlamaForCausalLM.from_pretrained(args.model, device_map='auto', torch_dtype=torch.bfloat16)
model.train()

# Loss function and output storage
criterion = nn.CrossEntropyLoss(reduction="none")
out_data = [[0] * 11008] * 32

# Process training tokens and compute gradients
progress_bar = tqdm(total=len(train_token), leave=True, desc='Training')
for input_ids, index in zip(train_token, indexs):
    progress_bar.update(1)
    if len(input_ids) > 4000:
        continue
    best_index = index
    input_index = [i - 1 for i in best_index]
    label_token = [input_ids[i] for i in best_index]

    input_ids = torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0).to(device)
    label_token = torch.tensor(label_token, dtype=torch.int64).to(device)

    output = model(input_ids)
    loss = criterion(output.logits[0, input_index, :], label_token)

    model.zero_grad()
    loss.sum().backward()

    for name, param in model.named_parameters():
        if param.grad is not None and "up_proj" in name:
            layer = int(name.split(".")[2])
            grad = torch.sum(param.grad, dim=1).cpu().tolist()
            out_data[layer] = [abs(a) + b for a, b in zip(grad, out_data[layer])]

# Save results
os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
with open(matrix_path, "w") as f:
    json.dump(out_data, f)

print(f"Results saved to {matrix_path}")
