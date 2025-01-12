from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import baukit.nethook
import numpy as np
from utils import process_and_save_tokens
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-t","--task", type=str, default="task274_overruling_legal_classification.json")
args = parser.parse_args()


model_dir = "/llm/wutianhao/model"
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16)
model = torch.nn.DataParallel(model)
model = model.to('cuda') 

# Load the tokenizer for your LLaMA model
tokenizer = AutoTokenizer.from_pretrained(model_dir)
# Check the ID of the EOS token
eos_token_id = tokenizer.eos_token_id
print(f"EOS token ID: {eos_token_id}")

def get_activations(model, input): 

    model.eval()
    MLP_act = [f"module.model.layers.{i}.mlp.act_fn" for i in range(32)]
    
    with torch.no_grad():
        with baukit.nethook.TraceDict(model, MLP_act) as ret:
            _ = model(input)
        MLP_act_value = [ret[act_value].output for act_value in MLP_act]
        return MLP_act_value
    
def act_llama(input):
    mlp_act = get_activations(model, input)
    return [t.cpu().numpy() for t in mlp_act] 

# process tokens
test_token, labels = process_and_save_tokens(
    task=args.task,
    shot=0,
    tokenizer_name=model_dir,
    percentage=0.25
)

def pad_lists(list_of_lists, padding_value=0):
    # Find the maximum length of the lists
    max_length = max(len(lst) for lst in list_of_lists)
    
    # Pad each list to the max length
    padded_lists = [lst + [padding_value] * (max_length - len(lst)) for lst in list_of_lists]
    
    return padded_lists

test_set = pad_lists(test_token, 2)
inputs = torch.tensor(test_set).to('cuda')

final = []
for data in tqdm(inputs, desc="Processing Prompts"):
    res = act_llama(data.unsqueeze(0))
    print(np.array(res).shape)
    final.append(res)

final = np.squeeze(final)
final = np.where(final > 0, 1, 0) # Save here for later use

for prompt_act, input_ids in zip(final, test_token):
    print(len(input_ids))
    sum_result = np.sum(prompt_act, axis=(0, 2))
    sum_result = sum_result[:len(input_ids)]
    print(len(sum_result))
    print(sum_result, '\n')
