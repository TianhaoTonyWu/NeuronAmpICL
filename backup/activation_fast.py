import torch
import baukit.nethook
import numpy as np
from utils import process_and_save_tokens
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import vllm

BATCH_SIZE = 25

model_dir = "/llm/wutianhao/model"

model = vllm.LLM(model=model_dir,  enforce_eager=True)
print(dir(model))
for n, m in model.llm_engine.model_executor.driver_worker.model_runner.model.named_modules():
    print(n, m)


def get_activations(model : vllm.LLM, input): 
    llama = model.llm_engine.model_executor.driver_worker.model_runner.model
    input = torch.tensor(input, dtype=torch.int64).unsqueeze(0).to('cuda')
    MLP_act = [f"model.layers.{i}.mlp.act_fn" for i in range(32)]
    llama.eval()
    with torch.no_grad():
        with baukit.nethook.TraceDict(llama, MLP_act) as ret:
            _ = llama(input, )
        MLP_act_value = [ret[act_value].output for act_value in MLP_act]
        return MLP_act_value
    
def act_llama(input):
    mlp_act = get_activations(model, input)
    mlp_act = [t.tolist() for t in mlp_act]
    print(np.array(mlp_act).shape)
    return mlp_act

# process tokens
test_set, labels = process_and_save_tokens(
    task="task274_overruling_legal_classification.json",
    shot=0,
    tokenizer_name=model_dir,
)

def pad_lists(list_of_lists, padding_value=0):
    # Find the maximum length of the lists
    max_length = max(len(lst) for lst in list_of_lists)
    
    # Pad each list to the max length
    padded_lists = [lst + [padding_value] * (max_length - len(lst)) for lst in list_of_lists]
    
    return padded_lists

test_set = pad_lists(test_set, 0)
    
class CustomDataset(Dataset):

    def __init__(self, test_set):
        self.len = len(test_set)
        self.data = test_set

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.int64)

    def __len__(self):
        return self.len

#loader = DataLoader(dataset=CustomDataset(test_set), batch_size=BATCH_SIZE, num_workers=4)

final = []

"""
for data in tqdm(loader, desc="Processing Batches"):
    input = data.to('cuda')
    res = act_llama(input)
    final.append(res)
"""

for data in test_set[:1]:
    #input = data.to('cuda')
    input = data
    res = act_llama(input)
    final.append(res)

print(np.array(final).shape)


