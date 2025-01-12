import argparse
import json
import torch
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("-t","--task", type=str, default="task274_overruling_legal_classification.json")
parser.add_argument("-md", "--mod", type=str, default="MI")
parser.add_argument("-st","--shot",type=int, default=5)
args = parser.parse_args()
task = args.task.split("_")[0]

### Analyze gradient sum
grad_matrix_path = f"/home/wutianhao/few_vs_zero/data/matrix/{task}/{args.mod}/{args.shot}shot_grad.json"

with open(grad_matrix_path,"r") as f:
    data = json.load(f)

grad_matrix = torch.tensor(data)
n_pos, n_layer, n_neuron = grad_matrix.shape

assert n_pos == 42

sum = []
avg = []

# Calcualte something for each p from the grads

for p in range(n_pos):
    flattened_matrix = grad_matrix[p].view(-1).tolist()
    sum.append(np.sum(flattened_matrix))
    avg.append(np.mean(flattened_matrix))


print(f"Sum: {sum}\n")
print(f"Avg: {avg}\n")

topk = 21

top_sum_indices = np.argsort(sum)[-21:][::-1]
top_avg_indices = np.argsort(avg)[-21:][::-1]

print("Top 21 sum indices:", sorted(top_sum_indices.tolist()))
print("Top 21 avg indices:", sorted(top_avg_indices.tolist()))

