import argparse
import json
import torch
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("-t","--task", type=str, default="task274_overruling_legal_classification.json")
parser.add_argument("-st","--shot",type=int, default=5)
parser.add_argument("-cs","--case_size",type=int)
args = parser.parse_args()

case_size = args.case_size
assert case_size

task = args.task.split("_")[0]
matrix_folder_path = f"/home/wutianhao/few_vs_zero/data/matrix/{task}/MI/{case_size}Cases"

fr_matrix_path = matrix_folder_path + f"/{args.shot}shot_fr.json"

with open(fr_matrix_path,"r") as f:
    data = json.load(f)

matrix = torch.tensor(data)

num_pos,layers,number = matrix.shape

total_matrix = torch.sum(matrix, dim=0).view(-1).tolist()

# calculate total frequency
fr_total = np.array(total_matrix) / (case_size * num_pos)

# To Avoid Divison by Zero
eps = 1e-10  # Very small number

mutual_info_sum = [0] * layers * number

for p in range(num_pos):
    flattened_matrix = matrix[p].view(-1).tolist()
    min_value = np.min(flattened_matrix)
    max_value = np.max(flattened_matrix)
    print(f"position: {p+1}, min: {min_value}, max: {max_value}")
    fr_pos = np.array(flattened_matrix) / case_size
    for idx, fr_n_pos in enumerate(fr_pos):
        # the total frequency of nth neuron
        fr_n = fr_total[idx]
        mutual_info_sum[idx] += fr_n_pos * np.log((fr_n_pos + eps) / (fr_n + eps)) + \
                        (1 - fr_n_pos) * np.log((1 - fr_n_pos + eps) / (1 - fr_n + eps))
mutual_info = np.array(mutual_info_sum) / num_pos

print(mutual_info)

PERCENT = 0.05
num_elements = len(mutual_info)
top_k = int(num_elements * PERCENT)

# 使用 torch.topk 找到最大的 5% 的值及其索引
top_values, top_indices = torch.topk(torch.tensor(mutual_info), top_k)

# 将展平后的索引转换回原始矩阵的索引
rows = top_indices // number
cols = top_indices % number

# 将行列索引组合起来
top_indices_2d = torch.stack([
    top_indices // number,  # 行索引
    top_indices % number   # 列索引
], dim=1)


# 打印结果
print(f"最大的{PERCENT*100}%的值：", top_values)
print("这些值在原始矩阵中的索引：", top_indices_2d)

output = [[[] for i in range(layers)]]
for i in top_indices_2d:
    l,c = i
    output[0][l].append(c.item())

save_output = [[]]
for j in output[0]:
    save_output[0].append(torch.tensor(j).type(torch.int64))



# 保存结果
folder_path = f"/home/wutianhao/few_vs_zero/data/activation_mask/{task}"
if not os.path.exists(folder_path):
    # 如果不存在，则创建文件夹
    os.makedirs(folder_path)
    print(f"文件夹 {folder_path} 已创建。")
else:
    print(f"文件夹 {folder_path} 已经存在。")
torch.save(save_output,folder_path+f"/activation_MI_{task}_{args.shot}shot_pth")



