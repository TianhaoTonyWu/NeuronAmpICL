import torch
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-md", "--mod", type=str, default="GV_intersect")
parser.add_argument("-st", "--shot", type=int, default=5)
parser.add_argument("-pt", "--percent", type=float, default=0.1)
args = parser.parse_args()

percent = args.percent

paths = [
    "/home/wth/few_vs_zero/data/gsm/matrix/GV_trace/tp1_5shot.json",
    "/home/wth/few_vs_zero/data/gsm/matrix/GV_trace/tp2_5shot.json",
    "/home/wth/few_vs_zero/data/gsm/matrix/GV_trace/tp3_5shot.json",
]

# Start with a zero matrix of the same shape as the first matrix
matrix = torch.zeros(32,11008)

# Loop through each matrix and add it to sum_matrix

for path in paths:
    with open(path,"r") as f:
        data = json.load(f)
    mat = torch.tensor(data)
    matrix += mat
        

# 创建一个示例矩阵
layers,number = matrix.shape
flattened_matrix = matrix.view(-1)

# 计算需要的元素数量（最大的5%）
num_elements = flattened_matrix.numel()
top_k = int(num_elements *percent)

# 使用 torch.topk 找到最大的 5% 的值及其索引
top_values, top_indices = torch.topk(flattened_matrix, top_k)

# 将展平后的索引转换回原始矩阵的索引
rows = top_indices // matrix.size(1)
cols = top_indices % matrix.size(1)

# 将行列索引组合起来
top_indices_2d = torch.stack([
    top_indices // number,  # 行索引
    top_indices % number   # 列索引
], dim=1)

# 打印结果
print(f"最大的{round(args.percent * 100.0, 2)}%的值：", top_values)
print("这些值在原始矩阵中的索引：", top_indices_2d)

output = [[[] for i in range(layers)]]
for i in top_indices_2d:
    l,c = i
    output[0][l].append(c.item())

save_output = [[]]
for j in output[0]:
    save_output[0].append(torch.tensor(j).type(torch.int64))

# 保存结果
folder_path = f"/home/wth/few_vs_zero/data/gsm/activation_mask"
if not os.path.exists(folder_path):
    # 如果不存在，则创建文件夹
    os.makedirs(folder_path)
    print(f"文件夹 {folder_path} 已创建。")
else:
    print(f"文件夹 {folder_path} 已经存在。")
torch.save(save_output,folder_path+f"/activation_{args.mod}_{args.shot}shot_{percent}percent_pth")

# # Function to find top N indices in a single matrix
# def find_top_indices(matrix, percent):
#     flattened_matrix = matrix.view(-1)
#     num_elements = flattened_matrix.numel()
#     top_k = int(num_elements * percent)
#     _, top_indices = torch.topk(flattened_matrix, top_k)
#     rows = top_indices // matrix.size(1)
#     cols = top_indices % matrix.size(1)
#     top_indices_2d = torch.stack([rows, cols], dim=1)
#     return set(tuple(idx.tolist()) for idx in top_indices_2d)

# # Find top neurons for each matrix
# top_indices_sets = []
# for path in paths:
#     with open(path, "r") as f:
#         data = json.load(f)
#     mat = torch.tensor(data)
#     top_indices_sets.append(find_top_indices(mat, percent))

# # Find the intersection of the three sets
# intersection_indices = top_indices_sets[0].intersection(*top_indices_sets[1:])

# # Convert the intersection indices to a structured output
# layers, number = torch.tensor(json.load(open(paths[0], "r"))).shape
# output = [[[] for _ in range(layers)]]
# for l, c in intersection_indices:
#     output[0][l].append(c)

# save_output = [[]]
# for layer_neurons in output[0]:
#     save_output[0].append(torch.tensor(layer_neurons).type(torch.int64))

# # Save the result
# folder_path = f"/home/wth/few_vs_zero/data/gsm/activation_mask"
# if not os.path.exists(folder_path):
#     os.makedirs(folder_path)
#     print(f"Folder {folder_path} created.")
# else:
#     print(f"Folder {folder_path} already exists.")

# torch.save(
#     save_output,
#     folder_path + f"/activation_{args.mod}_{args.shot}shot_{percent}percent_pth",
# )

# print(f"Top {percent * 100:.2f}% intersection saved to {folder_path}.")
