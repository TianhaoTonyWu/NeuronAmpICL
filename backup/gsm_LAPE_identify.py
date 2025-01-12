import os
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="/home/wth/model")
parser.add_argument("-pt","--percent",type=float, default=0.1)
parser.add_argument("-dt","--dataset",type=str, default="gsm")
parser.add_argument("-mx", "--matrix", type=str, default="over_zero") # sum1, sum2, sum3, sum4, over_zero
args = parser.parse_args()

percent = args.percent
mod = "LAPE"

# Define the path where the .pth file was saved
save_path = f"/home/wth/few_vs_zero/data/{args.dataset}/matrix/"
save_file = os.path.join(save_path, f"activation_{mod}.pth")

# Check if the file exists before attempting to load it
if os.path.exists(save_file):
    # Load the .pth file
    output = torch.load(save_file)
else:
    raise FileNotFoundError("No LAPE file")

# Access the matrices stored in the output dictionary
n = output['n']
print("\nNumber of tokens: ", n)

matrix = output[args.matrix]
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
print(f"最大的{round(percent * 100.0, 2)}%的值：", top_values)
print("这些值在原始矩阵中的索引：", top_indices_2d)

output = [[[] for i in range(layers)]]
for i in top_indices_2d:
    l,c = i
    output[0][l].append(c.item())

save_output = [[]]
for j in output[0]:
    save_output[0].append(torch.tensor(j).type(torch.int64))

# 保存结果
folder_path = f"/home/wth/few_vs_zero/data/{args.dataset}/activation_mask/"
if not os.path.exists(folder_path):
    # 如果不存在，则创建文件夹
    os.makedirs(folder_path)
    print(f"文件夹 {folder_path} 已创建。")
else:
    print(f"文件夹 {folder_path} 已经存在。")
torch.save(save_output,folder_path+f"/activation_{mod}_5shot_{percent}percent_pth")
