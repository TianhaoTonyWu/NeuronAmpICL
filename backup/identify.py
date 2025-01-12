import torch
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-in", "--input_file", type=str, default="/home/wth/few_vs_zero/data/sni/task274/matrix/GV.json")
parser.add_argument("-out", "--output_dir", type=str, default="/home/wth/few_vs_zero/data/sni/task274/activation_mask/GV/")
parser.add_argument("-pt","--percent",type=float, default=0.01)
args = parser.parse_args()


percent = args.percent
input_path = args.input_file
output_path = args.output_dir
file_name = f"{percent}p.pth"


if input_path.split(".")[-1] == "pth":

    # Load LAPE matrix
    output = torch.load(input_path, weights_only=True)
    matrix = output['over_zero']


else:

    # Load GV matrix
    with open(input_path,"r") as f:
        data = json.load(f)

    # 创建一个示例矩阵
    matrix = torch.tensor(data)







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



# 保存结
os.makedirs(os.path.dirname(output_path), exist_ok=True)
torch.save(save_output, output_path+file_name)
