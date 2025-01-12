from transformers import LlamaForCausalLM
import torch
import os
import pickle
from tqdm import tqdm 
import os
import pickle
import argparse
import  torch.nn as nn
import json

## 解析参数
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="/home/wth/model")
parser.add_argument("-ct","--category", type=str, default="others") # 4 categories
parser.add_argument("-k","--shot",type=int, default=5)
parser.add_argument("-md", "--mod", type=str, default="GV_trace")
parser.add_argument("--template", "-tp", type=int, default=4)
args = parser.parse_args()

assert args.mod == "GV_trace"

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
TOPK_POS = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## 加载模型并设置为训练模式
model = LlamaForCausalLM.from_pretrained(args.model, device_map='auto', torch_dtype=torch.bfloat16)
model.train()

## 损失函数
criterion = nn.CrossEntropyLoss(reduction="none")
out_data = [[0]*11008]*32

token_path = f"/home/wth/few_vs_zero/data/mmlu/token{args.template}/{args.category}/train_{args.shot}.pkl"
if os.path.exists(token_path):
    with open(token_path, 'rb') as f:
        data = pickle.load(f)
        train_token, indexs = data["inputs"], data["indices"]

## 处理训练token并计算梯度
progress_bar = tqdm(total=len(train_token), leave=True, desc='Getting data')
for input_ids,index in zip(train_token, indexs):
    progress_bar.update(1)
    
    if len(input_ids) > 4000:
        print("Too large")
        continue

    if TOPK_POS:
        best_index = [index[i] for i in TOPK_POS]
    else:
        best_index = index

    input_index = [i-1 for i in best_index]

    label_token = [input_ids[i] for i in best_index]

    input_ids = torch.tensor(input_ids,dtype=torch.int64).unsqueeze(0).to(device)
    label_token = torch.tensor(label_token,dtype=torch.int64).to(device)

    output = model(input_ids)

    # 计算损失
    loss = criterion(output.logits[0, input_index, :] ,label_token)

    model.zero_grad()

    loss.sum().backward()

    for name, param in model.named_parameters():
        if param.grad is not None and "up_proj" in name:
            layer = int(name.split(".")[2])
            grad = torch.sum(param.grad,dim=1).cpu().tolist()
            out_data[layer] =  [abs(a) + b for a, b in zip(grad, out_data[layer])]

# 保存结果

folder_path = f"/home/wth/few_vs_zero/data/mmlu/matrix/{args.category}/{args.mod}"
if not os.path.exists(folder_path): 
    # 如果不存在，则创建文件夹
    os.makedirs(folder_path)
    print(f"文件夹 {folder_path} 已创建。")
else:
    print(f"文件夹 {folder_path} 已经存在。")
with open(f"/home/wth/few_vs_zero/data/mmlu/matrix/{args.category}/{args.mod}/tp{args.template}_{args.shot}shot.json","w") as f:
    json.dump(out_data,f)