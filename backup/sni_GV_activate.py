from transformers import LlamaForCausalLM, AutoTokenizer
import torch
import os
from utils import find_all_sublists, construct_prompts, build_trace_indices
from transformers import AutoTokenizer
import pickle
from tqdm import tqdm 
import os
from jinja2 import Template
import pickle
import argparse
import  torch.nn as nn
import json
import few_vs_zero.src.globals_vars as globals_vars

## 解析参数
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="/home/wth/model")
parser.add_argument("-t","--task", type=str, default="task274_overruling_legal_classification.json")
parser.add_argument("-d","--device",type=str, default="0,1,2,3")
parser.add_argument("-st","--shot",type=int, default=5)
parser.add_argument("-md", "--mod", type=str, default="GV_trace")
parser.add_argument("--template", "-tp", type=int, default=1)
args = parser.parse_args()

assert args.mod == "GV_trace"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

# TOPK_POS = None

sub_sequence_list = globals_vars.sub_sequence_list

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = "/home/wth/few_vs_zero/datasets/SNI"
# 获取任务名称
task = args.task.split("_")[0]

# 加载数据并拆分为训练集和测试集
data_path = os.path.join(DATA_PATH, args.task)
with open(data_path, "r") as f:
    data = json.load(f)
    instruction = data["Definition"]
    instance = data["Instances"]
    data_number = len(instance)
    train = instance[:data_number//2]
    train_message = construct_prompts(train, instruction, shot=args.shot, template=args.template)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.chat_template = globals_vars.ct

# 处理训练数据并保存token
train_file = f'/home/wth/few_vs_zero/data/sni/token{args.template}/{task}/train_trace_{args.shot}.pkl'
if os.path.exists(train_file):
    with open(train_file, 'rb') as f:
        data = pickle.load(f)
        train_token = data["inputs"]
        if "indexs" in list(data.keys()):
            indexs = data["indexs"]
else:
    ## 如果文件不存在则处理新的数据
    train_token = []
    indexs = []
    progress_bar = tqdm(total=len(train_message), desc='Train Processing data')
    for i in range(len(train_message)):
        progress_bar.update(1)
        message = train_message[i]
        template_str = tokenizer.chat_template
        template = Template(template_str)
        bos_token = ""
        eos_token = ""
        result = template.render(messages=message, bos_token=bos_token, eos_token=eos_token).replace("<spe>"," ")

        ## 如果是trace模式，则提取子序列的索引
        if "trace" in args.mod:
            input_ids = tokenizer.encode(result)
            flat_list = build_trace_indices(input_ids, args.template)
            indexs.append(sorted(flat_list))
        train_token.append(tokenizer.encode(result))
    
    ## 将token序列化并保存
    os.makedirs(f"/home/wth/few_vs_zero/data/sni/token{args.template}/{task}", exist_ok=True)
    with open(train_file, 'wb') as f:
        pickle.dump({"inputs": train_token,"indexs":indexs}, f)

## 加载模型并设置为训练模式
model = LlamaForCausalLM.from_pretrained(args.model, device_map='auto', torch_dtype=torch.bfloat16)
model.train()

## 损失函数
criterion = nn.CrossEntropyLoss(reduction="none")
out_data = [[0]*11008]*32


## 处理训练token并计算梯度
progress_bar = tqdm(total=len(train_token), leave=True, desc='Getting data')
for input_ids,index in zip(train_token, indexs):
    progress_bar.update(1)

    if len(input_ids)>1300:
        continue

    # if TOPK_POS:
    #     best_index = [index[i] for i in TOPK_POS]
    # else:
    best_index = index

    input_index = [i-1 for i in best_index]

    label_token = [input_ids[i] for i in input_index]

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

folder_path = f"/home/wth/few_vs_zero/data/sni/matrix/{task}/{args.mod}"
if not os.path.exists(folder_path): 
    # 如果不存在，则创建文件夹
    os.makedirs(folder_path)
    print(f"文件夹 {folder_path} 已创建。")
else:
    print(f"文件夹 {folder_path} 已经存在。")
with open(f"/home/wth/few_vs_zero/data/sni/matrix/{task}/{args.mod}/tp{args.template}_{args.shot}shot.json","w") as f:
    json.dump(out_data,f)