import argparse
from types import MethodType
import json
import torch
from vllm import LLM, SamplingParams
import os
from utils import data_construct
from jinja2 import Template
from transformers import AutoTokenizer
import pickle
from tqdm import tqdm
from globals_vars import ct

# 设置数据路径和文件列表
data_path = "/home/wth/"

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="/home/wth/model")
parser.add_argument("-s", "--shot", type=str, default=5)
parser.add_argument("-t", "--task", type=str, default="task274_overruling_legal_classification.json")
parser.add_argument("-d", "--device", type=str, default="2,3")
parser.add_argument("-md", "--mod", type=str, default="LAPE")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

# 获取任务名称
task = args.task.split("_")[0]

# 加载数据并拆分为训练集和测试集
data_path = os.path.join(data_path, args.task)
with open(data_path, "r") as f:
    data = json.load(f)
    instruction = data["Definition"]
    instance = data["Instances"]
    data_number = len(instance)
    train, test = instance[:data_number//2], instance[data_number//2:]
    train_message = data_construct(train, instruction, shot=args.shot)
    test_message = data_construct(test, instruction, shot=args.shot)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)

# 处理训练数据并保存token
train_file = f'/home/wth/few_vs_zero/data/sni/token/{task}/train_{str(args.shot)}.pkl'
if os.path.exists(train_file):
    with open(train_file, 'rb') as f:
        data = pickle.load(f)
        train_token = data["inputs"]
else:
    train_token = []
    progress_bar = tqdm(total=len(train_message), desc='Train Processing data')
    for i in range(len(train_message)):
        progress_bar.update(1)
        message = train_message[i]
        template_str = ct
        template = Template(template_str)
        bos_token = ""
        eos_token = ""
        result = template.render(messages=message, bos_token=bos_token, eos_token=eos_token)
        train_token.append(tokenizer.encode(result))
    os.makedirs(f'/home/wth/few_vs_zero/data/sni/token/{task}', exist_ok=True)
    with open(train_file, 'wb') as f:
        pickle.dump({"inputs": train_token}, f)

# 处理测试数据并保存token
test_file = f'/home/wth/few_vs_zero/data/sni/token/{task}/test_{str(args.shot)}.pkl'
if os.path.exists(test_file):
    with open(test_file, 'rb') as f:
        data = pickle.load(f)
        test_token = data["inputs"]
        labels = data["labels"]
else:
    test_token = []
    labels = []
    progress_bar = tqdm(total=len(test_message), desc='Test Processing data')
    for i in range(len(test_message)):
        progress_bar.update(1)
        message = test_message[i]
        prompt, output = message[:-1], message[1]
        template_str = ct
        template = Template(template_str)
        bos_token = ""
        eos_token = ""
        result = template.render(messages=prompt, bos_token=bos_token, eos_token=eos_token)
        test_token.append(tokenizer.encode(result))
        labels.append(output["content"])
    progress_bar.close()
    os.makedirs(f'/home/wth/few_vs_zero/data/sni/token/{task}', exist_ok=True)
    with open(test_file, 'wb') as f:
        pickle.dump({"inputs": test_token, "labels": labels}, f)

# 检查模型类型是否为LLama
# is_llama = bool(args.model.lower().find('llama') >= 0)

# 加载模型
model = LLM(model=args.model, enforce_eager=True)

# 获取模型配置
max_length = model.llm_engine.model_config.max_model_len
num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
# intermediate_size = model.llm_engine.model_config.hf_config.intermediate_size if is_llama else model.llm_engine.model_config.hf_config.hidden_size * 4
intermediate_size = model.llm_engine.model_config.hf_config.intermediate_size 

# 初始化张量
sum1 = torch.zeros(num_layers, intermediate_size).to('cuda')
sum2 = torch.zeros(num_layers, intermediate_size).to('cuda')
sum3 = torch.zeros(num_layers, intermediate_size).to('cuda')
sum4 = torch.zeros(num_layers, intermediate_size).to('cuda')
over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')

# 定义前向传播函数
def factory(idx):
    def llama_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        i = gate_up.size(-1)
        gate_up[:, : i // 2] = torch.nn.SiLU()(gate_up[:, : i // 2])
        activation = gate_up[:, : i // 2].float()
        sum1[idx, :] += activation.sum(dim=0)
        sum2[idx, :] += activation.pow(2).sum(dim=0)
        sum3[idx, :] += activation.pow(3).sum(dim=0)
        sum4[idx, :] += activation.pow(4).sum(dim=0)
        over_zero[idx, :] += (activation > 0).sum(dim=0)
        x = gate_up[:, : i // 2] * gate_up[:, i // 2 :]
        x, _ = self.down_proj(x)
        return x
    return llama_forward

# 将前向传播函数绑定到模型层
for i in range(num_layers):
    # if is_llama:
    obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp
    # else:
    #     obj = model.llm_engine.model_executor.driver_worker.model_runner.model.transformer.h[i].mlp
    obj.forward = MethodType(factory(i), obj)

# 生成并保存输出
output = model.generate(prompt_token_ids=train_token, sampling_params=SamplingParams(max_tokens=1))
output = dict(n=len(train_token), sum1=sum1.to('cpu'), sum2=sum2.to('cpu'), sum3=sum3.to('cpu'), sum4=sum4.to('cpu'), over_zero=over_zero.to('cpu'))

save_path = os.path.join("/home/wth/few_vs_zero/data/sni//matrix", task)
os.makedirs(save_path, exist_ok=True)
save_file = os.path.join(save_path, f"activation_{args.mod}.pth")
torch.save(output, save_file)