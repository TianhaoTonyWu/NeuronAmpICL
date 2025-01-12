import argparse
from types import MethodType
import torch
from vllm import LLM, SamplingParams
import os
import pickle

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="/home/wth/model")
parser.add_argument("-ct","--category", type=str, default="social_sciences") # 4 categories
parser.add_argument("-md", "--mod", type=str, default="LAPE")
parser.add_argument("-tp", "--template", type=int, default=0)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


train_file = f'/home/wth/few_vs_zero/data/mmlu/token{args.template}/{args.category}/train_5.pkl'
if os.path.exists(train_file):
    with open(train_file, 'rb') as f:
        data = pickle.load(f)
        train_token = data["inputs"]

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

    obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp

    obj.forward = MethodType(factory(i), obj)

# 生成并保存输出
output = model.generate(prompt_token_ids=train_token, sampling_params=SamplingParams(max_tokens=1))
output = dict(n=len(train_token), sum1=sum1.to('cpu'), sum2=sum2.to('cpu'), sum3=sum3.to('cpu'), sum4=sum4.to('cpu'), over_zero=over_zero.to('cpu'))

save_path = os.path.join("/home/wth/few_vs_zero/data/mmlu/matrix", args.category)
os.makedirs(save_path, exist_ok=True)
save_file = os.path.join(save_path, f"activation_tp{args.template}_{args.mod}.pth")
torch.save(output, save_file)