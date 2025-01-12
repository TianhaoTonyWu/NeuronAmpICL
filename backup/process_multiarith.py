import os
import pickle
from tqdm import tqdm
from jinja2 import Template
from transformers import AutoTokenizer
import argparse
import globals
from utils import find_all_sublists, random_shot
import json



def data_construct(data, instruction, shot=5):

    """
        adapted from same function in utils.py
    """

    out_data = []
    instruction = "".join(instruction)

    for i in data:
        message = [{"role":"system","content":instruction}]
        demo_content = random_shot(data, shot=shot)
        for j in demo_content:
            message.append({"role":"user","content":"".join(j["question"])})
            message.append({"role":"assistant","content":"".join("#### " + j["final_ans"])})
        message.append({"role":"user","content":"".join(i["question"])})
        message.append({"role":"assistant","content":"".join("#### " + i["final_ans"])})
        out_data.append(message)
    return out_data


def main(args):

    tokenizer_name = "/llm/wutianhao/model"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.chat_template = globals.ct

    # load data
    test_data_file =  "/home/wutianhao/few_vs_zero/datasets/MultiArith/test.json"
    with open(test_data_file, "r") as f:
        test_data = json.load(f)
   
    train_data_file = "/home/wutianhao/few_vs_zero/datasets/MultiArith/train.json"
    with open(train_data_file, "r") as f:
        train_data = json.load(f)

    instruction = "Solve the math problem and provide only the final solution like this: #### <solution>."
    train_message = data_construct(train_data, instruction, shot=args.nshot)
    test_message = data_construct(test_data, instruction, shot=args.nshot)
    
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nshot", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="/home/wutianhao/few_vs_zero/datasets/MultiArith")
    parser.add_argument("--save_dir", "-s", type=str, default="/home/wutianhao/few_vs_zero/data/MultiArith/token")
    args = parser.parse_args()
    main(args)