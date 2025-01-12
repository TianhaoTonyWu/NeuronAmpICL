from transformers import AutoTokenizer
import argparse
import globals
from utils import random_shot, process_data_for_simple_datasets
import json

def data_tranform(data):
    for ins in data:
        # Extracting the question stem and choices
        stem = ins["question"]["stem"]
        choices = ins["question"]["choices"]

        # Formatting the plain text
        plain_text = f"{stem}\n"
        for choice in choices:
            plain_text += f"{choice['label']}: {choice['text']}\n"

        # modify question
        ins["question"] = plain_text

    return data



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
            message.append({"role":"assistant","content":"".join("#### " + j["answerKey"])})
        message.append({"role":"user","content":"".join(i["question"])})
        message.append({"role":"assistant","content":"".join("#### " + i["answerKey"])})
        out_data.append(message)
    return out_data


def main(args):

    tokenizer_name = "/llm/wutianhao/model"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.chat_template = globals.ct

    # load data
    test_data_file =  "/home/wutianhao/few_vs_zero/datasets/CSQA/dev_rand_split.jsonl"
    test_data = []
    with open(test_data_file, "r") as f:
        for line in f:
            test_data.append(json.loads(line))
    test_data = data_tranform(test_data)
    total_length = len(test_data)

    train_data = []
    train_data_file = "/home/wutianhao/few_vs_zero/datasets/CSQA/train_rand_split.jsonl"
    with open(train_data_file, "r") as f:
        for line in f:
            train_data.append(json.loads(line))
    train_data = data_tranform(train_data)
    train_data = train_data[:total_length] # Let train size equal to test size

    instruction = "The following question is about commonsense among human beings. You have to choose one among the five choices: A, B, C, D and E. Provide your choice like this: #### <choice>"
    train_message = data_construct(train_data, instruction, shot=args.nshot)
    test_message = data_construct(test_data, instruction, shot=args.nshot)
    process_data_for_simple_datasets(train_message, test_message, args, tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nshot", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="/home/wutianhao/few_vs_zero/datasets/CSQA")
    parser.add_argument("--save_dir", "-s", type=str, default="/home/wutianhao/few_vs_zero/data/csqa/token")
    args = parser.parse_args()
    main(args)