import json
import random
import os
import string
import pickle
from tqdm import tqdm
from jinja2 import Template
from transformers import AutoTokenizer
import re
import globals_vars

def find_all_sublists(main,sub):
    # 获取子列表的长度
    sub_len = len(sub)
    # 存储所有匹配子列表的起始索引
    indices = []
    # 遍历主列表，长度减去子列表的长度
    for i in range(len(main) - sub_len + 1):
        # 如果在主列表中找到与子列表匹配的部分
        if main[i:i+sub_len] == sub:
            indices.append([j+1 for j in range(i,i+sub_len)])  # 添加索引到列表
    return indices  # 返回所有匹配的索引列表

def random_shot(data,shot):
     random_selection = random.sample(data, shot)
     return random_selection


##### SNI #####

def data_construct(data, instruction, shot=3):
    """
        Constructs data for the Super Natural Instructions dataset only
    """
    
    out_data = []
    instruction = "".join(instruction)

    # TODO: 固定的例子？
    for i in data:
        messages = [{"role":"system","content":instruction}]
        shot_content = random_shot(data, shot=shot)
        for j in shot_content:
            messages.append({"role":"user","content":"".join(j["input"])})
            messages.append({"role":"assistant","content":"".join(j["output"])})
        messages.append({"role":"user","content":"".join(i["input"])})
        messages.append({"role":"assistant","content":"".join(i["output"])})
        out_data.append(messages)
    return out_data


def clean_text(text : str):
    return text.strip().lower().rstrip(string.punctuation)


def process_and_save_tokens(task : str, shot : int, tokenizer_name: str, percentage : float, isTest : bool, temp : int):
    """
        FOR SNI dataset only

        returns train token or test token
    """
    
    DATA_PATH = "/home/wth/few_vs_zero/datasets/SNI"
    base_path='/home/wth/few_vs_zero/data/sni/token/'
    task_name = task.split("_")[0]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.chat_template = globals_vars.ct

    # Load data and split into training and test sets
    data_file = os.path.join(DATA_PATH, task)
    with open(data_file, "r") as f:
        data = json.load(f)
        instruction = data["Definition"]
        instances = data["Instances"][:6500]
        total_length = len(instances)
        sub_len = int(total_length*percentage)
        train, test = instances[:sub_len], instances[sub_len:]
    
      
        
    # Process and save test tokens and labels

    if isTest:
        test_message = data_construct(test, instruction, shot=shot)
        test_file = os.path.join(base_path, task_name, f'test_{shot}.pkl')
        if os.path.exists(test_file):
            with open(test_file, 'rb') as f:
                data = pickle.load(f)
                test_token, labels = data["inputs"], data["labels"]
        else:
            test_token, labels = [], []
            progress_bar = tqdm(total=len(test_message), desc='Processing SNI test data')
            for message in test_message:
                progress_bar.update(1)
                prompt, output = message[:-1], message[-1]
                template_str = tokenizer.chat_template
                template = Template(template_str)
                result = template.render(messages=prompt, bos_token="", eos_token="")
                test_token.append(tokenizer.encode(result))
                labels.append(output["content"])
            progress_bar.close()

            os.makedirs(os.path.dirname(test_file), exist_ok=True)
            with open(test_file, 'wb') as f:
                pickle.dump({"inputs": test_token, "labels": labels}, f)

        return test_token, labels
    
    else: # process train set
        train_message = data_construct(train, instruction, shot=shot)
        train_file = os.path.join(base_path, task_name, f'train_{shot}.pkl')
        train_token, indexs = [], []
        progress_bar = tqdm(total=len(train_message), desc='Processing SNI train data')

        for message in train_message:
            progress_bar.update(1)
            template = Template(tokenizer.chat_template)
            result = template.render(messages=message)
            input_ids = tokenizer.encode(result)
            flat_list = build_trace_indices(input_ids, temp)
            indexs.append(sorted(flat_list))
            train_token.append(input_ids)

        os.makedirs(os.path.dirname(train_file), exist_ok=True)
        with open(train_file, 'wb') as f:
            pickle.dump({"inputs": train_token, "indices": indexs}, f)

        return train_token, indexs


def evaluate_sni(output_file, tokenizer_name, labels, outputs):

    ## evaluate VLLM output

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    with open(output_file, "w", encoding="utf8") as f:
        # 使用列表推导式简化对 data_output 列表的构建
        data_output = [{"prompt": tokenizer.decode(output.prompt_token_ids),
                        "pred": output.outputs[0].text,
                        "label": l}
                    for l, output in zip(labels, outputs)]

        correct = 0

        for j in data_output:
            #  清理结果
            clean_pred = clean_text(j["pred"])
            clean_label = clean_text(j["label"])

            # 计算 EM
            if clean_pred == clean_label:
                correct+=1

            
            j['clean_pred'] = clean_pred
            j['clean_label'] = clean_label
            
            # 将 data_output 中的数据以 JSON 格式写入文件
            f.write(json.dumps(j,ensure_ascii=False) + "\n")

        num = len(data_output)
        accuracy = correct / num

        f.write(json.dumps({"num": num,"correct": correct,"acc": accuracy},ensure_ascii=False) + "\n")
        return num, correct, accuracy



def evaluate_sni1(output_file, tokenizer_name, labels, outputs):

    def extract_answer(prediction):
        match = re.search(r"(?<=Answer:\s)([\w-]+)", prediction)
        if match:
            return match.group(1)  
        return 'Invalid Answer'
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    with open(output_file, "w", encoding="utf8") as f:
        ## evaluate VLLM output
        data_output = [{"prompt": tokenizer.decode(output.prompt_token_ids),
                        "pred": output.outputs[0].text,
                        "label": l}
                    for l, output in zip(labels, outputs)]

        correct = 0

        for j in data_output:
            #  清理结果
            clean_pred = clean_text(extract_answer(j["pred"]))
            clean_label = clean_text(j["label"])

            # 计算 EM
            if clean_pred == clean_label:
                correct+=1

            
            j['clean_pred'] = clean_pred
            j['clean_label'] = clean_label
            
            # 将 data_output 中的数据以 JSON 格式写入文件
            f.write(json.dumps(j,ensure_ascii=False) + "\n")

        num = len(data_output)
        accuracy = correct / num

        f.write(json.dumps({"num": num,"correct": correct,"acc": accuracy},ensure_ascii=False) + "\n")
        return num, correct, accuracy
########################################################################################################





def evaluate_mmlu(output_file, tokenizer_name, labels, outputs):
    def extract_answer(prediction):
        # Regular expression to match "### " followed by exactly one letter
        match = re.search(r'#### ([A-Za-z])', prediction)
        if match:
            return match.group(1)  # Extract and return the letter
        return 'Invalid Answer'

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    with open(output_file, "w", encoding="utf8") as f:
        # 使用列表推导式简化对 data_output 列表的构建
        data_output = [{"prompt": tokenizer.decode(output.prompt_token_ids),
                        "pred": output.outputs[0].text,
                        "label": l}
                    for l, output in zip(labels, outputs)]

        correct = 0

        for j in data_output:

            clean_pred = extract_answer(j["pred"])
            clean_pred = clean_text(clean_pred)
            clean_label = clean_text(j["label"])

                # 计算 EM
            if clean_pred == clean_label:
                correct+=1
            
            j['clean_pred'] = clean_pred
            j['clean_label'] = clean_label
            
            # 将 data_output 中的数据以 JSON 格式写入文件
            f.write(json.dumps(j,ensure_ascii=False) + "\n")

        num = len(data_output)
        accuracy = correct / num

        f.write(json.dumps({"num": num,"correct": correct,"acc": accuracy},ensure_ascii=False) + "\n")
        return num, correct, accuracy

def evaluate_gsm(output_file, tokenizer_name, labels, outputs):
    # ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    
    # Include dollar sign 
    ANS_RE = re.compile(r"#### \$?(\-?[0-9\.\,]+)")
    INVALID_ANS = "[invalid]"

    def extract_answer(completion):
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return INVALID_ANS
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    with open(output_file, "w", encoding="utf8") as f:
        # 使用列表推导式简化对 data_output 列表的构建
        data_output = [{"prompt": tokenizer.decode(output.prompt_token_ids),
                        "pred": output.outputs[0].text,
                        "label": l}
                    for l, output in zip(labels, outputs)]

        correct = 0

        for j in data_output:
            clean_pred = extract_answer(j["pred"])
            clean_label = extract_answer(j["label"])

            if clean_label == clean_pred:
                correct += 1

            j['clean_pred'] = clean_pred
            j['clean_label'] = clean_label

            f.write(json.dumps(j,ensure_ascii=False) + "\n")

        num = len(data_output)
        accuracy = correct / num

        f.write(json.dumps({"num": num,"correct": correct,"acc": accuracy},ensure_ascii=False) + "\n")

        return num, correct, accuracy

def build_trace_indices(input_ids, template):
    """
        Determines which special tokens are tracked and find their trace
    """
    sub_sequence_list = globals_vars.sub_sequence_list
    track_index = []
    
    if template < 4:

        if template == 0:
            sp_ids = [0,1]

        elif template == 1:
            sp_ids = [0,1,4,5,6,7,8,9]

        elif template == 2:
            sp_ids = [0,1,10,11,12,13,14,15]

        elif template == 3:
            sp_ids = [0,1,16,17,18,19,20,21]

        for id in sp_ids:
            index = find_all_sublists(input_ids,sub_sequence_list[id])
            track_index += index 

    elif template == 4:
       ## final [/INST]
       index1 = find_all_sublists(input_ids,sub_sequence_list[1])[-1]
       track_index = [index1]

    
    elif template == 5:
        index1 = find_all_sublists(input_ids,sub_sequence_list[1])[-1][-1]
        track_index = [[index1]]

    else:
        raise ValueError("Undefined Template!")
      

    flat_list = [item for sublist in track_index for item in sublist]
    return flat_list



def process_data_and_save_tokens(save_path : str, train_message, test_message, val_message, tokenizer,  nshot : int, temp_num : int):
    """
        Process messages into tokens
    """
    save_dir = save_path + f"/tp{str(temp_num)}"
    
    ## TEST
    test_file = os.path.join(save_dir, f'test_{nshot}.pkl')
    if os.path.exists(test_file):
        print("File already exists")

    else:
        test_token, labels = [], []
        progress_bar = tqdm(total=len(test_message), desc=f'Processing {nshot}-shot Test data', leave=True)
        for message in test_message:
            progress_bar.update(1)
            prompt, output = message[:-1], message[-1]
            template_str = tokenizer.chat_template
            template = Template(template_str)
            result = template.render(messages=prompt, bos_token="", eos_token="")
            test_token.append(tokenizer.encode(result))
            labels.append(output["content"])
        progress_bar.close()


        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        with open(test_file, 'wb') as f:
            pickle.dump({"inputs": test_token, "labels": labels}, f)

    ## VAL
    if nshot == 0:
        val_file = os.path.join(save_dir, f'val_{nshot}.pkl')
        if os.path.exists(val_file):
            print("File already exists")
        else:
            val_token, labels = [], []
            progress_bar = tqdm(total=len(val_message), desc=f'Processing {nshot}-shot  Validation data', leave=True)
            for message in val_message:
                progress_bar.update(1)
                prompt, output = message[:-1], message[-1]
                template_str = tokenizer.chat_template
                template = Template(template_str)
                result = template.render(messages=prompt, bos_token="", eos_token="")
                val_token.append(tokenizer.encode(result))
                labels.append(output["content"])
            progress_bar.close()

            val_file = os.path.join(save_dir, f'val_{nshot}.pkl')
            os.makedirs(os.path.dirname(val_file), exist_ok=True)
            with open(val_file, 'wb') as f:
                pickle.dump({"inputs": val_token, "labels": labels}, f)

    ## TRAIN
    if nshot > 0:
        train_file = os.path.join(save_dir, f'train_{nshot}.pkl')
        if os.path.exists(train_file):
            print("File already exists")
        else:
            train_tokens = []
            indices = []
            progress_bar = tqdm(total=len(train_message), desc=f'Processing {nshot}-shot Train data', leave=True)
            for i in range(len(train_message)):
                progress_bar.update(1)
                message = train_message[i]
                template_str = tokenizer.chat_template
                templat = Template(template_str)
                result = templat.render(messages=message, bos_token="", eos_token="")
                input_ids = tokenizer.encode(result)
                flat_list = build_trace_indices(input_ids, temp_num)
                indices.append(sorted(flat_list))
                train_tokens.append(input_ids)

        
            os.makedirs(os.path.dirname(train_file), exist_ok=True)
            with open(train_file, 'wb') as f:
                pickle.dump({"inputs": train_tokens, "indices": indices}, f)

    



