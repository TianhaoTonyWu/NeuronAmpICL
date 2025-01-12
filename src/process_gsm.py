
from transformers import AutoTokenizer
import argparse
import globals_vars
from utils import random_shot, process_data_and_save_tokens
import re
import json


def construct_prompts(data, dev_data, instruction, shot=5, template=0):
    """
    Constructs prompts for LLM by combining instruction, demonstrations, and queries,
    with consistent formatting for questions and answers.

    Args:
        data (list): The main data containing questions and answers.
        dev_data (list): The development data for constructing demonstrations.
        instruction (str): The instruction to be provided to the system.
        shot (int): Number of demonstrations to include.
        template (int): Template format to use for formatting.

    Returns:
        list: A list of messages ready for LLM input.
    """
    def format_demo_and_query(qry, ans, isDemo):
        if template == 0:
            query = qry
            answer = ans
        elif template == 1:
            if isDemo:
                query = " <DEMO> <QUERY> {} </QUERY>".format(qry)
                answer = "<ANS> {} </ANS> </DEMO>".format(ans)
            else:
                query = "<QUERY> {} </QUERY>".format(qry)
                answer = ans
        elif template == 2:
            if isDemo:
                query = " <CASE> <QUESTION> {} </QUESTION>".format(qry)
                answer = "<ANSWER> {} </ANSWER> </CASE>".format(ans)
            else:
                query = "<QUESTION> {} </QUESTION>".format(qry)
                answer = ans
        elif template == 3:
            if isDemo:
                query = " <Eg> <Q> {} </Q>".format(qry)
                answer = "<A> {} </A> </Eg>".format(ans)
            else:
                query = "<Q> {} </Q>".format(qry)
                answer = ans

        elif template == 4:

            query = qry
            answer = ans

        else:
            raise ValueError("Undefined template.")
        return query, answer

    # Begin constructing the prompt
    out_data = []
    instruction = "".join(instruction)
    demo_content = random_shot(dev_data, shot=shot)

    for i in data:
        message = [{"role": "system", "content": instruction}]
        # Add demonstrations
        for j in demo_content:
            query, answer = format_demo_and_query(j["question"], j["answer"], isDemo=True)
            message.append({"role": "user", "content": query})
            message.append({"role": "assistant", "content": answer})
        # Add the main query
        query, answer = format_demo_and_query(i["question"], i["answer"], isDemo=False)
        message.append({"role": "user", "content": query})
        message.append({"role": "assistant", "content": answer})
        out_data.append(message)
    
    return out_data





def rm_calculators(data):
    """
        Remove calculator annotations from raw data
    """
    for instance in data:
        instance["answer"] = re.sub(r"<<.*?>>", "", instance["answer"])
    return data

def main(args):

    tokenizer_name = "/home/wth/model"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.chat_template = globals_vars.ct

    # load data
    test_data_file =  "/home/wth/few_vs_zero/datasets/GSM/test.jsonl"
    test_data = []
    with open(test_data_file, "r") as f:
        for line in f:
            test_data.append(json.loads(line))
    test_data = rm_calculators(test_data)
   
    total_length = len(test_data)

    train_data = []
    train_data_file = "/home/wth/few_vs_zero/datasets/GSM/train.jsonl"
    with open(train_data_file, "r") as f:
        for line in f:
            train_data.append(json.loads(line))

    dev_data = train_data[total_length:]
    assert len(dev_data) >= 5
    train_data = train_data[:total_length] # Let train size equal to test size
    dev_data = rm_calculators(dev_data)
    train_data = rm_calculators(train_data)
   

    instruction = "Solve the math problem. Let's think step by step. At the end, you MUST write the answer as an integer after '####'."
    # instruction = "Solve the math problem and provide a detailed explanation of the steps taken to arrive at the solution. Answer with only numbers. Add '#### ' in front of the answer."
    # instruction = "Solve the math problem and provide a detailed explanation of the steps taken to arrive at the solution. After the explanation, output the final solution like this: #### <solution>"
    
    train_message = construct_prompts(train_data, dev_data, instruction, shot=5, template=args.template)
    test_message = construct_prompts(test_data, dev_data, instruction, shot=5, template=0)
    
    process_data_and_save_tokens(
                                save_path=args.save_dir, 
                                train_message=train_message, 
                                test_message=test_message, 
                                val_message=None,  
                                tokenizer=tokenizer,
                                nshot=5,
                                temp_num=args.template
                                )


    test_message = construct_prompts(test_data, dev_data, instruction, shot=0, template=0)
    val_message = construct_prompts(train_data, dev_data, instruction, shot=0, template=0)

    process_data_and_save_tokens(
                            save_path=args.save_dir, 
                            train_message=None, 
                            test_message=test_message, 
                            val_message=val_message,  
                            tokenizer=tokenizer,
                            nshot=0,
                            temp_num=args.template
                            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default="/home/wth/few_vs_zero/datasets/GSM")
    parser.add_argument("--save_dir", "-s", type=str, default="/home/wth/few_vs_zero/data/gsm/token")
    parser.add_argument("--template", "-tp", type=int, default=4)
    args = parser.parse_args()
    main(args)