import json
import argparse
import os
from transformers import AutoTokenizer
import globals_vars
from utils import  random_shot, process_data_and_save_tokens


## process token for SNI dataset
# TODO： Need debugging
def construct_prompts(data, instruction, shot, template):
    """
        apply different templates to prompt
    """


    def format_demo_and_query(qry, ans, isDemo):

        if template == 0 or template == 4 or template == 5:
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
    
        else:
            raise ValueError("Undefined template.")
        return query, answer

    # Begin constructing the prompt
    out_data = []
    instruction = "".join(instruction)
    demo_content = random_shot(data, shot=shot)

    for i in data:
        message = [{"role": "system", "content": instruction}]
        # Add demonstrations
        for j in demo_content:
            query, answer = format_demo_and_query("".join(j["input"]), "".join(j["output"]), isDemo=True)
            message.append({"role": "user", "content": query})
            message.append({"role": "assistant", "content": answer})
        # Add the main query
        query, answer = format_demo_and_query("".join(i["input"]), "".join(i["output"]), isDemo=False)
        message.append({"role": "user", "content": query})
        message.append({"role": "assistant", "content": answer})
        out_data.append(message)
    
    return out_data

def main(args):
    tasks = [
    "task242_tweetqa_classification.json",
    "task274_overruling_legal_classification.json",
    "task1447_drug_extraction_ade.json",
    # "task403_creak_commonsense_inference.json",
    # "task645_summarization.json",
    # "task475_yelp_polarity_classification.json"

    ]

    kshot = args.kshot

    DATA_PATH = "/home/wth/few_vs_zero/datasets/SNI"

    base_save_path='/home/wth/few_vs_zero/data/sni/'


    tokenizer = AutoTokenizer.from_pretrained("/home/wth/model")
    tokenizer.chat_template = globals_vars.ct

    # Load data and split into training and test sets

    
    for task in tasks:
        task_num = task.split("_")[0]  
        # sni/taskxxx/token
        task_save_path = base_save_path + f"k{kshot}token/{task_num}"

        data_file = os.path.join(DATA_PATH, task)
        with open(data_file, "r") as f:
            data = json.load(f)
        instruction = data["Definition"]
        instances = data["Instances"][:6500]
        total_length = len(instances)
        sub_len = total_length // 2
        train, test = instances[:sub_len], instances[sub_len:]

        # k-shot train and test
        train_message = construct_prompts(train, instruction, kshot, args.template)
        test_message = construct_prompts(test, instruction, kshot, 0)

        process_data_and_save_tokens(
                            save_path=task_save_path, 
                            train_message=train_message, 
                            test_message=test_message, 
                            val_message=None,  
                            tokenizer=tokenizer,
                            nshot=kshot,
                            temp_num=args.template
                            )

        # 0-shot test and valitation
        test_message = construct_prompts(test, instruction, 0, 0)
        val_message = construct_prompts(train, instruction, 0, 0)

        process_data_and_save_tokens(
                        save_path=task_save_path, 
                        train_message=None, 
                        test_message=test_message, 
                        val_message=val_message,  
                        tokenizer=tokenizer,
                        nshot=0,
                        temp_num=args.template
                        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", "-tp", type=int, default=0)
    parser.add_argument("--kshot", "-k", type=int, default=1)
    args = parser.parse_args()
    main(args)