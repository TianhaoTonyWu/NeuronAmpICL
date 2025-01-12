import os
import pickle
from tqdm import tqdm
from jinja2 import Template
from transformers import AutoTokenizer
import argparse
import pandas as pd
import few_vs_zero.src.globals_vars as globals_vars
from utils import  build_trace_indices

choices = ["A", "B", "C", "D"]
subjects = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
# subjects = ['management']
merged = ["stem", "humanities", "social_sciences", "others"]
# merged = []

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s



###################################################
def format_demo_and_query(qry, ans, isDemo):
    if args.template == 0:
        query = qry
        answer = ans
    elif args.template == 1:
        if isDemo:
            query = " <DEMO> <QUERY> {} </QUERY>".format(qry)
            answer = "<ANS> #### {} </ANS> </DEMO>".format(ans)
        else:
            query = "<QUERY> {} </QUERY>".format(qry)
            answer = ans
    elif args.template == 2:
        if isDemo:
            query = " <CASE> <QUESTION> {} </QUESTION>".format(qry)
            answer = "<ANSWER> #### {} </ANSWER> </CASE>".format(ans)
        else:
            query = "<QUESTION> {} </QUESTION>".format(qry)
            answer = ans
    elif args.template == 3:
        if isDemo:
            query = " <Eg> <Q> {} </Q>".format(qry)
            answer = "<A> #### {} </A> </Eg>".format(ans)
        else:
            query = "<Q> {} </Q>".format(qry)
            answer = ans

    elif args.template == 4:
        query = qry
        answer = ans
    else:
        raise ValueError("Undefined template.")
    return query, answer
######################################################

def format_example(df, idx, include_answer, isDemo):

    # Construct question with choices
    query = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        query += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])

    answer = df.iloc[idx, k + 1]

    query, answer = format_demo_and_query(query, answer, isDemo)

    message = [{"role":"user","content": query}]  
    if include_answer:
        message.append({"role":"assistant","content": answer})
    return message


def gen_prompt(dev_df, subject, k):
    # instruction = "The following are multiple choice questions (with answers) about{}.".format(format_subject(subject))
    # instruction = "The following are multiple choice questions (with answers) about{}. Answer with only a single letter.".format(format_subject(subject))
    # instruction = "The following are multiple choice questions (with answers) about{}. Answer with only a single letter like this: #### <letter>..".format(format_subject(subject))
    instruction = "The following are multiple choice questions (with answers) about{}. The answer is one of A, B, C or D. Solve the question and output the answer like this: #### <letter>.".format(format_subject(subject))
   
    message = [{"role":"system","content":instruction}]
    for i in range(k):
        message += format_example(dev_df, i, include_answer=True, isDemo=True)
    return message



def main(args):
    tokenizer_name = "/home/wth/model"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.chat_template = globals_vars.ct
    save_dir = args.save_dir + str(args.template)
    # Process test data for all subcategories
    
    for subject in tqdm(subjects, desc=f"Processing test tokens" ):
        for nshot in [5,0]:
            print("\n\n"+subject, nshot)
            # Load dataframes
            # Original dataset has no header for csv
            dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), delimiter=",", header=None)[:nshot] 
            test_df = pd.read_csv(os.path.join(args.data_dir, "new_test", subject + "_test.csv"), delimiter=",")
           

            # Process test data
            test_messages = []
            labels = []
            test_tokens = []
            for i in tqdm(range(test_df.shape[0]), desc="Constructing test prompts: ", leave=True):
                k = nshot
                demo = gen_prompt(dev_df, subject, k)
                prompt_end = format_example(test_df, i, include_answer=False, isDemo=False) # Query
                prompt = demo + prompt_end
                label = test_df.iloc[i, test_df.shape[1]-1]
                test_messages.append(prompt)
                labels.append(label)
                
            for message in tqdm(test_messages, desc="Encoding test data: ", leave=True):   
                template_str = tokenizer.chat_template
                template = Template(template_str)
                result = template.render(messages=message, bos_token="", eos_token="")
                test_tokens.append(tokenizer.encode(result))

            test_file = os.path.join(save_dir, subject, f'test_{nshot}.pkl')
            os.makedirs(os.path.dirname(test_file), exist_ok=True)
            with open(test_file, 'wb') as f:
                pickle.dump({"inputs": test_tokens, "labels": labels}, f)
    
            #### Process val tokesn for MMLU
            if args.template == 0 and nshot == 0:
                val_df = pd.read_csv(os.path.join(args.data_dir, "new_val", subject + "_val.csv"), delimiter=",")
                val_messages = []
                val_labels = []
                val_tokens = []
                for i in tqdm(range(val_df.shape[0]), desc="Constructing VAL prompts: ", leave=True):
                    k = 0
                    demo = gen_prompt(dev_df, subject, k)
                    prompt_end = format_example(val_df, i, include_answer=False, isDemo=False) # Query
                    prompt = demo + prompt_end
                    label = val_df.iloc[i, val_df.shape[1]-1]
                    val_messages.append(prompt)
                    val_labels.append(label)
                    
                for message in tqdm(val_messages, desc="Encoding VAL data: ", leave=True):   
                    template_str = tokenizer.chat_template
                    template = Template(template_str)
                    result = template.render(messages=message, bos_token="", eos_token="")
                    val_tokens.append(tokenizer.encode(result))

                val_file = os.path.join(save_dir, subject, f'val_{nshot}.pkl')
                os.makedirs(os.path.dirname(val_file), exist_ok=True)
                with open(val_file, 'wb') as f:
                    pickle.dump({"inputs": val_tokens, "labels": val_labels}, f)


    for category in merged:
        print("\n\n"+category,": ")
        dev_file = pd.read_csv(os.path.join(args.data_dir, "merged_dev", category + "_dev.csv"), delimiter=",")
        dev_df = dev_file[:5]
        dev_df = dev_df.sample(frac=1, random_state=42)
        train_df = pd.read_csv(os.path.join(args.data_dir, "merged_train", category + "_train.csv"), delimiter=",")

  

        ##### Process train tokens for MMLU
        
        train_messages = []
        for i in tqdm(range(train_df.shape[0]), desc="Constructing training prompts: ", leave=True):
            k = 5
            demo = gen_prompt(dev_df, category, k) # instruction and demo
            prompt_end = format_example(train_df, i, include_answer=True, isDemo=False) # query
            prompt = demo + prompt_end
            train_messages.append(prompt)

        train_tokens = []
        indices = []
        progress_bar = tqdm(total=len(train_messages), desc='Encoding training data', leave=True)
        for i in range(len(train_messages)):
            progress_bar.update(1)
            message = train_messages[i]
            template_str = tokenizer.chat_template
            template = Template(template_str)
            result = template.render(messages=message, bos_token="", eos_token="")
            input_ids = tokenizer.encode(result)
            flat_list = build_trace_indices(input_ids, args.template)
            indices.append(sorted(flat_list))
            train_tokens.append(input_ids)

        train_file = os.path.join(save_dir, category, f'train_5.pkl')
        os.makedirs(os.path.dirname(train_file), exist_ok=True)
        with open(train_file, 'wb') as f:
            pickle.dump({"inputs": train_tokens, "indices": indices}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default="/home/wth/few_vs_zero/datasets/MMLU")
    parser.add_argument("--save_dir", "-s", type=str, default="/home/wth/few_vs_zero/data/mmlu/token")
    parser.add_argument("--template", "-tp", type=int, default=0)
    args = parser.parse_args()
    main(args)