import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import process_and_save_tokens
from tqdm.auto import tqdm 
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

MAX_NEW_TOKENS = 6
model_path = "/llm/wutianhao/model"
out_file_dir = '/home/wutianhao/few_vs_zero/results/conf_score'
os.makedirs(out_file_dir, exist_ok=True)

def get_confidence_score(label: str, outputs):
    # Prepare vocab probabilities once
    vocab_probs = torch.nn.functional.softmax(outputs.logits, dim=-1)  # shape: (batch_size, seq_len, vocab_size)
    label_enc = tokenizer(label, return_tensors='pt', add_special_tokens=False)
    label_ids = label_enc["input_ids"].squeeze().numpy()
    
    mean_label_confs = []

    for i in tqdm(range(outputs.logits.shape[1]), desc="\nFinding Confidence Score", ):
        
        # Extract relevant probabilities
        token_probs = vocab_probs[:, i, :].squeeze().tolist()  # Get probabilities for timestep i
        next_token_dict = {t : p for t, p in enumerate(token_probs)}

        # Compute average confidence score over tokens of a label
        sum_conf = sum(next_token_dict.get(l, 0.) for l in label_ids)
        mean_label_confs.append(sum_conf / len(label_ids))

    return mean_label_confs

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Prepare input
test_input_ids, labels = process_and_save_tokens(
    task="task274_overruling_legal_classification.json",
    shot=5,
    tokenizer_name=model_path,
    percentage=0.01
)

input_ids = torch.tensor(test_input_ids, dtype=torch.int64).unsqueeze(0).to('cuda')

for id in input_ids:
    # Generate answer
    output_ids = input_ids
    for i in range(MAX_NEW_TOKENS):

        # Get logits
        with torch.no_grad():
            outputs = model(output_ids)  # Generate only one next token
        next_token_logits = outputs.logits[:, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)

        # Generate next token
        next_token_id = torch.multinomial(next_token_probs, num_samples=1)
        output_ids = torch.cat([output_ids, next_token_id], dim=1)

    #####################################

    # Get confidence score
    mean_conf_scores = get_confidence_score(labels[0], outputs)

    # Plot conf score
    all_toks = [t.item() for t in output_ids[0]]

    print(len(all_toks))
    all_toks = all_toks[:-1]
    all_toks = tokenizer.convert_ids_to_tokens(all_toks)


    """
    # Create a DataFrame
    data = pd.DataFrame({
        'Index' : np.arange(len(all_toks)),
        'Token': all_toks,
        'Confidence Score': mean_conf_scores
    })


    # Create the bar plot using Seaborn
    plt.figure(figsize=(100, 6))
    sns.barplot(data=data, x='Index', y='Confidence Score', dodge=False)
    plt.title("Confidence Score")
    plt.xlabel("Token Index")
    plt.ylabel("Score")
    plt.xticks(ticks=data['Index'], labels=data['Token'], rotation=50, ha='right')
    plt.subplots_adjust(bottom=0.2)
    plt.savefig("/home/wutianhao/few_vs_zero/results/conf_score/test.png")
    """
    file_name = '/confidence.jsonl'
    # Save results
    with open(out_file_dir + file_name, 'w') as jsonl_file:
        jsonl_file.write(json.dumps(mean_conf_scores))







