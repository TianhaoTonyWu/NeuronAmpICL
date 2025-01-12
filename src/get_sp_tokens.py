import few_vs_zero.src.globals_vars as globals_vars
from transformers import AutoTokenizer

sp_list = globals_vars.custom_special_tokens


tokenizer = AutoTokenizer.from_pretrained("/home/wth/model", add_bos_token=False)
sequences = []
tokens = []
for sp in sp_list:
    ids = tokenizer.encode(sp)
    sequences.append(ids)
    tokens.append(tokenizer.convert_ids_to_tokens(ids))


for k, v in zip(tokens, sequences):
    print(k, v)



print(sequences)
    
