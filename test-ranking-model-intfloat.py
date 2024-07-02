import json
import os
from tqdm import tqdm
import numpy as np
import torch
from sentence_transformers import util
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
encoder = AutoModel.from_pretrained('intfloat/e5-large-v2').to(device)

# load rule data
rule_path = "data/rule/aihub_rules_pair_en.json"
with open(rule_path, "r", encoding="UTF-8") as j:
    aihub_rule = json.load(j)

rules = {}
rules_list = []
cnt = 0
for first_key in aihub_rule.keys():
    for second_key in aihub_rule[first_key].keys():
        third_keys = list(aihub_rule[first_key][second_key].keys())
        rule = f"the accident type is {first_key.lower()}, the operation type is {second_key[:-2].lower()}, the normal scenario is {aihub_rule[first_key][second_key][third_keys[0]].lower()}, the abnormal scenario is {aihub_rule[first_key][second_key][third_keys[1]].lower()}."
        answer_index = int(third_keys[-1][2:]) - 1
        rules_list.append(rule)
        rules[answer_index] = rule
        if cnt != answer_index:
            print(cnt, answer_index)
            break
        cnt += 1
        
        
ctx_input = tokenizer(
    rules_list, padding=True, truncation=True, max_length=512, return_tensors="pt"
)
ctx_input = {key: value.to(device) for key, value in ctx_input.items()}
ctx_emb = encoder(**ctx_input).last_hidden_state[:, 0, :]

# test sample
root_path = "data/test"
files = sorted([i for i in os.listdir(root_path) if i.endswith(".json")])


top_50_indices = []
for idx, file in enumerate(tqdm(files)):
    with open(os.path.join(root_path, file), "r", encoding="UTF-8") as j:
        try:
            file_caption = json.load(j)["outputs"]
        except:
            print(file)
            continue
    query_input = tokenizer(
        file_caption, padding=True, return_tensors="pt", truncation=True, max_length=512
    )
    query_input = {key: value.to(device) for key, value in query_input.items()}
    query_emb = encoder(**query_input).last_hidden_state[:, 0, :]

    initial_top_k_scores = util.dot_score(query_emb, ctx_emb)[0]
    initial_top_k_indices = torch.topk(initial_top_k_scores, k=50).indices.tolist()

    answer = file.split("_")[2]
    answer_index = int(answer[-2:]) - 1

    top_50_indices.append(initial_top_k_indices)
    
    
# get Top-k Acc
def find_rank_of_answer_in_results(final_indices, answer_index):
    try:
        return final_indices.index(answer_index) + 1
    except ValueError:
        return -1
    
for final_top_k in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
    correct = 0
    
    for idx, file in enumerate(tqdm(files)):
        
        final_top_k_indices = top_50_indices[idx]


        answer = file.split('_')[2]
        answer_index = int(answer[2:]) - 1
            
        rank = find_rank_of_answer_in_results(final_top_k_indices, answer_index)

        if rank <= final_top_k and rank != -1:
            correct += 1

    print(f"Top-{final_top_k} accuracy:", correct / len(files))