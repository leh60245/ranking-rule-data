import argparse
import json
import os
from tqdm import tqdm
import pickle
import random

random.seed(42)

import torch



from sentence_transformers import util
from transformers import AutoTokenizer, AutoModel

# fmt: off
parser = argparse.ArgumentParser(description="Test a classification model with configurable hyperparameters.")
parser.add_argument("--model_identifier", type=str, default=None, help="Identifier for the model",)
parser.add_argument("--top_k_neg", type=int, default=None, help="Set negative sampling numbers",)
parser.add_argument("--device", type=str, default="cuda:6", help="Device to use for training")


args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
top_k_neg = args.top_k_neg  # negative 뽑는 갯수

# 데이터 로드
data = "data/rule/aihub_rules_pair_en.json"
with open(data, "r", encoding="UTF-8") as j:
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

if args.model_identifier == 'facebook/dragon-roberta-query-encoder':
    query_tokenizer = AutoTokenizer.from_pretrained('facebook/dragon-roberta-query-encoder')
    context_tokenizer = AutoTokenizer.from_pretrained('facebook/dragon-roberta-context-encoder')
    query_encoder = AutoModel.from_pretrained('facebook/dragon-roberta-query-encoder').to(device)
    context_encoder = AutoModel.from_pretrained('facebook/dragon-roberta-context-encoder').to(device)
elif args.model_identifier == 'intfloat/e5-large-v2':
    query_tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
    context_tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
    query_encoder = AutoModel.from_pretrained('intfloat/e5-large-v2').to(device)
    context_encoder = AutoModel.from_pretrained('intfloat/e5-large-v2').to(device)   
else:
    print("No encoder model")
    
ctx_input = context_tokenizer(
    rules_list, padding=True, truncation=True, max_length=512, return_tensors="pt"
)
ctx_input = {key: value.to(device) for key, value in ctx_input.items()}
ctx_emb = context_encoder(**ctx_input).last_hidden_state[:, 0, :]

# train sample
root_path = "data/train"
files = sorted([i for i in os.listdir(root_path) if i.endswith(".json")])

negative_sample_indices = []
for idx, file in enumerate(tqdm(files)):
    with open(os.path.join(root_path, file), "r", encoding="UTF-8") as j:
        try:
            file_caption = json.load(j)["outputs"]
        except:
            print(file)
            continue
    query_input = query_tokenizer(
        file_caption, padding=True, return_tensors="pt", truncation=True, max_length=512
    )
    query_input = {key: value.to(device) for key, value in query_input.items()}
    query_emb = query_encoder(**query_input).last_hidden_state[:, 0, :]

    initial_top_k_scores = util.dot_score(query_emb, ctx_emb)[0]
    initial_top_k_indices = torch.topk(initial_top_k_scores, k=50).indices.tolist()

    answer = file.split("_")[2]
    answer_index = int(answer[-2:]) - 1

    torch.cuda.empty_cache()
    initial_top_k_indices.remove(answer_index)
    negative_sample_indices.append(initial_top_k_indices)


sentence1 = []  # query
sentence2 = []  # document
labels = []  # 0(neg) or 1(pos)

for idx, file in enumerate(tqdm(files)):
    with open(os.path.join(root_path, file), "r", encoding="UTF-8") as j:
        try:
            file_caption = json.load(j)["outputs"]
        except:
            print(file)
            continue

    answer = file.split("_")[2]
    answer_index = int(answer[-2:]) - 1

    # positive sampling
    sentence1.append("query: " + file_caption)
    labels.append(1)
    sentence2.append("passage: " + rules[answer_index])
    # negative sampling
    for neg_rule_index in negative_sample_indices[idx][:top_k_neg]:
        sentence1.append("query: " + file_caption)
        labels.append(0)
        sentence2.append("passage: " + rules[neg_rule_index])

# Create directories for saving logs and models
data_dir = f'{args.model_identifier}'
os.makedirs(data_dir, exist_ok=True)

data = {"sentence1": sentence1, "sentence2": sentence2, "labels": labels}
with open(f"{data_dir}/data-train-neg{top_k_neg}.pkl", "wb") as file:
    pickle.dump(data, file)

# # valid sample
# root_path = "data/val"
# files = sorted([i for i in os.listdir(root_path) if i.endswith(".json")])

# negative_sample_indices = []
# for idx, file in enumerate(tqdm(files)):
#     with open(os.path.join(root_path, file), "r", encoding="UTF-8") as j:
#         try:
#             file_caption = json.load(j)["outputs"]
#         except:
#             print(file)
#             continue
#     query_input = query_tokenizer(
#         file_caption, padding=True, return_tensors="pt", truncation=True, max_length=512
#     )
#     query_input = {key: value.to(device) for key, value in query_input.items()}
#     query_emb = query_encoder(**query_input).last_hidden_state[:, 0, :]

#     initial_top_k_scores = util.cos_sim(query_emb, ctx_emb)[0]
#     initial_top_k_indices = torch.topk(initial_top_k_scores, k=50).indices.tolist()

#     answer = file.split("_")[2]
#     answer_index = int(answer[-2:]) - 1

#     torch.cuda.empty_cache()

#     initial_top_k_indices.remove(answer_index)
#     negative_sample_indices.append(initial_top_k_indices)

# sentence1 = []  # query
# sentence2 = []  # document
# labels = []  # 0(neg) or 1(pos)
# for idx, file in enumerate(tqdm(files)):
#     with open(os.path.join(root_path, file), "r", encoding="UTF-8") as j:
#         try:
#             file_caption = json.load(j)["outputs"]
#         except:
#             print(file)
#             continue

#     answer = file.split("_")[2]
#     answer_index = int(answer[-2:]) - 1

#     # positive sampling
#     sentence1.append("query: " + file_caption)  # pos
#     labels.append(1)  # pos
#     sentence2.append("passage: " + rules[answer_index])
#     # negative sampling
#     for neg_rule_index in negative_sample_indices[idx][:top_k_neg]:
#         sentence1.append("query: " + file_caption)  # neg
#         labels.append(0)  # neg
#         sentence2.append("passage: " + rules[neg_rule_index])  # 바로 다음 rule 지목

# data = {"sentence1": sentence1, "sentence2": sentence2, "labels": labels}
# with open(f"data-val-neg{top_k_neg}.pkl", "wb") as file:
#     pickle.dump(data, file)
