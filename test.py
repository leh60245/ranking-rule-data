import argparse
import json
import os
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# fmt: off
parser = argparse.ArgumentParser(description="Test a classification model with configurable hyperparameters.")
parser.add_argument("--model_identifier", type=str, default="cross-encoder/stsb-roberta-large", help="Identifier for the model",)
parser.add_argument("--model_save_path", type=str, default=None, help="load the model")
parser.add_argument("--reranking", type=str, default=None, help="Ranking result")
parser.add_argument("--device", type=str, default="cuda:6", help="Device to use for training")

args = parser.parse_args()

# model load
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(args.model_identifier)
model = AutoModelForSequenceClassification.from_pretrained(args.model_identifier)

if args.model_save_path:
    checkpoint = torch.load(args.model_save_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
model.eval()

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

# load file name
root_path = "data/test"
files = sorted([i for i in os.listdir(root_path) if i.endswith(".json")])

# load ranking data
if args.reranking:
    with open(args.reranking, 'r') as file:
        reranking_data = json.load(file)
    rules_list = []

# get Top-50 indices list
save_top_k_indices = []
for idx, file in enumerate(tqdm(files)):
    with open(os.path.join(root_path, file), "r", encoding="UTF-8") as j:
        try:
            file_caption = json.load(j)["outputs"]
        except:
            print(file)
            continue

    logits_list = []
    for rule in rules_list:
        query_input = tokenizer(
            file_caption,
            rule,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        query_input = {key: value.to(device) for key, value in query_input.items()}
        # 모델 예측
        with torch.no_grad():
            outputs = model(**query_input)
            logits = outputs.logits.item()  # logits 값을 리스트에 추가
            logits_list.append(logits)
    logits_list = torch.tensor(logits_list)

    top_k_indices = torch.topk(logits_list, k=50).indices.tolist()
    save_top_k_indices.append(top_k_indices)

# get Top-k Acc
def find_rank_of_answer_in_results(final_indices, answer_index):
    try:
        return final_indices.index(answer_index) + 1
    except ValueError:
        return -1
    
for final_top_k in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
    correct = 0

    for idx, file in enumerate(tqdm(files)):
        final_top_k_indices = save_top_k_indices[idx][:final_top_k]

        answer = file.split('_')[2]
        answer_index = int(answer[2:]) - 1

        rank = find_rank_of_answer_in_results(final_top_k_indices, answer_index)

        if rank <= final_top_k and rank != -1:
            correct += 1

    print(f"Top-{final_top_k} accuracy", correct / len(files))
