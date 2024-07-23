import json
import os
from tqdm import tqdm
import numpy as np
import torch
from sentence_transformers import util
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

query_tokenizer = AutoTokenizer.from_pretrained("facebook/dragon-roberta-query-encoder")
context_tokenizer = AutoTokenizer.from_pretrained(
    "facebook/dragon-roberta-context-encoder"
)
query_encoder = AutoModel.from_pretrained("facebook/dragon-roberta-query-encoder").to(
    device
)
context_encoder = AutoModel.from_pretrained(
    "facebook/dragon-roberta-context-encoder"
).to(device)

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


ctx_input = context_tokenizer(
    rules_list, padding=True, truncation=True, max_length=512, return_tensors="pt"
)
ctx_input = {key: value.to(device) for key, value in ctx_input.items()}
ctx_emb = context_encoder(**ctx_input).last_hidden_state[:, 0, :]


# Function to extract class from filename
def extract_class_from_filename(filename):
    parts = filename.split("_")
    class_id = int(parts[2].split("-")[-1])
    return class_id


# Inference function with top-50 ranking
def infer_file_class(
    file_caption, query_tokenizer, query_encoder, text_embeddings, k=50
):
    query_input = query_tokenizer(
        file_caption, padding=True, return_tensors="pt", truncation=True, max_length=512
    )
    query_input = {key: value.to(device) for key, value in query_input.items()}
    query_emb = query_encoder(**query_input).last_hidden_state[:, 0, :]

    # Compute cosine similarities
    cos_scores = util.pytorch_cos_sim(query_emb, text_embeddings)[0]
    top_k_indices = cos_scores.topk(k).indices.numpy()
    return top_k_indices


# Function to calculate Precision@k, Recall@k, and Accuracy@k
def compute_metrics(target, predictions, k):
    pred = predictions[:k]
    num_hit = len(set(pred).intersection(set(target)))
    precision = float(num_hit) / k if k > 0 else 0
    recall = float(num_hit) / len(target) if len(target) > 0 else 0
    accuracy = 1.0 if num_hit > 0 else 0.0
    return precision, recall, accuracy


# Metrics calculation for different top-k values
def calculate_metrics(
    file_folder_path, query_tokenizer, query_encoder, text_embeddings, top_k_values
):
    total_files = 0
    top_k_precision = {k: [] for k in top_k_values}
    top_k_recall = {k: [] for k in top_k_values}
    top_k_accuracy = {k: [] for k in top_k_values}

    for filename in tqdm(os.listdir(file_folder_path)):
        if filename.endswith(".json"):
            file_path = os.path.join(file_folder_path, filename)
            with open(file_path, "r", encoding="UTF-8") as j:
                file_caption = json.load(j)["outputs"]
            predicted_classes = infer_file_class(
                file_caption, query_tokenizer, query_encoder, text_embeddings, k=50
            )
            actual_class = (
                extract_class_from_filename(filename) - 1
            )  # Convert to 0-indexed

            total_files += 1
            targets = [actual_class]

            for k in top_k_values:
                precision, recall, accuracy = compute_metrics(
                    targets, predicted_classes, k
                )
                top_k_precision[k].append(precision)
                top_k_recall[k].append(recall)
                top_k_accuracy[k].append(accuracy)

    avg_top_k_precision = {
        k: sum(top_k_precision[k]) / total_files for k in top_k_values
    }
    avg_top_k_recall = {k: sum(top_k_recall[k]) / total_files for k in top_k_values}
    avg_top_k_accuracy = {k: sum(top_k_accuracy[k]) / total_files for k in top_k_values}

    return avg_top_k_precision, avg_top_k_recall, avg_top_k_accuracy


# Calculate metrics for top-1, top-5, top-10
file_folder_path = "data/test"
top_k_values = [i for i in range(1, 51)]
top_k_precision, top_k_recall, top_k_accuracy = calculate_metrics(
    file_folder_path, query_tokenizer, query_encoder, ctx_emb, top_k_values
)

for k in top_k_values:
    print(
        f"Top-{k} Precision: {top_k_precision[k]} \t Recall: {top_k_recall[k]} \t Accuracy: {top_k_accuracy[k]}"
    )
