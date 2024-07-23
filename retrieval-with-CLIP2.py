import argparse
import json
import os
from PIL import Image
import glob
from tqdm import tqdm

import torch
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel
from sentence_transformers import SentenceTransformer, util
from PIL import Image, ImageFile

# fmt: off
parser = argparse.ArgumentParser(description="Test a classification model with configurable hyperparameters.")
parser.add_argument("--model_identifier", type=str, default="cross-encoder/stsb-roberta-large", help="Identifier for the model",)
parser.add_argument("--model_save_path", type=str, default=None, help="load the model")
parser.add_argument("--reranking", type=str, default=None, help="Ranking result")
parser.add_argument("--device", type=str, default="cuda:6", help="Device to use for training")

args = parser.parse_args()

# model load
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(device)

model_name = "google/siglip-so400m-patch14-384"
image_folder_path = "/data/jwsuh/construction/sampled_test/"
model = AutoModel.from_pretrained(model_name).to(device)
processor = AutoProcessor.from_pretrained(model_name)

# if args.model_save_path:
#     checkpoint = torch.load(args.model_save_path, map_location=torch.device('cpu'))
#     img_model.load_state_dict(checkpoint["model_state_dict"])

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

# Function to extract class from filename
def extract_class_from_filename(filename):
    parts = filename.split('_')
    class_id = int(parts[2].split('-')[-1])
    return class_id

# Inference function with top-50 ranking
def infer_image_class(image_path, model, processor, class_texts, k=50):
    # Load and preprocess the image
    image = Image.open(image_path)
    image = image.convert("RGB")
    
    # Encode the image 
    inputs = processor(text=class_texts, images=image, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    
    # Compute cosine similarities
    logits_per_image  = outputs.logits_per_image[0]
    # print(logits_per_image)
    top_k_indices = logits_per_image.topk(k).indices.cpu().numpy().tolist()
    # print(top_k_indices)
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
def calculate_metrics(image_folder_path, model, processor, class_texts, top_k_values):
    total_images = 0
    top_k_precision = {k: [] for k in top_k_values}
    top_k_recall = {k: [] for k in top_k_values}
    top_k_accuracy = {k: [] for k in top_k_values}

    for filename in tqdm(os.listdir(image_folder_path)):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_folder_path, filename)
            predicted_classes = infer_image_class(image_path, model, processor, class_texts, k=50)
            actual_class = extract_class_from_filename(filename) - 1  # Convert to 0-indexed

            total_images += 1
            targets = [actual_class]

            for k in top_k_values:
                precision, recall, accuracy = compute_metrics(targets, predicted_classes, k)
                top_k_precision[k].append(precision)
                top_k_recall[k].append(recall)
                top_k_accuracy[k].append(accuracy)

    avg_top_k_precision = {k: sum(top_k_precision[k]) / total_images for k in top_k_values}
    avg_top_k_recall = {k: sum(top_k_recall[k]) / total_images for k in top_k_values}
    avg_top_k_accuracy = {k: sum(top_k_accuracy[k]) / total_images for k in top_k_values}

    return avg_top_k_precision, avg_top_k_recall, avg_top_k_accuracy

# Calculate metrics for top-1, top-5, top-10
top_k_values = [i for i in range(1,51)]
top_k_precision, top_k_recall, top_k_accuracy = calculate_metrics(image_folder_path, model, processor, rules_list, top_k_values)

for k in top_k_values:
    print(f"Top-{k} Precision: {top_k_precision[k]} \t Recall: {top_k_recall[k]} \n Accuracy: {top_k_accuracy[k]}")
