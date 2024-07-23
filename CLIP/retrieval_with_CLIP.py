import argparse
import json
import os
from PIL import Image
from tqdm import tqdm
import torch
import open_clip
from sentence_transformers import util

# Argument parsing for command line execution
parser = argparse.ArgumentParser(
    description="Test a classification model with configurable hyperparameters."
)
parser.add_argument(
    "--device", type=str, default="cuda:4", help="Device to use for inference"
)
args = parser.parse_args()

# Set device to GPU if available
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image folder path
image_folder_path = "/data/jwsuh/construction/sampled_test/"

# Load rule data from JSON file
rule_path = "data/rule/aihub_rules_pair_en.json"
with open(rule_path, "r", encoding="UTF-8") as j:
    aihub_rule = json.load(j)

# Prepare rules and their corresponding descriptions
rules = {}
rules_list = []
cnt = 0
for accident_type in aihub_rule.keys():
    for operation_type in aihub_rule[accident_type].keys():
        scenarios = list(aihub_rule[accident_type][operation_type].keys())
        rule_description = (
            f"the accident type is {accident_type.lower()}, the operation type is {operation_type[:-2].lower()}, "
            f"the normal scenario is {aihub_rule[accident_type][operation_type][scenarios[0]].lower()}, "
            f"the abnormal scenario is {aihub_rule[accident_type][operation_type][scenarios[1]].lower()}."
        )
        answer_index = int(scenarios[-1][2:]) - 1
        rules_list.append(rule_description)
        rules[answer_index] = rule_description
        if cnt != answer_index:
            print(f"Error in rule indexing: {cnt} != {answer_index}")
            break
        cnt += 1


# Function to extract class from filename
def extract_class_from_filename(filename, get_case = None):
    parts = filename.split("_")
    class_id = int(parts[2].split("-")[-1])
    if get_case is not None:
        return parts
    return class_id


# Inference function with top-50 ranking
def infer_image_class(image_path, model, processor, text_features, k=50):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = processor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        cos_scores = util.pytorch_cos_sim(image_features, text_features)[0]

    top_k_indices = cos_scores.topk(k).indices.cpu().numpy()
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
def calculate_metrics(image_folder_path, model, processor, text_features, top_k_values):
    total_images = 0
    top_k_precision = {k: [] for k in top_k_values}
    top_k_recall = {k: [] for k in top_k_values}
    top_k_accuracy = {k: [] for k in top_k_values}

    for filename in tqdm(os.listdir(image_folder_path)):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_folder_path, filename)
            predicted_classes = infer_image_class(
                image_path, model, processor, text_features, k=50
            )
            actual_class = (
                extract_class_from_filename(filename) - 1
            )  # Convert to 0-indexed

            total_images += 1
            targets = [actual_class]

            for k in top_k_values:
                precision, recall, accuracy = compute_metrics(
                    targets, predicted_classes, k
                )
                top_k_precision[k].append(precision)
                top_k_recall[k].append(recall)
                top_k_accuracy[k].append(accuracy)

    avg_top_k_precision = {
        k: sum(top_k_precision[k]) / total_images for k in top_k_values
    }
    avg_top_k_recall = {k: sum(top_k_recall[k]) / total_images for k in top_k_values}
    avg_top_k_accuracy = {
        k: sum(top_k_accuracy[k]) / total_images for k in top_k_values
    }

    return avg_top_k_precision, avg_top_k_recall, avg_top_k_accuracy


# List of available pretrained models
model_list = [("ViT-H-14-378-quickgelu", "dfn5b")]


# Open the result file to save the metrics
result_file_path = "model_comparison_results6.txt"
with open(result_file_path, "w") as result_file:
    # Iterate over each model
    for backbone_name, model_name in model_list:
        # Load the model and processor
        model, _, processor = open_clip.create_model_and_transforms(
            backbone_name, pretrained=model_name
        )
        model.to(device)
        model.eval()
        tokenizer = open_clip.get_tokenizer(backbone_name)

        # Encode text features once
        with torch.no_grad():
            text_tokens = tokenizer(rules_list).to(device)
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Calculate metrics for different top-k values
        top_k_values = list(range(1, 51))
        avg_top_k_precision, avg_top_k_recall, avg_top_k_accuracy = calculate_metrics(
            image_folder_path, model, processor, text_features, top_k_values
        )

        # Write the results to the file
        result_file.write(f"Results for model {backbone_name} - {model_name}:\n")
        for k in top_k_values:
            result_file.write(
                f"Top-{k} Precision: {avg_top_k_precision[k]} \t Recall: {avg_top_k_recall[k]} \t Accuracy: {avg_top_k_accuracy[k]}\n"
            )
        result_file.write("\n")
        result_file.flush()  # Ensure data is written to the file after each model

print(f"Results saved to {result_file_path}")
