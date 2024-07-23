import argparse
import json
import os
from PIL import Image
from tqdm import tqdm
import torch
import open_clip
from sentence_transformers import util

from retrieval_with_CLIP import extract_class_from_filename, infer_image_class, compute_metrics

#fmt: off
# Argument parsing for command line execution
parser = argparse.ArgumentParser(description="Test a classification model with configurable hyperparameters.")
parser.add_argument("--backbone", type=str, default="ViT-H-14-378-quickgelu", help="Backbone")
parser.add_argument("--pretrain_data", type=str, default="dfn5b", help="Pre-training data")

parser.add_argument("--image_folder_path", type=str, default="/data/jwsuh/construction/sampled_test/", help="Image folder path")
parser.add_argument("--rule_path", type=str, default="data/rule/aihub_rules_pair_en.json", help="Rule file path")

parser.add_argument("--device", type=str, default="cuda:4", help="Device to use for inference")
args = parser.parse_args()



# Set device to GPU if available
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# Load the model and processor
backbone_name = args.backbone
pretrain_data = args.pretrain_data
model, _, processor = open_clip.create_model_and_transforms(
    backbone_name, pretrained=pretrain_data
)
model.to(device)
model.eval()
tokenizer = open_clip.get_tokenizer(backbone_name)
       
       
            
# Load rule data from JSON file
with open(args.rule_path, "r", encoding="UTF-8") as j:
    aihub_rule = json.load(j)

# Prepare rules and their corresponding descriptions
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
        if cnt != answer_index:
            print(f"Error in rule indexing: {cnt} != {answer_index}")
            break
        cnt += 1

scenario_list = []  # size is 101
for accident_type in aihub_rule.keys():
    for operation_type in aihub_rule[accident_type].keys():
        for scenario_type in aihub_rule[accident_type][operation_type].keys():
            if "Y" in scenario_type:
                scenario_description = (
                    f"the normal scenario in image is {aihub_rule[accident_type][operation_type][scenario_type].lower()}."
                )
                scenario_list.append(scenario_description)
            else:
                scenario_description = (
                    f"the abnormal scenario in image is {aihub_rule[accident_type][operation_type][scenario_type].lower()}."
                )
                scenario_list.append(scenario_description)            
scenario_list.append("the scenario in image is none")             

# Encode text features once
with torch.no_grad():
    text_tokens = tokenizer(rules_list).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    text_tokens = tokenizer(scenario_list).to(device)
    scenario_features = model.encode_text(text_tokens)
    scenario_features = scenario_features / scenario_features.norm(dim=-1, keepdim=True)
    
    

for filename in tqdm(os.listdir(args.image_folder_path)):
    if filename.endswith(".jpg"):
        image_path = os.path.join(args.image_folder_path, filename)
        predicted_classes = infer_image_class(
            image_path, model, processor, text_features, k=5
        )
        input_scenario_index = [i*2 for i in predicted_classes] + [i*2+1 for i in predicted_classes] + [100]
        

