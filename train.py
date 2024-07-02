import argparse
import json
import torch
import numpy as np
import random

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_scheduler,
)
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import pickle
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import Dropout
# python train.py --dataset_identifier neg40 --train_data data/data-train-neg40-dragon-roberta.pkl --resume_from_checkpoint models/cross-encoder/stsb-roberta-large_neg40_20240627_181927/model_epoch_2_step_270000.bin --device cuda:7
# Argument parser
parser = argparse.ArgumentParser(description='Train a sequence classification model with configurable hyperparameters.')
parser.add_argument('--model_identifier', type=str, default="cross-encoder/stsb-roberta-large", help='Identifier for the model')
parser.add_argument('--dataset_identifier', type=str, default=None, help='Identifier for the dataset')
parser.add_argument('--train_data', type=str, default=None, help="Train data path")
parser.add_argument('--valid_data', type=str, default="data/data-val.pkl", help="Valid data path")

parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and validation')
parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate for the optimizer')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for the optimizer')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--drop', type=float, default=0.3, help='Proportion of drop')
parser.add_argument('--warmup_proportion', type=float, default=0.06, help='Proportion of training steps to perform learning rate warmup')

parser.add_argument('--device', type=str, default=None, help='Device to use for training')
parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to a checkpoint to resume training from')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

args = parser.parse_args()

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)

# Load data from pickle files
def load_data(train_path, val_path):
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(val_path, 'rb') as f:
        val_data = pickle.load(f)
    return (train_data['sentence1'], train_data['sentence2'], train_data['labels'],
            val_data['sentence1'], val_data['sentence2'], val_data['labels'])

train_path = args.train_data
val_path = args.valid_data  # Ensure the validation file path is correct
sentences1_train, sentences2_train, labels_train, sentences1_val, sentences2_val, labels_val = load_data(train_path, val_path)

# Dynamic settings
args.pos_weight = int(args.dataset_identifier.replace("neg", ""))  # Set pos_weight to the number of negative samples

# Unique identifier for this training run
run_id = f"{args.model_identifier}_{args.dataset_identifier}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(args.model_identifier)
model = AutoModelForSequenceClassification.from_pretrained(args.model_identifier)

# Add Dropout layer
if args.drop:
    model.classifier.dropout = Dropout(p=args.drop)

# Sample Dataset Class
class CustomDataset(Dataset):
    def __init__(self, sentences1, sentences2, labels, tokenizer):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences1)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.sentences1[idx], self.sentences2[idx], 
            truncation=True, padding='max_length', max_length=512, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(), 
            'attention_mask': encoding['attention_mask'].squeeze(), 
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)  # Use float for BCEWithLogitsLoss
        }

train_dataset = CustomDataset(sentences1_train, sentences2_train, labels_train, tokenizer)
val_dataset = CustomDataset(sentences1_val, sentences2_val, labels_val, tokenizer)

generator = torch.Generator().manual_seed(args.seed)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

# Training loop with periodic evaluation and model saving
device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Define pos_weight for binary classification with class imbalance
pos_weight = torch.tensor([args.pos_weight], device=device)  # Set pos_weight to handle imbalance
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Optimizer settings
optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

# Scheduler settings
num_training_steps = args.num_epochs * len(train_dataloader)
num_warmup_steps = int(num_training_steps * args.warmup_proportion)

lr_scheduler = get_scheduler(
    "cosine", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)

# Create directories for saving logs and models
log_dir = f'logs/{run_id}'
model_dir = f'models/{run_id}'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# TensorBoard writer
writer = SummaryWriter(log_dir)

# Save hyperparameters to log
with open(os.path.join(log_dir, 'hyperparameters.txt'), 'w') as f:
    json.dump(vars(args), f, indent=4)

# Metrics storage
training_loss = []
training_accuracy = []
validation_loss = []
validation_accuracy = []

best_val_accuracy = 0.0  # Track the best validation accuracy
best_model_path = None   # Track the path to the best model
start_epoch = 0
start_step = 0  # Track the starting step
step = 0  # Track the current step

# Load from checkpoint if specified
if args.resume_from_checkpoint:
    checkpoint = torch.load(args.resume_from_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # start_epoch = checkpoint['epoch']
    start_step = checkpoint['step']
    best_val_accuracy = checkpoint['best_val_accuracy']
    print(f"Resumed training from step {checkpoint['step'] }")

def save_checkpoint(model, optimizer, scheduler, epoch, step, best_val_accuracy, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'step': step,
        'best_val_accuracy': best_val_accuracy,
        'args': vars(args)
    }
    torch.save(checkpoint, path)

def evaluate(model, val_dataloader, criterion, device):
    model.eval()
    val_epoch_loss = 0
    all_val_preds = []
    all_val_labels = []
    with torch.no_grad():
        for val_batch in tqdm(val_dataloader, desc="Validation"):
            val_batch = {k: v.to(device) for k, v in val_batch.items()}
            outputs = model(**val_batch)
            val_logits = outputs.logits.squeeze()  # Squeeze to make 1-dimensional
            val_loss = criterion(val_logits, val_batch['labels'])
            val_epoch_loss += val_loss.item()
            val_preds = torch.round(torch.sigmoid(val_logits))  # Convert logits to binary predictions
            all_val_preds.extend(val_preds.cpu().numpy())
            all_val_labels.extend(val_batch['labels'].cpu().numpy())

    avg_val_epoch_loss = val_epoch_loss / len(val_dataloader)
    avg_val_epoch_accuracy = accuracy_score(all_val_labels, all_val_preds)
    return avg_val_epoch_loss, avg_val_epoch_accuracy

for epoch in range(start_epoch, args.num_epochs):
    train_log_file = open(f'{log_dir}/train_log_epoch_{epoch+1}.txt', 'w')
    val_log_file = open(f'{log_dir}/val_log_epoch_{epoch+1}.txt', 'w')

    model.train()
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
    for batch in progress_bar:
        step += 1
        if step < start_step:
            continue
        elif step == start_step:
            avg_val_epoch_loss, avg_val_epoch_accuracy = evaluate(model, val_dataloader, criterion, device)
            validation_loss.append(avg_val_epoch_loss)
            validation_accuracy.append(avg_val_epoch_accuracy)
            
            # Logging to text file
            val_log_file.write(f"Epoch {epoch+1}, Step {step} - Val Loss: {avg_val_epoch_loss:.4f}, Val Accuracy: {avg_val_epoch_accuracy:.4f}\n")
            
            # Logging to TensorBoard
            writer.add_scalar(f'Validation/Loss', avg_val_epoch_loss, step)
            writer.add_scalar(f'Validation/Accuracy', avg_val_epoch_accuracy, step)

            print(f"Epoch {epoch+1} Step {step} - Validation Loss: {avg_val_epoch_loss:.4f}, Validation Accuracy: {avg_val_epoch_accuracy:.4f}")

            # Save model
            model_save_path = f'{model_dir}/model_epoch_{epoch+1}.bin'
            save_checkpoint(model, optimizer, lr_scheduler, epoch, step, best_val_accuracy, model_save_path)
            print(f"Model saved to {model_save_path}")

            # Save the best model
            if avg_val_epoch_accuracy > best_val_accuracy:
                best_val_accuracy = avg_val_epoch_accuracy
                best_model_path = f'{model_dir}/best_model.bin'
                save_checkpoint(model, optimizer, lr_scheduler, epoch, step, best_val_accuracy, best_model_path)
                print(f"New best model saved to {best_model_path} with accuracy {best_val_accuracy:.4f}")
            continue
                
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits.squeeze()  # Squeeze to make 1-dimensional
        loss = criterion(logits, batch['labels'])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        preds = torch.round(torch.sigmoid(logits))  # Convert logits to binary predictions
        accuracy = accuracy_score(batch['labels'].cpu().numpy(), preds.detach().cpu().numpy())
        
        training_loss.append(loss.item())
        training_accuracy.append(accuracy)
        
        # Logging to text file
        train_log_file.write(f"Epoch {epoch+1}, Step {step} - Train Loss: {loss.item():.6f}, Train Accuracy: {accuracy:.6f}\n")
        
        # Logging to TensorBoard
        writer.add_scalar(f'Train/Loss', loss.item(), step)
        writer.add_scalar(f'Train/Accuracy', accuracy, step)

        # Update progress bar
        progress_bar.set_postfix({"Train Loss": loss.item(), "Train Accuracy": accuracy})

        # Periodic evaluation and model saving
        if step % 10000 == 0:
            avg_val_epoch_loss, avg_val_epoch_accuracy = evaluate(model, val_dataloader, criterion, device)
            validation_loss.append(avg_val_epoch_loss)
            validation_accuracy.append(avg_val_epoch_accuracy)
            
            # Logging to text file
            val_log_file.write(f"Epoch {epoch+1}, Step {step} - Val Loss: {avg_val_epoch_loss:.4f}, Val Accuracy: {avg_val_epoch_accuracy:.4f}\n")
            
            # Logging to TensorBoard
            writer.add_scalar(f'Validation/Loss', avg_val_epoch_loss, step)
            writer.add_scalar(f'Validation/Accuracy', avg_val_epoch_accuracy, step)

            print(f"Epoch {epoch+1} Step {step} - Validation Loss: {avg_val_epoch_loss:.4f}, Validation Accuracy: {avg_val_epoch_accuracy:.4f}")

            # Save model
            model_save_path = f'{model_dir}/model_epoch_{epoch+1}.bin'
            save_checkpoint(model, optimizer, lr_scheduler, epoch, step, best_val_accuracy, model_save_path)
            print(f"Model saved to {model_save_path}")

            # Save the best model
            if avg_val_epoch_accuracy > best_val_accuracy:
                best_val_accuracy = avg_val_epoch_accuracy
                best_model_path = f'{model_dir}/best_model.bin'
                save_checkpoint(model, optimizer, lr_scheduler, epoch, step, best_val_accuracy, best_model_path)
                print(f"New best model saved to {best_model_path} with accuracy {best_val_accuracy:.4f}")


    train_log_file.close()
    val_log_file.close()

    # Save training and validation logs for each epoch
    with open(f'{log_dir}/epoch_{epoch+1}_training_log.json', 'w') as f:
        json.dump({
            'training_loss': training_loss,
            'training_accuracy': training_accuracy,
            'validation_loss': validation_loss,
            'validation_accuracy': validation_accuracy
        }, f, indent=4)


# Close the TensorBoard writer
writer.close()

print(f"Best model saved to {best_model_path} with accuracy {best_val_accuracy:.4f}")
