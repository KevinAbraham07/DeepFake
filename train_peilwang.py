import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from core.registry import get_model
import models
from data.txt_dataset import TxtLabelDataset
from data.dataset import get_default_transforms

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for inputs, labels in progress_bar:
        inputs = inputs.to(device)
        labels = labels.view(-1, 1).float().to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix({'loss': running_loss / (progress_bar.n + 1)})

def main():
    parser = argparse.ArgumentParser(description="Train Deepfake Detector on peilwang/deepfake")
    parser.add_argument('--dataset_path', type=str, default='/kaggle/input/datasets/peilwang/deepfake/phase1', help='Path to phase1 directory containing trainset and trainset_label.txt')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint .pth file to resume from')
    parser.add_argument('--start_epoch', type=int, default=1, help='Epoch number to resume from')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    config = load_config("configs/train_config_peilwang.yaml")

    # Locate images dir and label file
    phase1_dir = args.dataset_path
    images_dir = os.path.join(phase1_dir, "trainset")
    label_file = os.path.join(phase1_dir, "trainset_label.txt")

    if not os.path.exists(label_file):
        # Fallback search if nested
        for root, dirs, files in os.walk(phase1_dir):
            if "trainset_label.txt" in files:
                label_file = os.path.join(root, "trainset_label.txt")
                images_dir = os.path.join(root, "trainset")
                break

    print(f"Loading images from: {images_dir}")
    print(f"Loading labels from: {label_file}")

    dataset = TxtLabelDataset(
        images_dir=images_dir,
        label_file=label_file,
        transform=get_default_transforms()
    )

    if len(dataset) == 0:
        raise ValueError(f"No images loaded from {label_file}! Please check directory paths.")

    print(f"Successfully loaded {len(dataset)} training images!")

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0
    )

    model_name = config["model"]["name"]
    model = get_model(model_name, config["model"]).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

    epochs = config["training"]["epochs"]
    save_dir = config["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nStarting Training for peilwang/deepfake from Epoch {args.start_epoch}...")
    for epoch in range(args.start_epoch, epochs + 1):
        train_one_epoch(model, dataloader, criterion, optimizer, device, epoch)
        checkpoint_path = os.path.join(save_dir, f"peilwang_detector_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    main()
