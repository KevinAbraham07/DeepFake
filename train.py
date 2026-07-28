import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.registry import get_model
import models
from data.dataset import DeepfakeImageDataset, get_default_transforms

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Trains the model for one full pass over the dataset.
    """
    model.train() # Set the model to training mode
    running_loss = 0.0
    
    # tqdm gives us a nice progress bar in the terminal (or Kaggle output)
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for inputs, labels in progress_bar:
        # Move the large image tensors and labels to the GPU (if available)
        inputs = inputs.to(device)
        # Ensure labels are the correct shape (Batch_size, 1) and type
        labels = labels.view(-1, 1).float().to(device)
        
        # 1. Zero out the gradients from the previous batch
        optimizer.zero_grad()
        
        # 2. Forward pass: push the batch of images through the model
        outputs = model(inputs)
        
        # 3. Calculate how wrong the model was (Loss)
        loss = criterion(outputs, labels)
        
        # 4. Backward pass: compute the gradients (the math that tells the model how to fix itself)
        loss.backward()
        
        # 5. Optimizer step: tweak the model's weights slightly to be more accurate next time
        optimizer.step()
        
        # Update progress bar
        running_loss += loss.item()
        progress_bar.set_postfix({'loss': running_loss / (progress_bar.n + 1)})

import argparse

def main():
    parser = argparse.ArgumentParser(description="Train Deepfake Detector")
    parser.add_argument('--dataset_path', type=str, default='./dataset', help='Path to the dataset directory')
    args = parser.parse_args()
    
    # 1. Setup Device (This automatically uses Kaggle's GPU if running there!)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # 2. Load Configuration
    config = load_config("configs/train_config.yaml")
    
    # 3. Create Dataset and DataLoader
    # IMPORTANT: On Kaggle, you would change './dataset' to '/kaggle/input/dataset_name'
    dataset_path = args.dataset_path
    
    # We use a dummy dataset here to prevent crashes if the folder doesn't exist
    # In reality, you'd ensure the folder structure exists first.
    os.makedirs(os.path.join(dataset_path, "real"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "fake"), exist_ok=True)
    
    dataset = DeepfakeImageDataset(
        root_dir=dataset_path,
        transform=get_default_transforms()
    )
    
    # The DataLoader is the magic that handles massive datasets by chunking them into 'batches'
    dataloader = DataLoader(
        dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=True, 
        num_workers=0 # Set to 2 or 4 on Kaggle to speed up data loading
    )
    
    # 4. Instantiate Model using the Plugin Framework
    model_name = config["model"]["name"]
    model = get_model(model_name, config["model"])
    model = model.to(device) # Push the model itself to the GPU
    
    # 5. Define Loss Function and Optimizer
    # BCEWithLogitsLoss is perfect for binary classification (Real vs Fake)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    
    # 6. The Training Loop
    epochs = config["training"]["epochs"]
    save_dir = config["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    
    print("\nStarting Training...")
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, dataloader, criterion, optimizer, device, epoch)
        
        # Save a checkpoint after every epoch
        # If Kaggle disconnects, you won't lose your progress!
        checkpoint_path = os.path.join(save_dir, f"{model_name}_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    main()
