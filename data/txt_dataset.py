import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from data.dataset import get_default_transforms

class TxtLabelDataset(Dataset):
    """
    Dataset loader for text label files like peilwang/deepfake (phase1/trainset_label.txt).
    Optimized for high performance on cloud filesystems (Kaggle/Colab).
    """
    def __init__(self, images_dir, label_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform or get_default_transforms()
        self.image_paths = []
        self.labels = []

        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file not found at: {label_file}")

        print("Indexing directory files for fast lookup...")
        existing_files = set(os.listdir(images_dir))
        print(f"Found {len(existing_files)} physical files in {images_dir}.")

        print("Parsing label file...")
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.replace(',', ' ').split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    try:
                        label = float(parts[1])
                    except ValueError:
                        continue
                    
                    if img_name in existing_files:
                        full_path = os.path.join(images_dir, img_name)
                        self.image_paths.append(full_path)
                        self.labels.append(label)

        print(f"Successfully matched {len(self.image_paths)} valid images with labels!")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
