import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DeepfakeImageDataset(Dataset):
    """
    A standard PyTorch Dataset for loading images from a directory structure:
    root_dir/
      ├── real/
      │   ├── image1.jpg
      │   └── ...
      └── fake/
          ├── image2.jpg
          └── ...
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Determine paths and labels (0 for real, 1 for fake)
        real_dir = os.path.join(root_dir, 'real')
        fake_dir = os.path.join(root_dir, 'fake')
        
        if os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                self.image_paths.append(os.path.join(real_dir, img_name))
                self.labels.append(0.0) # Real
                
        if os.path.exists(fake_dir):
            for img_name in os.listdir(fake_dir):
                self.image_paths.append(os.path.join(fake_dir, img_name))
                self.labels.append(1.0) # Fake
                
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_default_transforms(image_size=256):
    """
    Returns standard torchvision transforms for image preprocessing.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # Standard ImageNet mean/std
                             std=[0.229, 0.224, 0.225])
    ])
