import torch
import torch.nn as nn
from core.registry import register_model
from core.base_model import BaseDeepfakeDetector

@register_model("dummy_cnn")
class DummyCNN(BaseDeepfakeDetector):
    """
    A minimal convolutional neural network for testing the framework pipeline.
    """
    def __init__(self, config):
        super().__init__(config)
        
        # Example of reading from the config dictionary
        in_channels = config.get("in_channels", 3)
        hidden_dim = config.get("hidden_dim", 16)
        num_classes = config.get("num_classes", 1)  # 1 for binary classification (fake vs real)
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Flattens spatial dimensions to 1x1
        )
        
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        """
        Forward pass.
        Expects image of shape (Batch, Channels, H, W).
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits
        
    def get_target_layer(self):
        """
        Return the final convolutional layer for Grad-CAM.
        """
        # The 4th item in self.features is the second Conv2d layer
        return self.features[3]
