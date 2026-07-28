import torch.nn as nn
from core.registry import register_model
from core.base_model import BaseDeepfakeDetector

@register_model("resnet_detector")
class ResNetDetector(BaseDeepfakeDetector):
    def __init__(self, config):
        super().__init__()
        
        # We lazy-import torchvision here so it doesn't crash your local Windows API 
        # (due to the numpy/dynamo bug). On Kaggle (Linux), this will import perfectly!
        from torchvision import models
        
        # Load a pretrained ResNet-50 model
        # 'weights=models.ResNet50_Weights.DEFAULT' is the modern PyTorch way to load pretrained weights
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # We freeze the early layers so we don't destroy the pretrained ImageNet features.
        # This makes training much faster and prevents overfitting on small datasets.
        freeze_layers = config.get("freeze_backbone", True)
        if freeze_layers:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Unfreeze the very last layer block (layer4) for fine-tuning
        if freeze_layers:
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True

        # Replace the final classification head
        # ResNet-50 has 2048 input features to its fully connected (fc) layer
        num_ftrs = self.backbone.fc.in_features
        
        # We replace it with our own head that outputs 1 value (Real vs Fake)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=config.get("dropout_rate", 0.5)),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(p=config.get("dropout_rate", 0.3)),
            nn.Linear(512, 1) # 1 output node for binary classification
        )

    def forward(self, x):
        return self.backbone(x)

    def get_target_layer(self):
        # Used for Explainable AI (Grad-CAM) later on
        return self.backbone.layer4[-1]
