import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseDeepfakeDetector(nn.Module, ABC):
    """
    Abstract Base Class for all Deepfake Detection models in the framework.
    Every custom model must inherit from this class to ensure compatibility
    with the training, benchmarking, and explainability pipelines.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x):
        """
        Standard forward pass for the model.
        
        Args:
            x (torch.Tensor): The input image or video tensor.
                - Image shape: (Batch, Channels, Height, Width)
                - Video shape: (Batch, Frames, Channels, Height, Width)
                
        Returns:
            torch.Tensor: Raw logits before sigmoid/softmax.
        """
        pass

    @abstractmethod
    def get_target_layer(self):
        """
        Returns the specific spatial layer used for Explainability (e.g., Grad-CAM).
        
        For CNNs, this is typically the final convolutional layer.
        For Transformers, this might be a specific attention block.
        
        Returns:
            nn.Module: The target layer for feature extraction.
        """
        pass
