from typing import Callable, Dict, Type
from .base_model import BaseDeepfakeDetector

# The global registry mapping string names to model classes
MODEL_REGISTRY: Dict[str, Type[BaseDeepfakeDetector]] = {}

def register_model(name: str) -> Callable:
    """
    A decorator used to register a new deepfake detection model class.
    
    Args:
        name (str): The unique string name to register the model under.
        
    Example:
        @register_model("efficientnet_b4")
        class EfficientNetB4Detector(BaseDeepfakeDetector):
            ...
    """
    def decorator(cls: Type[BaseDeepfakeDetector]) -> Type[BaseDeepfakeDetector]:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Cannot register duplicate model name: '{name}'")
        if not issubclass(cls, BaseDeepfakeDetector):
            raise TypeError(f"Model '{name}' must inherit from BaseDeepfakeDetector")
            
        MODEL_REGISTRY[name] = cls
        return cls
        
    return decorator

def get_model(name: str, config: dict) -> BaseDeepfakeDetector:
    """
    Factory function to instantiate a model by its registered name.
    
    Args:
        name (str): The registered name of the model.
        config (dict): A dictionary of hyperparameters to pass to the model.
        
    Returns:
        BaseDeepfakeDetector: The instantiated model.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Model '{name}' not found in registry. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )
        
    model_cls = MODEL_REGISTRY[name]
    return model_cls(config)
