import torch
from core.registry import get_model
import models  # This triggers the __init__.py which registers dummy_model

def main():
    print("Testing the Deepfake Framework Plugin System...\n")
    
    # 1. We define our config (in reality, this would come from a YAML file)
    config = {
        "in_channels": 3,
        "hidden_dim": 16,
        "num_classes": 1
    }
    
    # 2. We dynamically load the model using the registry by its string name!
    model_name = "dummy_cnn"
    print(f"Loading '{model_name}' from registry...")
    
    try:
        detector = get_model(model_name, config)
        print(f"Success! Instantiated {type(detector).__name__}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 3. Test a dummy forward pass
    print("\nTesting forward pass...")
    # Create a fake batch of 2 images: (Batch=2, Channels=3, Height=256, Width=256)
    dummy_input = torch.randn(2, 3, 256, 256)
    
    with torch.no_grad():
        output = detector(dummy_input)
        
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape} (Batch size, Num classes)")
    print(f"Output logits:\n{output}")
    
    # 4. Test explainability hook
    print("\nTesting Explainability Hook...")
    target_layer = detector.get_target_layer()
    print(f"Target layer for Grad-CAM extracted successfully: {target_layer}")
    
    print("\nAll systems go! The framework core is functional.")

if __name__ == "__main__":
    main()
