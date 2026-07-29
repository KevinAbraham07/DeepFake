import time
import io
import os
import yaml
from PIL import Image
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from core.registry import get_model
import models  # <--- Triggers model registry!
from data.dataset import get_default_transforms
from api.schemas import PredictionResponse, ModelBreakdownItem

app = FastAPI(
    title="Deepfake Detection Framework API",
    description="REST API for evaluating images against deepfake models.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models_loaded = {}
device = None
transforms = None

@app.on_event("startup")
def load_framework():
    global models_loaded, device, transforms
    
    print("Loading Framework Configuration...")
    with open("configs/train_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading models onto {device}...")
    
    # Model 1: ResNet-50 trained on 140k StyleGAN dataset
    model1_path = "checkpoints/resnet_detector_epoch_5.pth"
    if os.path.exists(model1_path):
        print(f"Loading Model 1 (StyleGAN Detector) from {model1_path}...")
        m1 = get_model("resnet_detector", config["model"])
        m1.load_state_dict(torch.load(model1_path, map_location=device))
        m1.to(device).eval()
        models_loaded["StyleGAN Detector (140k Dataset)"] = m1
    else:
        print("Warning: Model 1 checkpoint not found!")

    # Model 2: Detector trained on peilwang/deepfake dataset
    model2_path = "checkpoints/peilwang_detector_epoch_5.pth"
    if os.path.exists(model2_path):
        print(f"Loading Model 2 (General Deepfake Detector) from {model2_path}...")
        m2 = get_model("resnet_detector", config["model"])
        m2.load_state_dict(torch.load(model2_path, map_location=device))
        m2.to(device).eval()
        models_loaded["General Deepfake Detector (peilwang Dataset)"] = m2
    else:
        print("Model 2 checkpoint not found yet (will use Model 1 + Ensemble pipeline).")
        
    transforms = get_default_transforms()
    print("Multi-Model Framework API is ready!")

@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
        
    start_time = time.time()
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        input_tensor = transforms(image).unsqueeze(0).to(device)
        
        breakdown = []
        probabilities = []
        
        with torch.no_grad():
            for name, model in models_loaded.items():
                logits = model(input_tensor)
                prob = torch.sigmoid(logits).item()
                probabilities.append(prob)
                
                is_fake_m = prob >= 0.5
                conf_m = prob if is_fake_m else 1.0 - prob
                
                breakdown.append(
                    ModelBreakdownItem(
                        model_name=name,
                        is_fake=is_fake_m,
                        confidence=round(conf_m, 4)
                    )
                )

        # Fallback if no models are loaded
        if not probabilities:
            combined_prob = 0.5
            breakdown.append(
                ModelBreakdownItem(
                    model_name="Fallback Evaluator",
                    is_fake=False,
                    confidence=0.5
                )
            )
        else:
            combined_prob = sum(probabilities) / len(probabilities)

        combined_is_fake = combined_prob >= 0.5
        combined_confidence = combined_prob if combined_is_fake else 1.0 - combined_prob
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            filename=file.filename,
            is_fake=combined_is_fake,
            confidence=round(combined_confidence, 4),
            processing_time=round(processing_time, 4),
            model_name="Ensemble Engine (Multi-Model Combined)",
            model_breakdown=breakdown
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
