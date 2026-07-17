import time
import io
import yaml
from PIL import Image
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from core.registry import get_model
from data.dataset import get_default_transforms
from api.schemas import PredictionResponse

# 1. Initialize the FastAPI app
app = FastAPI(
    title="Deepfake Detection Framework API",
    description="REST API for evaluating images against deepfake models.",
    version="1.0.0"
)

# Allow Cross-Origin requests so the React frontend can talk to it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to hold our loaded model
# We load it ONCE during startup, not during every request
detector = None
device = None
transforms = None

@app.on_event("startup")
def load_framework():
    """
    Runs once when the server starts. Loads the model into memory.
    """
    global detector, device, transforms
    
    print("Loading Framework Configuration...")
    with open("configs/train_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model onto {device}...")
    
    # Load the dummy model from the registry
    model_name = config["model"]["name"]
    detector = get_model(model_name, config["model"])
    
    # Load saved weights if they exist (For now, we just use the initialized weights)
    # detector.load_state_dict(torch.load("checkpoints/best_model.pth"))
    
    detector = detector.to(device)
    detector.eval() # Set model to evaluation mode (turns off dropout, etc.)
    
    transforms = get_default_transforms()
    print("Framework API is ready to receive requests!")

@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Endpoint that accepts an image file and returns a deepfake prediction.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
        
    start_time = time.time()
    
    try:
        # Read the uploaded file directly into memory
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess the image
        input_tensor = transforms(image).unsqueeze(0).to(device) # Add batch dimension
        
        # Run inference
        with torch.no_grad():
            logits = detector(input_tensor)
            
            # Convert logits to probability using Sigmoid
            probability = torch.sigmoid(logits).item()
            
        # Format the response
        prediction = "Fake" if probability >= 0.5 else "Real"
        confidence = probability if prediction == "Fake" else 1.0 - probability
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            filename=file.filename,
            prediction=prediction,
            confidence=round(confidence, 4),
            processing_time_ms=round(processing_time_ms, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
