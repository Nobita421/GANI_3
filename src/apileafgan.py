from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference import InferenceEngine
from src.monitoring import monitor
import base64
from io import BytesIO
import torch

app = FastAPI(title="CropLeafGAN API", version="1.0")

# Global engine (lazy load)
engine = None
CHECKPOINT_PATH = "checkpoints/G_epoch_1.pth" # Default to epoch 1 for demo or latest

class GenerateRequest(BaseModel):
    crop: str
    disease: str
    count: int

@app.on_event("startup")
def load_model():
    global engine
    # In a real app, we might scan for the latest checkpoint
    import os
    if os.path.exists(CHECKPOINT_PATH):
        engine = InferenceEngine(CHECKPOINT_PATH)
        print(f"Model loaded from {CHECKPOINT_PATH}")
    else:
        print(f"Warning: Checkpoint {CHECKPOINT_PATH} not found. API might fail.")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": engine is not None}

@app.get("/info")
def info():
    return {
        "model": "DCGAN",
        "version": "v1.0",
        "supported_crops": ["tomato", "potato", "corn"], # metadata
        "supported_diseases": ["blight", "rust", "healthy"]
    }

@app.post("/generate")
def generate_images(req: GenerateRequest):
    if not engine:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    # Log usage
    monitor.log_request(req.crop, req.disease, req.count)
    
    # Generate
    try:
        images = engine.generate(req.count) # Tensor list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Convert to base64
    results = []
    from torchvision.transforms import ToPILImage
    to_pil = ToPILImage()
    
    for img in images:
        pil_img = to_pil(img)
        buff = BytesIO()
        pil_img.save(buff, format="PNG")
        img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
        results.append(img_str)
        
    return {"images": results}
