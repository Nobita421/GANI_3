from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference import load_model
from src.monitoring import monitor
import base64
from io import BytesIO
import os

app = FastAPI(title="CropLeafGAN API", version="1.0")

# Global engine (lazy load)
engine = None
CONFIG_PATH = "configs/trainconfig.yaml"

class GenerateRequest(BaseModel):
    crop: str
    disease: str
    count: int

@app.on_event("startup")
def load_model_on_startup():
    global engine
    try:
        engine = load_model(config_path=CONFIG_PATH)
        print("Model loaded.")
    except Exception as e:
        print(f"Warning: Model not loaded. {e}")

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
        images = engine.generate(req.count)
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
