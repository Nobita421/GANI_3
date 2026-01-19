
import sys
import os
import torch

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from src.inference import InferenceEngine
    print("Attempting to load model...")
    # Point to the default checkpoint path used in appleafgan.py
    checkpoint_path = "checkpoints/G_epoch_1.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Skipping load test.")
        sys.exit(0)
        
    engine = InferenceEngine(checkpoint_path)
    print("Model loaded successfully!")
    
    # Try a forward pass
    img = engine.generate(1)
    print(f"Generated image shape: {img.shape}")
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
