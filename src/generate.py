import argparse
import os
import torchvision.utils as vutils
from src.inference import InferenceEngine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop", type=str, default="tomato", help="Crop name") # Placeholder for conditioning
    parser.add_argument("--disease", type=str, default="blight", help="Disease name") # Placeholder
    parser.add_argument("--n", type=int, default=10, help="Number of images")
    parser.add_argument("--out", type=str, default="output", help="Output directory")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator checkpoint")
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    engine = InferenceEngine(args.checkpoint)
    images = engine.generate(args.n)
    
    for i, img in enumerate(images):
        vutils.save_image(img, os.path.join(args.out, f"{args.crop}_{args.disease}_{i}.png"))
        
    print(f"Generated {args.n} images in {args.out}")

if __name__ == "__main__":
    main()
