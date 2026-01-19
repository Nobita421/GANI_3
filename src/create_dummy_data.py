import os
import numpy as np
from PIL import Image

def create_dummy_data():
    base_dir = "Data/Real/train"
    classes = ["healthy", "blight", "rust"]
    
    print(f"Creating dummy data in {base_dir}...")
    
    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        
        for i in range(10): # 10 images per class
            # Create random RGB image
            arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            img.save(os.path.join(cls_dir, f"img_{i}.jpg"))
            
    print("Done.")

if __name__ == "__main__":
    create_dummy_data()
