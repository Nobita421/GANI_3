import os
import numpy as np
from PIL import Image

def create_dummy_data():
    base_dir = "Data/Real"
    classes = ["healthy", "blight", "rust"]
    splits = ["train", "val", "test"]

    print(f"Creating dummy data in {base_dir}...")

    for split in splits:
        for cls in classes:
            cls_dir = os.path.join(base_dir, split, cls)
            os.makedirs(cls_dir, exist_ok=True)

            for i in range(10):
                arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                img = Image.fromarray(arr)
                img.save(os.path.join(cls_dir, f"img_{i}.jpg"))

    print("Done.")

if __name__ == "__main__":
    create_dummy_data()
