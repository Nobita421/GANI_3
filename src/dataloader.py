import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any, Iterable, Optional

def get_transforms(image_size: int, channels: int = 3):
    """
    Get the data transformations.
    Resizes to image_size, CenterCrops to square, converts to Tensor, 
    and normalizes to [-1, 1] for DCGAN.
    """
    transform_list = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * channels, (0.5,) * channels)
    ]
    return transforms.Compose(transform_list)

def _looks_like_class_root(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    # Heuristic: at least one subdirectory with image files
    for entry in os.listdir(path):
        candidate = os.path.join(path, entry)
        if os.path.isdir(candidate):
            for img_name in os.listdir(candidate):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    return True
    return False

def _candidate_split_names(split: str) -> Iterable[str]:
    split = split.strip().lower()
    if split in {'val', 'valid', 'validation'}:
        return ['val', 'valid', 'validation', 'Val', 'Valid', 'Validation']
    if split in {'test', 'testing'}:
        return ['test', 'testing', 'Test', 'Testing']
    if split in {'train', 'training'}:
        return ['train', 'training', 'Train', 'Training']
    return [split, split.capitalize(), split.upper()]

def _resolve_split_path(root_path: str, split: str) -> Optional[str]:
    if not os.path.exists(root_path):
        return None

    # If root path already points to class subfolders
    if _looks_like_class_root(root_path):
        return root_path

    root_candidates = [root_path]
    if os.path.isdir(os.path.join(root_path, 'Real')):
        root_candidates.append(os.path.join(root_path, 'Real'))
    if os.path.isdir(os.path.join(root_path, 'real')):
        root_candidates.append(os.path.join(root_path, 'real'))

    for root in root_candidates:
        for split_name in _candidate_split_names(split):
            candidate = os.path.join(root, split_name)
            if _looks_like_class_root(candidate):
                return candidate
    return None

def get_dataloader(config: Dict[str, Any], split: str = 'train') -> DataLoader:
    """
    Creates and returns a PyTorch DataLoader based on the configuration.
    
    Args:
        config: Configuration dictionary loaded from YAML.
        split: 'train' or 'val' or 'test'. Assumes folder structure Data/Real/{split}/...
    """
    data_cfg = config['dataset']
    root_path = data_cfg['path']
    split_path = _resolve_split_path(root_path, split)
    if not split_path:
        raise FileNotFoundError(
            f"Dataset path not found for split '{split}'. Checked under: {root_path}"
        )

    transform = get_transforms(data_cfg['image_size'], data_cfg['channels'])

    dataset = datasets.ImageFolder(root=split_path, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=data_cfg['batch_size'],
        shuffle=(split == 'train' and data_cfg.get('shuffle', True)),
        num_workers=data_cfg.get('num_workers', 2),
        pin_memory=True
    )
    
    return dataloader

if __name__ == "__main__":
    # Smoke test for dataloader
    import sys
    sys.path.append(os.getcwd()) # Ensure src can be imported if running directly
    from src.config import load_config
    
    # Create a dummy config if file doesn't exist just to test syntax
    cfg = {
        'dataset': {
            'path': 'Data/Real',
            'image_size': 64,
            'channels': 3,
            'batch_size': 4,
            'num_workers': 0,
            'shuffle': False
        }
    }
    
    # Print info
    print("Dataloader module loaded successfully.")
