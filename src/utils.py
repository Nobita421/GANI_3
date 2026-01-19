import torch
import random
import numpy as np
import os

def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_device():
    """Get the available device (CPU or CUDA)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
