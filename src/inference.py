import os
import glob
import torch
from src.generator import Generator
from src.config import load_config
from src.utils import get_device

def _latest_checkpoint(checkpoints_dir: str) -> str:
    pattern = os.path.join(checkpoints_dir, "G_epoch_*.pth")
    candidates = glob.glob(pattern)
    if not candidates:
        return ""
    candidates.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[-1]))
    return candidates[-1]


def resolve_checkpoint(config, crop=None, disease=None) -> str:
    registry = config.get("model_registry", {})
    if crop and disease:
        key = f"{crop.lower()}::{disease.lower()}"
        if key in registry:
            return registry[key]
    default_ckpt = registry.get("default")
    if default_ckpt and os.path.exists(default_ckpt):
        return default_ckpt

    checkpoints_dir = config.get("training", {}).get("checkpoints_dir", "checkpoints")
    return _latest_checkpoint(checkpoints_dir)


class InferenceEngine:
    def __init__(self, checkpoint_path=None, config_path="configs/trainconfig.yaml", crop=None, disease=None):
        self.device = get_device()
        self.config = load_config(config_path)

        if checkpoint_path is None:
            checkpoint_path = resolve_checkpoint(self.config, crop=crop, disease=disease)
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError("No generator checkpoint found for inference.")

        self.z_dim = self.config['model']['z_dim']
        self.netG = Generator(
            z_dim=self.z_dim,
            feature_maps=self.config['model']['feature_maps_g']
        ).to(self.device)

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.netG.load_state_dict(state_dict)
        self.netG.eval()

    def generate(self, num_images):
        noise = torch.randn(num_images, self.z_dim, 1, 1, device=self.device)
        with torch.no_grad():
            fake_images = self.netG(noise)
        fake_images = (fake_images + 1) / 2.0
        return fake_images.cpu()


def load_model(config_path="configs/trainconfig.yaml", crop=None, disease=None, checkpoint_path=None):
    return InferenceEngine(checkpoint_path=checkpoint_path, config_path=config_path, crop=crop, disease=disease)


def generate_images(crop, disease, n, config_path="configs/trainconfig.yaml", checkpoint_path=None):
    engine = load_model(config_path=config_path, crop=crop, disease=disease, checkpoint_path=checkpoint_path)
    return engine.generate(n)
