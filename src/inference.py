import torch
import torchvision.utils as vutils
from src.generator import Generator
from src.config import load_config
from src.utils import get_device

class InferenceEngine:
    def __init__(self, checkpoint_path, config_path="configs/trainconfig.yaml"):
        self.device = get_device()
        self.config = load_config(config_path)
        
        self.z_dim = self.config['model']['z_dim']
        self.netG = Generator(
            z_dim=self.z_dim,
            feature_maps=self.config['model']['feature_maps_g']
        ).to(self.device)
        
        # Load weights
        if torch.cuda.is_available():
            state_dict = torch.load(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            
        self.netG.load_state_dict(state_dict)
        self.netG.eval()

    def generate(self, num_images):
        noise = torch.randn(num_images, self.z_dim, 1, 1, device=self.device)
        with torch.no_grad():
            fake_images = self.netG(noise)
        # Denormalize
        fake_images = (fake_images + 1) / 2.0
        return fake_images.cpu()
