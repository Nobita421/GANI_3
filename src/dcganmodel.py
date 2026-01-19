import torch
from .generator import Generator
from .discriminator import Discriminator

class DCGANModel:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        self.netG = Generator(
            z_dim=config['model']['z_dim'],
            feature_maps=config['model']['feature_maps_g'],
            channels=3 # Assuming RGB
        ).to(device)
        
        self.netD = Discriminator(
            channels=3,
            feature_maps=config['model']['feature_maps_d']
        ).to(device)
        
    def get_models(self):
        return self.netG, self.netD
