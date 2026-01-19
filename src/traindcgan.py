import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
from tqdm import tqdm

from src.config import load_config, parse_args
from src.utils import set_seed, get_device
from src.logger import setup_logger
from src.dataloader import get_dataloader
from src.dcganmodel import DCGANModel

def train():
    args = parse_args()
    config_path = args.config if args.config else "configs/trainconfig.yaml"
    config = load_config(config_path)
    
    # Load data config as well for completeness (or merge them)
    # Ideally should be passed or loaded via include, but here we assume simple structure
    # For now, let's load dataconfig separately or assume user provides strict paths
    data_config = load_config("configs/dataconfig.yaml")
    config.update(data_config) # Simple merge

    set_seed(42)
    device = get_device()
    
    # Setup Logger
    logger, writer = setup_logger("logs")
    logger.info(f"Loaded config: {config}")

    # DataLoader
    dataloader = get_dataloader(config, split='train')
    
    # Model
    model_wrapper = DCGANModel(config, device)
    netG, netD = model_wrapper.get_models()
    
    # Optimizers
    lr = config['training']['lr']
    beta1 = config['training']['beta1']
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    
    criterion = nn.BCELoss()
    
    # Params
    num_epochs = config['training']['epochs']
    z_dim = config['model']['z_dim']
    fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)
    
    real_label = 1.0
    fake_label = 0.0
    
    # Stabilization
    label_smoothing = config['stabilization']['label_smoothing']
    input_noise = config['stabilization']['input_noise']
    
    logger.info("Starting Training Loop...")
    
    for epoch in range(num_epochs):
        for i, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ############################
            netD.zero_grad()
            
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            
            # Label Smoothing for real labels
            target_real = real_label
            if label_smoothing:
                target_real = 0.9
            
            label = torch.full((b_size,), target_real, dtype=torch.float, device=device)
            
            # Input Noise
            if input_noise and epoch < 10: # Add noise only in early epochs or decay it
                 real_cpu = real_cpu + 0.05 * torch.randn_like(real_cpu)

            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            noise = torch.randn(b_size, z_dim, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ############################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            # Logging
            if i % config['training']['log_interval_step'] == 0:
                logger.info(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                            f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                            f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
                writer.add_scalar('Loss/D', errD.item(), epoch * len(dataloader) + i)
                writer.add_scalar('Loss/G', errG.item(), epoch * len(dataloader) + i)

        # Checkpointing
        if (epoch + 1) % config['training']['save_interval_epoch'] == 0:
            torch.save(netG.state_dict(), f"checkpoints/G_epoch_{epoch+1}.pth")
            torch.save(netD.state_dict(), f"checkpoints/D_epoch_{epoch+1}.pth")
            
            # Save samples
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            vutils.save_image(fake, f"samples/fake_samples_epoch_{epoch+1}.png", padding=2, normalize=True)

    logger.info("Training Finished.")

if __name__ == "__main__":
    train()
