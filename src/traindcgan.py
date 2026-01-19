import os
import csv
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
from src.visualization import plot_loss

def train():
    args = parse_args()
    config_path = args.config if args.config else "configs/trainconfig.yaml"
    config = load_config(config_path)

    # Merge data config (shallow merge is ok for dataset/classes keys)
    data_config = load_config("configs/dataconfig.yaml")
    merged = dict(data_config)
    merged.update(config)
    config = merged

    set_seed(42)
    device = get_device()
    preferred_device = config.get('training', {}).get('device')
    if preferred_device:
        preferred_device = preferred_device.lower()
        if preferred_device == 'cpu':
            device = torch.device('cpu')
        elif preferred_device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
    
    # Setup Logger
    log_dir = config.get('training', {}).get('log_dir', "logs")
    logger, writer = setup_logger(log_dir)
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
    
    # Output dirs
    checkpoints_dir = config.get('training', {}).get('checkpoints_dir', "checkpoints")
    samples_dir = config.get('training', {}).get('samples_dir', "samples")
    figures_dir = config.get('training', {}).get('figures_dir', "figures")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Params
    num_epochs = config['training']['epochs']
    z_dim = config['model']['z_dim']
    fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)
    
    real_label = 1.0
    fake_label = 0.0
    
    # Stabilization
    label_smoothing = config['stabilization'].get('label_smoothing', False)
    label_flipping = config['stabilization'].get('label_flipping', False)
    input_noise = config['stabilization'].get('input_noise', False)
    flip_prob = config['stabilization'].get('flip_prob', 0.05)
    grad_clip = config['stabilization'].get('grad_clip', None)
    input_noise_max = config['stabilization'].get('input_noise_max', 0.05)
    input_noise_decay_epochs = config['stabilization'].get('input_noise_decay_epochs', 10)

    d_losses = []
    g_losses = []

    csv_log_path = os.path.join(log_dir, 'train_metrics.csv')
    if not os.path.exists(csv_log_path):
        with open(csv_log_path, 'w', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(["step", "epoch", "batch", "loss_d", "loss_g", "d_x", "d_g_z1", "d_g_z2"])
    
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

            # Label Flipping
            if label_flipping and np.random.rand() < flip_prob:
                label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)

            # Input Noise (decayed)
            if input_noise:
                decay = max(0.0, 1.0 - (epoch / max(1, input_noise_decay_epochs)))
                noise_strength = input_noise_max * decay
                if noise_strength > 0:
                    real_cpu = real_cpu + noise_strength * torch.randn_like(real_cpu)

            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            noise = torch.randn(b_size, z_dim, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            if label_flipping and np.random.rand() < flip_prob:
                label.fill_(real_label)
            
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            errD = errD_real + errD_fake
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(netD.parameters(), grad_clip)
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ############################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(netG.parameters(), grad_clip)
            D_G_z2 = output.mean().item()
            optimizerG.step()

            d_losses.append(errD.item())
            g_losses.append(errG.item())
            
            # Logging
            if i % config['training']['log_interval_step'] == 0:
                logger.info(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                            f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                            f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
                step = epoch * len(dataloader) + i
                writer.add_scalar('Loss/D', errD.item(), step)
                writer.add_scalar('Loss/G', errG.item(), step)
                with open(csv_log_path, 'a', newline='') as f:
                    writer_csv = csv.writer(f)
                    writer_csv.writerow([step, epoch, i, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2])

        # Checkpointing
        if (epoch + 1) % config['training']['save_interval_epoch'] == 0:
            torch.save(netG.state_dict(), os.path.join(checkpoints_dir, f"G_epoch_{epoch+1}.pth"))
            torch.save(netD.state_dict(), os.path.join(checkpoints_dir, f"D_epoch_{epoch+1}.pth"))

            # Save samples
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            vutils.save_image(fake, os.path.join(samples_dir, f"fake_samples_epoch_{epoch+1}.png"), padding=2, normalize=True)

    # Save loss plot
    if d_losses and g_losses:
        plot_loss(d_losses, g_losses, os.path.join(figures_dir, "loss_curve.png"))

    writer.close()
    logger.info("Training Finished.")

if __name__ == "__main__":
    train()
