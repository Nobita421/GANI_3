import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

from src.config import load_config
from src.utils import get_device
from src.dataloader import get_dataloader
from src.generator import Generator
from src.metrics import compute_activation_stats, compute_fid, inception_score


class GeneratedDataset(Dataset):
    def __init__(self, generator, z_dim, num_samples, device):
        self.generator = generator
        self.z_dim = z_dim
        self.num_samples = num_samples
        self.device = device

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        noise = torch.randn(1, self.z_dim, 1, 1, device=self.device)
        with torch.no_grad():
            img = self.generator(noise).squeeze(0)
        return img.cpu(), 0


def _collect_features(dataloader, device, max_batches=None):
    from src.metrics import _get_inception, _get_pool_features

    model, preprocess = _get_inception(device)
    feats = []
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            images = preprocess(images)
            feat = _get_pool_features(model, images)
            feats.append(feat.cpu().numpy())
            if max_batches is not None and (i + 1) >= max_batches:
                break
    return np.concatenate(feats, axis=0)


def nearest_neighbor_grid(real_loader, fake_loader, device, out_path, max_batches=5):
    real_feats = _collect_features(real_loader, device, max_batches=max_batches)
    fake_feats = _collect_features(fake_loader, device, max_batches=max_batches)

    # Collect corresponding images (small subset)
    real_imgs = []
    fake_imgs = []
    for (images, _), (fimages, _) in zip(real_loader, fake_loader):
        real_imgs.append(images)
        fake_imgs.append(fimages)
        if len(real_imgs) >= max_batches:
            break
    real_imgs = torch.cat(real_imgs, dim=0)
    fake_imgs = torch.cat(fake_imgs, dim=0)

    # Compute nearest neighbors
    neighbors = []
    for i in range(min(len(fake_feats), len(fake_imgs))):
        distances = np.linalg.norm(real_feats - fake_feats[i], axis=1)
        nn_idx = int(np.argmin(distances))
        neighbors.append(real_imgs[nn_idx])

    # Save grid: fake on top, nearest real below
    fake_imgs = fake_imgs[:len(neighbors)]
    neighbors = torch.stack(neighbors, dim=0)
    grid = torch.cat([fake_imgs, neighbors], dim=0)
    save_image((grid + 1) / 2.0, out_path, nrow=len(fake_imgs), normalize=False)


def run_evaluation(checkpoint_path, config_path, split, num_images, batch_size):
    device = get_device()
    config = load_config(config_path)

    data_cfg = load_config("configs/dataconfig.yaml")
    merged = dict(data_cfg)
    merged.update(config)
    config = merged

    # Real dataloader
    real_loader = get_dataloader(config, split=split)

    # Load generator
    netG = Generator(
        z_dim=config['model']['z_dim'],
        feature_maps=config['model']['feature_maps_g'],
        channels=3
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    netG.load_state_dict(state_dict)
    netG.eval()

    # Fake dataloader
    fake_dataset = GeneratedDataset(netG, config['model']['z_dim'], num_images, device)
    fake_loader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=False)

    # Metrics
    mu_real, sigma_real = compute_activation_stats(real_loader, device)
    mu_fake, sigma_fake = compute_activation_stats(fake_loader, device)
    fid = compute_fid(mu_real, sigma_real, mu_fake, sigma_fake)
    is_mean, is_std = inception_score(fake_loader, device)

    # Nearest neighbor grid
    figures_dir = config.get('training', {}).get('figures_dir', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    nn_path = os.path.join(figures_dir, "nearest_neighbors.png")
    nearest_neighbor_grid(real_loader, fake_loader, device, nn_path)

    print(f"FID: {fid:.4f}")
    print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
    print(f"Nearest neighbor grid saved to {nn_path}")


def main():
    parser = argparse.ArgumentParser(description="GAN Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator checkpoint")
    parser.add_argument("--config", type=str, default="configs/trainconfig.yaml", help="Train config path")
    parser.add_argument("--split", type=str, default="val", help="Dataset split: train/val/test")
    parser.add_argument("--num_images", type=int, default=256, help="Number of generated images")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for eval")
    args = parser.parse_args()

    run_evaluation(args.checkpoint, args.config, args.split, args.num_images, args.batch_size)


if __name__ == "__main__":
    main()
