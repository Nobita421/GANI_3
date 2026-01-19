import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision.utils import save_image

def plot_loss(d_losses, g_losses, save_path):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss")
    plt.plot(d_losses, label="G")
    plt.plot(g_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def visualize_batch(batch, save_path):
    # Batch is tensor (B, C, H, W) in [-1, 1]
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    # Denormalize
    batch = (batch + 1) / 2.0
    plt.imshow(np.transpose(batch[0].cpu().numpy(), (1,2,0))) # Show first image
    plt.savefig(save_path)
    plt.close()


def latent_interpolation(generator, z_start, z_end, steps, save_path, device):
    """Generate a linear interpolation grid in latent space."""
    generator.eval()
    alphas = torch.linspace(0, 1, steps, device=device)
    zs = torch.stack([(1 - a) * z_start + a * z_end for a in alphas])
    zs = zs.view(steps, -1, 1, 1)
    with torch.no_grad():
        fake = generator(zs).detach().cpu()
    save_image((fake + 1) / 2.0, save_path, nrow=steps)


def feature_embedding(features, labels, save_path, method="pca"):
    """Plot PCA or t-SNE embeddings for feature vectors."""
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, init="pca")
    else:
        reducer = PCA(n_components=2, random_state=42)

    reduced = reducer.fit_transform(features)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=8, alpha=0.8)
    plt.colorbar(scatter)
    plt.title(f"Feature Embedding ({method.upper()})")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
