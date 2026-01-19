import matplotlib.pyplot as plt
import numpy as np

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
