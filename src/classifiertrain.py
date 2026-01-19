import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets

from src.dataloader import get_dataloader, get_transforms
from src.config import load_config
from src.utils import get_device

def train_classifier():
    parser = argparse.ArgumentParser(description="Train downstream classifier")
    parser.add_argument("--config", type=str, default="configs/dataconfig.yaml")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--augment_dir", type=str, default=None, help="Path to synthetic images root")
    parser.add_argument("--checkpoint", type=str, default=None, help="Output checkpoint name")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    # Real data loader
    if args.batch_size is not None:
        config['dataset']['batch_size'] = args.batch_size
    real_loader = get_dataloader({"dataset": config['dataset']}, split='train')

    datasets_list = [real_loader.dataset]
    if args.augment_dir:
        transform = get_transforms(config['dataset']['image_size'], config['dataset']['channels'])
        synth_dataset = datasets.ImageFolder(root=args.augment_dir, transform=transform)
        datasets_list.append(synth_dataset)

    train_dataset = ConcatDataset(datasets_list)
    dataloader = DataLoader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=config['dataset'].get('num_workers', 2),
        pin_memory=True
    )

    num_classes = len(config.get('classes', [0, 1, 2]))

    # Load ResNet
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    print("Starting Classifier Training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {running_loss / max(1, len(dataloader)):.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_name = args.checkpoint or ("classifier_augmented.pth" if args.augment_dir else "classifier_baseline.pth")
    torch.save(model.state_dict(), os.path.join("checkpoints", ckpt_name))
    print(f"Classifier Training Finished. Saved to checkpoints/{ckpt_name}")

if __name__ == "__main__":
    train_classifier()
