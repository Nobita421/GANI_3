import os
import argparse
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from src.dataloader import get_dataloader
from src.config import load_config
from src.utils import get_device

def eval_classifier():
    parser = argparse.ArgumentParser(description="Evaluate downstream classifier")
    parser.add_argument("--config", type=str, default="configs/dataconfig.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/classifier_baseline.pth")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--figures_dir", type=str, default="figures")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    dataloader = get_dataloader({"dataset": config['dataset']}, split=args.split)

    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(config.get('classes', [0, 1, 2])))

    if not os.path.exists(args.checkpoint):
        print("No checkpoint found.")
        return

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())

    labels = list(range(len(config.get('classes', [0, 1, 2]))))
    target_names = config.get('classes', [str(i) for i in labels])

    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print(report)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    os.makedirs(args.figures_dir, exist_ok=True)
    fig_path = os.path.join(args.figures_dir, "confusion_matrix.png")
    _plot_confusion_matrix(cm, target_names, fig_path)
    print(f"Confusion matrix saved to {fig_path}")


def _plot_confusion_matrix(cm, classes, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

if __name__ == "__main__":
    eval_classifier()
