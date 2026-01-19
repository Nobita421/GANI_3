import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from src.dataloader import get_dataloader
from src.config import load_config
from src.utils import get_device

def train_classifier():
    # Simplified training script for downstream task
    config = load_config("configs/dataconfig.yaml")
    device = get_device()
    
    dataloader = get_dataloader(config, split='train')
    
    # Load ResNet
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(config.get('classes', [0,1,2]))) # default 3 classes
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    print("Starting Classifier Training (Demo)...")
    for epoch in range(1): # Just 1 epoch demo
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    print("Classifier Training Finished.")
    torch.save(model.state_dict(), "checkpoints/classifier_baseline.pth")

if __name__ == "__main__":
    train_classifier()
