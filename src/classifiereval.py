import torch
from torchvision import models
from src.dataloader import get_dataloader
from src.config import load_config
from src.utils import get_device
import torch.nn as nn

def eval_classifier():
    config = load_config("configs/dataconfig.yaml")
    device = get_device()
    
    dataloader = get_dataloader(config, split='val') # or test
    
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(config.get('classes', [0,1,2])))
    
    try:
        model.load_state_dict(torch.load("checkpoints/classifier_baseline.pth"))
    except:
        print("No checkpoint found.")
        return

    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f'Accuracy: {100 * correct / total}%')

if __name__ == "__main__":
    eval_classifier()
