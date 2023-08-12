#!/usr/bin/env python
import torch
from samples.pytorch.vgg.model import VGG16_CF
from dataset import get_loaders
from demo_train import Config
from common import utils

def load_model(conf):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGG16_CF().to(device)
    
    utils.load_model_from_path(model, conf.epoch_trained_model)
    model.eval()
    return model

def main():
    conf = Config()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(conf)
    print("device: ", next(model.parameters()).device)
    _, val_loader = get_loaders(conf)
    total = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            #print(outputs.shape)
            #print(outputs)
            pred = outputs.argmax(dim=1)
            #print(pred.shape)
            total += labels.size(0)
            #print("label shape: ", labels.shape)
            #print("equal check same: ", torch.eq(pred, labels).sum().item())

            correct += torch.eq(pred, labels).sum().item()
    
    print("Accuracy: {:.2f}%".format(100.0 * correct / total))

if __name__ == '__main__':
    main()