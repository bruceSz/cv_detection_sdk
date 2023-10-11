#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from model import MNistCNN2
from dataset import create_loader


def do_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
           
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
def test():
    _, test_loader   = create_loader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNistCNN2().to(device)
    # adding best model will increase accuracy from 533 to 9249 (total 10000)
    model.load_state_dict(torch.load('best_mnist_model.pth', map_location=device))
    do_test(model, device, test_loader)

if __name__ == '__main__':
    test()