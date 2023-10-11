#!/usr/bin/env python3

import torch

from torch import optim
import torch.nn.functional as F

from model import MNistCNN2
from dataset import create_loader

def do_train(model, device, train_loader, opt, epoch, dry_run=True):
    best_loss = float('inf')
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        opt.zero_grad()

        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        opt.step()
        if loss < best_loss :
            best_loss = loss
            torch.save(model.state_dict(),'best_mnist_model.pth')


        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()
            ))
            if dry_run:
                break


def train():
    epochs = 14
    train_loader, test_loader = create_loader()
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNistCNN2().to(device)
    opt = optim.Adadelta(model.parameters(), lr=0.001)
    for epoch in range(1, epochs +1):
        do_train(model,device, train_loader,opt, epoch,dry_run=False)


if __name__ == '__main__':
    train()