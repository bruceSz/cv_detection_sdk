#!/usr/bin/env python3

import time

from model import UNet

from dataset import ISBI_Loader
from common.utils import MyTime

from torch import optim
import torch.nn as nn
import torch



def warm_up(net, device, loader):
    net.eval()
    with torch.no_grad():
        for image, label in loader:
            image = image.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.float32)
            pred = net(image)
            #print("img shape:",image.shape)
            #print("pred shape:",pred.shape)
            #print("label shape:",label.shape)
            break

def train_net(net, device, data_path, epochs = 40, batch_size=1, lr = 0.00001):
    isbi_dataset = ISBI_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset = isbi_dataset, 
                                               batch_size = batch_size, 
                                               shuffle = True)
    #optimizer = optim.Adam(net.parameters(), lr = lr)
    optimizer = optim.RMSprop(net.parameters(), lr = lr, weight_decay=1e-8, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float('inf')
    
    warm_up(net, device, train_loader)

    with MyTime():
        for epoch in range(epochs):
            print("epoch: ", epoch)
            # train mode.
            net.train()

            for image, label in train_loader:
                # reset grad for each epoch
                optimizer.zero_grad()
                image = image.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.float32)


                pred = net(image)
                print("img shape:",image.shape)
                print("pred shape:",pred.shape)
                print("label shape:",label.shape)
                loss = criterion(pred, label)
                print("Loss/train: ", loss.item())

                if loss < best_loss:
                    best_loss = loss
                    torch.save(net.state_dict(), 'best_model_{}.pth'.format(epoch))

                # update and pass bachward gradient
                loss.backward()
                # update parameters
                optimizer.step()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    # gray picture, single channel
    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)
    #print("using cuda: ", torch.cuda.)
    data_path = '../data/ISBI/train'
    
    train_net(net, device, data_path, epochs=1)
        

if __name__ == "__main__":
    main()
    