#!/usr/bin/env python3
from torch import nn
import torch
import torch.nn.functional as F

class MNistFC(nn.Module):
    def __init__(self):
        super(MNistFC, self).__init__()
        self.linear1 = nn.Linear(28*28, 100)
        self.linear2 = nn.Linear(100, 50)
        self.final = nn.Linear(50, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        a = x.view(-1, 28*28)
        a = self.relu(self.linear1(a))
        a = self.relu(self.linear2(a))
        a = self.final(a)
        return a
    
class MNistCNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(MNistCNN, self).__init__(*args, **kwargs)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=18, kernel_size=7, stride=3, padding=4),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=4),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(18, 34, 7, 3, padding=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.out = nn.Linear(34*9*9, 12)

    def forward(self, x):
        a = self.conv1(x)
        a = self.conv2(a)


class MNistCNN2(nn.Module):
    def __init__(self) -> None:
        super(MNistCNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        return out



def main():
    model = MNistFC()
    print(model)

if __name__ == '__main__':
    main()