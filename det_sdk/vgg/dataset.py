
#!/usr/bin/env python3

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def transform_flip():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.226, 0.225, 0.224))
    ])

    transform_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406),(0.226, 0.225, 0.224))])
    

    train_dat = datasets.CIFAR10(root='./data', train=True, 
                                 download=True, transform=transform_train)
    
    val_dat = datasets.CIFAR10(root='./data', train=False,
                                 download=True, transform=transform_val)
    
    return train_dat, val_dat

def get_loaders(conf):
    train_dat, val_dat  = transform_flip()
    train_loader = DataLoader(train_dat, batch_size=conf.batch_size, 
                              shuffle=True)
    
    val_loader = DataLoader(val_dat, batch_size=conf.batch_size,
                            shuffle=False)
    
    return train_loader, val_loader
