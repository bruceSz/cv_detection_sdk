#!/usr/bin/env python3

import torch
from torchvision.models import resnet50
from  torchvision import models
from torch import nn



class TrainContext(object):
    def __init__(self):
        self.model_save_path =  "resnet50_best.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.model = get_model().to(self.device)
        self.optimizer = get_optimizer(self.model)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5 )
        self.criterion = nn.BCELoss()
        self.train_logs = {
            'loss':[],
            'acc':[],
            'time':[]
        }
        self.val_logs = {
            'loss':[],
            'acc':[],
            'time':[]
        }
        self.epochs = 10



def get_model():
    #before 0.13
    #model = resnet50(pretrained=True)
    # after 0.13
    model = resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(nn.Linear(2048, 1, bias=True),
                             nn.Sigmoid())
    return model


def get_optimizer(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return optimizer
    
    