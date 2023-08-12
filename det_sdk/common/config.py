#!/usr/bin/env python3

import torch

class Config(object):
    def __init__(self):
        self.lr = 1e-4
        self.batch_size = 64
        self.epochs = 20
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')