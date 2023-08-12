#!/usr/bin/env python3
"""
Mask R-CNN
The main model implementation.

Copyright (c) 2023 X, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by bruceSz
"""

import numpy as np
import torch

from torch import nn

class MyBN(object):
    """
        momentum: samples mean and variance momenta
        eps: avoid numerical computation error
        n_features: feature dimension.
    """
    def __init__(self, momentum, eps, n_features) -> None:
        pass

        self.running_mean_ = 0
        self.running_var_ = 1
        self._momenta = momentum
        self._eps = eps
        self._beta = np.zeros(shape=(n_features,))
        self._gamma = np.ones(shape=(n_features,))

    def forward(self,x):
        x_mean = x.mean(axis=0)
        x_var  = x.var(axis=0)

        self.running_mean_ = (1-self._momenta) * x_mean + self._momenta * self.running_mean_
        self.running_var_ = (1-self._momenta) * x_var + self._momenta * self.running_var_
        x_hat = (x-x_mean)/np.sqrt(x_var + self._eps)
        y = self._gamma * x_hat + self._beta
        return y

def bn_test():
    torch.set_default_tensor_type(torch.DoubleTensor)
    data = np.array([[1.0,2.0],
                     [1.0,3.0],
                     [1.0,4.0]])
    bn_torch = nn.BatchNorm1d(num_features=2)
    
    data_torch = torch.from_numpy(data)
    data_torch = data_torch.to(torch.float64)
    print(data_torch.dtype)
    bn_output_torch = bn_torch(data_torch.double())
    print("bn torch done.")
    print("bn output: ",bn_output_torch)

    mbn = MyBN(momentum=0.01, eps = 0.001, n_features=2)
    mbn._beta = bn_torch.bias.detach().numpy()
    mbn._gamma = bn_torch.weight.detach().numpy()
    bn_out = mbn.forward(data)
    print(bn_out)

if __name__ == "__main__":
    bn_test()