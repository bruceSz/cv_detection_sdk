#!/usr/bin/env python

import torch.nn as nn
import torch

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x-mean)/(var + self.eps).sqrt() * self.g + self.bias

class PatchEmbedding(nn.Module):
    def __init__(self, dim, dim_out,patch_size,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 
        self.dim = dim
        # emb_dim
        self.dim_out = dim_out
        self.patch_size = patch_size

        # project input 2d patch into 1d vector
        self.proj = nn.Sequential(
            LayerNorm(patch_size **2 *dim)

