#!/usr/bin/env python3

from torch import nn

from utils import NestedTensor

from typing import List

class Joiner(nn.Sequential):

    def __init__(self, backbone, pos_emb):
        super(Joiner, self).__init__(backbone, pos_emb)
    
    def forward(self, tensor_list: NestedTensor):
        # forward with backbone
        xs = self[0](tensor_list)
        out : List(NestedTensor) = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


        

