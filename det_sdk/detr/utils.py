#!/usr/bin/env python3

from typing import Optional
from torch import Tensor

class NestedTensor(object):
    def __init__(self, tensors,mask:Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask
    
    def to(self, device):
        cast_t = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert(mask is not None)
            cast_m = mask.to(device)
        else:
            cast_m = None
        return NestedTensor(cast_t, cast_m)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self) -> str:
        return str(self.tensors)