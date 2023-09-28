#!/usr/bin/env python3

from torch import nn
from torch import Tensor
from typing import List
from utils import NestedTensor
import torch

@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor])->NestedTensor:
    max_size = []
    # compute max size for each dim of this tensor_list.
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape(i) for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)


    padded_imgs = []
    padded_masks = []

    for img in tensor_list:
        # compute padding size (max_size - img.shape) for each dim.
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        # TODO? padded_img: [bs h, p1, w, p2, c] ?
        padded_i = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_i)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask)

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


    