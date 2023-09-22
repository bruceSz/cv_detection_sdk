#!/usr/bin/env python3

import math
import torch
from torch import nn

from utils import NestedTensor

class PositionEmbeddingSine(nn.Module):
    """
        Presented by 'attention is all you need' paper.
    """
    def __init__(self, num_pos_feats = 64, temp = 10000, normalize = False, 
                 scale=None):
        super(PositionEmbeddingSine, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temp = temp
        self.normalize = normalize

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        
        self.scale = scale


    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert(mask is not None)
        not_mask = ~mask

        y_emb = not_mask.cumsum(1, dtype=torch.flaot32)
        x_emb = not_mask.cumsum(2, dtype=torch.float32)

        # the last index should be sum of all the mask
        # normalize here is to device each element by the last element
        # so that the last element is 1, and the rest is a fraction of 1
        # even distribution.
        if self.normalize:
            eps = 1e-6
            y_emb = y_emb / (y_emb[:,-1:, :] + eps) * self.scale
            x_emb = x_emb / (x_emb[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, 
                             device=x.device)
        # 10000^(2i/d)
        dim_t = self.temp ** (2 * (dim_t // 2)/self.num_pos_feats)
        
        # ? [b, h, w]
        # sin(10000^(2i/d))
        #TODO figure it out. usage of None here.
        # answer: it will be broadcasted to the shape of x_emb
        pos_x = x_emb[:,:,:, None]/dim_t
        pos_y = y_emb[:,:,:, None]/dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sim(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # final shape: [batch_size, pos_emd, h, w]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    def __init__(self, num_pos_feats=256):
        super(PositionEmbeddingLearned, self).__init__()
        self.row_emb = nn.Embedding(50, num_pos_feats)
        self.col_emb = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_emb.weight)
        nn.init.uniform_(self.col_emb.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)

        x_emb = self.col_emb(i)
        y_emb = self.row_emb(j)

        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1,1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1 , 1, 1)

        return pos


def build_pos_emb(args):
    steps = args.hidden_dim // 2
    if args.pos_emb in ('v2', 'sin'):
        pos_emb = PositionEmbeddingSine(steps, normalize = True)
    elif args.pos_emb in ('v3', 'learned'):
        pos_emb = PositionEmbeddingLearned(steps)
    else:
        raise ValueError(f"not supported {args.pos_emb}")
    return pos_emb