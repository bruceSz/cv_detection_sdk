#!/usr/bin/env python3


from torch import nn
import torch.nn.functional as F
from typing import Optional, List
from torch import Tensor
import torch

def _expand(t, length: int):
    return t.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)

class MaskHeadSmallConv(nn.Module):

    def __init__(self, dim, fpn_dims, context_dim):
        super(MaskHeadSmallConv, self).__init__()
        inter_dims = [dim, context_dim // 2, context_dim//4, context_dim//8,
                      context_dim // 16, context_dim // 64]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1  = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        self.dim = dim
        # 1* 1 conv2d to change change dim.
        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # scale factor: sqrt(2/n), where n is input num.
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)



    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):
        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0,1)], 1)

        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        # do interpolation.
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        
        # do interpolation.
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        x = self.out_lay(x)
        return x





class MHAttentionMap(nn.Module):
    """
    """
    def __init__(self, query_dim, hidden_dim, heads, dropout=0.1, bias=True):
        super(MHAttentionMap, self).__init__()
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_liear.bias)
        nn.init.zeros_(self.q_linear.bias)

        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.k_linear.weight)
        self.normalize = float(hidden_dim/ self.heads) ** -0.5
    
    def forward(self, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        # this k_linear is treated as 1*1 conv2d with depth as query_dim 
        # output channel is hidden_dim
        k = F.conv2d(k, self.k_linear.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.heads, self.hidden_dim // self.heads)
        kh = k.view(k.shape[0], self.heads, self.hidden_dim // self.heads, k.shape[-2], k.shape[-1])
        # 1. same b
        # 2. qnc, nc -> qn
        # broadcast _, hw -> hw
        #TODO
        weights = torch.einsum("bqnc,bnchw -> bqnhw", qh * self.normalize, kh)


        if mask is not None:
            weights = weights.masked_fill(mask == 0, float("-inf"))
        #1. flatten hw to 1 dim.
        #2. softmax for each member of last dim
        #3. reshape to original shape
        weights = F.softmax(weights.flatten(-2), dim=-1).view(weights.size())
        #dropout 
        weights = self.dropout(weights)
        return weights