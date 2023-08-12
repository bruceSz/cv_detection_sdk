#!/usr/bin/env python3

from distutils.version import LooseVersion
import torch
import  torch.nn.functional as F

def ce_loss(pred, target, size_avg = True):
    # pred: (n, c, h, w), target: (n, h, w)
    n, c, h, w = pred.size()

    if LooseVersion(torch.__version__)< LooseVersion('0.3'):
        # here dim is 1 instead of 0
        # dim == 0 only when pred's dim is 0, 1, 3
        log_p = F.log_softmax(pred)
    else:
        # set dim explicitly
        log_p = F.log_softmax(pred, dim=1)

    log_p = log_p.transpose(1,2).transpose(2,3).contiguous()
    # filter target not 255.
    log_p = log_p[target.view(n,h,w,1).repeat(1, 1, 1, c) >= 0]
    # flatten to (n*h*w, c)
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    # log_p : n, h, w, c(p for each c), target, n, h, w, each target is a class id
    loss = F.nll_loss(log_p, target, weight=w, reduction='sum')
    if size_avg:
        loss /= mask.data.sum()
    return loss 


