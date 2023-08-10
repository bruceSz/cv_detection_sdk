#!/usr/bin/env python3

import torch
import torch.nn.functional as F
#
# refer: https://blog.csdn.net/c20081052/article/details/89358658
#

def focal_loss(pred, target):
    #TODO n,c,h,w -> n,h,w,c?
    pred = pred.permute(0, 2, 3, 1)
    # where target == 1, pos_inds = 1, else 0
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    alpha = 2
    beta = 4 
    
    # weight-decay: (1-y)^beta
    # loss at each point:
    #    (1- pred)^alpha * log(pred) , where y == 1
    #    (1- y)^beta * pred^alpha * log(1-pred), where y == 0
    
    neg_weights = torch.pow(1 - target, 4)
    # clamp pred to range (0 + eps,1- eps)
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    # sum of pos_inds == number of pos, as pos_inds ==1 for all pos.
    # here num_pos is used to normalize the loss.
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos

    return loss

def reg_l1_loss(pred, target, mask):

    pred = pred.permute(0,2,3,1)
    # make mask same shape as pred (4-dim)
    expand_mask = torch.unsqueeze(mask, -1).repeat(1,1,1,2)

    loss = F.l1_loss(pred*expand_mask, target*expand_mask, reduction='sum')
    loss = loss/(mask.sum()+1e-4)
    return loss
