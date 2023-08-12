#!/usr/bin/env python3
import torch
from loss import ce_loss
import numpy as np

from framework.arg_parser import create_parser
from framework.train_config import TrainConfig
from framework.model_mgr import ModelManager
from framework.train_loop import train_loop

def train_fcn(model_train, model, loss_history, eval_cb, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, unfreeze_epoch, Cuda, fp16, scaler,
                      backbone_name, save_period, save_dir, local_rank):
    
    for iter, batch in enumerate(gen):
        print("iter: ", iter)
        if iter > epoch_step:
            break
        with torch.no_grad():
            if Cuda:
                batch = [ann.cuda(local_rank) for ann  in batch]
        batch_imgs, batch_labels = batch
        
        optimizer.zero_grad()

        if not fp16:
            if Cuda:
                data, target = batch_imgs.cuda(), batch_labels.cuda()
            score = model_train(data)
            loss = ce_loss(score, target)
            # devided by batch_size
            loss /= len(data)

            loss_dat = loss.data.item()
            if np.isnan(loss_dat):
                raise ValueError('loss is nan while training')
            loss.backward()
            optimizer.step()


        else:
            raise RuntimeError("fp16 is not supported yet")
    

def train():
    print("train fcn start.")
    flags = create_parser()
    args = flags.parse_args()
    print(args)
   
    tc = TrainConfig(args)
    model_mgr = ModelManager(tc)

    model_mgr.init_backbone()
    model = model_mgr.get_model()
    model_train = tc.get_model_train(model)

    #print(model)
    train_loop(model, model_train,  tc, train_fcn)
    


if __name__ == "__main__":
    train()
    
