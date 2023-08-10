#!/usr/bin/env python3
import torch
import argparse
import os



from framework.arg_parser import create_parser
from framework.train_config import TrainConfig
from framework.model_mgr import ModelManager


from framework.scheduler import set_optimizer_lr
#from framework.train_helper import fit_one_epoch




def train_loop(model, model_train, tc, model_cus_train):
    
    

    if tc.freeze_train and  hasattr(model, "freeze_backbone"):
        model.freeze_backbone()

    loss_history = tc.get_loss_history(model)
    nbs = 64
    batch_size = tc.freeze_batch_size if tc.freeze_train else tc.unfreeze_batch_size
    init_lr_fit, min_lr_fit, lr_scheduler_func =  tc.get_train_adapt_lr(batch_size, nbs=nbs)
    unfreeze_flag = False
    
    # lr_limit_max = 5e-4 if tc.opt_type == "adam" else 5e-2
    # lr_limit_min = 2.5e-4 if tc.opt_type == "adam" else 5e-4

    # init_lr_fit = min(max(tc.batch_size/nbs * tc.init_lr, lr_limit_min), lr_limit_max)
    # min_lr_fit = min(max(tc.batch_size/nbs * tc.min_lr, lr_limit_min * 1e-2), lr_limit_max*1e-2)

    optimizer = tc.get_optim(model, init_lr_fit, tc, tc.opt_type)

    
    tc.check_dataset_size(batch_size)
    
    epoch_step = tc.num_train // batch_size
    epoch_step_val = tc.num_train // batch_size
    
    gen = tc.get_train(batch_size)
    gen_val = tc.get_val(batch_size)

    eval_cb = model.get_eval_cb()


    
    for epoch in range(tc.init_epoch, tc.unfreeze_epoch):
        if epoch >= tc.freeze_epoch and not unfreeze_flag and tc.freeze_train:
            batch_size = tc.unfreeze_batch_size
            nbs = 64

            init_lr_fit, min_lr_fit, lr_scheduler_func =  tc.get_train_adapt_lr(batch_size)
            

            model.unfreeze_backbone()
            epoch_step = tc.num_train // batch_size
            epoch_step_val = tc.num_val // batch_size
            tc.check_dataset_size(batch_size)
            if tc.distributed:
                batch_size = batch_size // tc.n_gpus

            gen = tc.get_train(batch_size)
            gen_val = tc.get_val(batch_size)

            unfreeze_flag = True
        if tc.distributed:
            tc.train_sampler.set_epoch(epoch)
            
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        model_cus_train(model_train, model, loss_history, eval_cb, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, tc.unfreeze_epoch, tc.Cuda, 
                    tc.fp16, tc.scaler,tc.backbone_name, tc.save_period, tc.save_dir, 
                    tc.local_rank)

    if tc.local_rank == 0:
        loss_history.writer.close()


            
            


def test():
    flags = create_parser()
    #args = flags.parse_args()
    #print(args)
   
    """ tc = TrainConfig(args)
    model_mgr = ModelManager(tc)

    model_mgr.init()
    model = model_mgr.get_model(flags.model)
 """


if __name__ == '__main__':
    test()