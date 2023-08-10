#!/usr/bin/env python3

import os
import datetime
import torch

import torch.optim as optim

import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from det_sdk.framework.scheduler import get_lr_scheduler
#from framework.train_helper import EvalCallback

from det_sdk.framework.train_helper import LossHistory
from det_sdk.framework.dataset_mgr import DataSetMgr
from det_sdk.framework.model_mgr import ModelManager

class CIFARConfig(object):
    def __init__(self) -> None:
        self.batch_size = 64

        self.num_print = 100
        self.epoch_n = 30

        self.lr = 0.01

        self.step_size = 10

        self.gamma = 0.3
        self.model_dir = './models'

        self.epoch_trained_model = './models/vgg16_epoch_31.pth'

class TrainConfig(object):
    
    def __init__(self, args) -> None:

        

        self.init_epoch = 0
        self.freeze_epoch = 1
        self.freeze_batch_size = 16
        self.unfreeze_epoch = 100
        self.unfreeze_batch_size = 8
        self.freeze_train = True
        self.init_lr = 5e-4
        self.min_lr = self.init_lr * 1e-2
        self.opt_type = "adam"
        self.momentum =  0.9
        self.weight_decay =  0
        self.lr_decay_type = 'cos'
        self.save_period = 5
        self.save_dir = "logs"
        self.eval_flag = True
        self.eval_period = 5
        self.num_workers = 4
        self.distributed = False
        #self.backbone_name = "restnet50"
        self.backbone_name = args.backbone
        self.unfreeze_flag = False
        self.fp16 =  False
        self.sync_bn = False
        self.Cuda = True
        self.ds_name = args.dataset
        self.pretrained = args.pretrained
        self.model_name = args.model_name
        


        #self.batch_size = self.freeze_batch_size if self.freeze_train else self.unfreeze_batch_size
      
        print("main model name {}, backbone name: {}".format(self.model_name, self.backbone_name))

        
        self.model_info = ModelManager.get_model_infos(self.model_name)
        #self.shapes = [ model_info["input_shape"], model_info["output_shape"]]
        #

        ds_mgr = DataSetMgr()
        ds_proxy, self.collate_fn = ds_mgr.get_dataset(self.ds_name, self.model_info)
        self.class_names = ds_proxy.class_names
        self.num_classes = ds_proxy.num_classes
        self.train_dataset = self._get_train_ds()
        self.val_dataset = self._get_val_ds()
        #self.val_lines = self.val_dataset.annotation_lines


        # if self.distributed:
        #     #TODO
        #     raise NotImplementedError("Distributed training is not implemented yet")
        # else:
        #     self.device = torch.device('cuda' if torch.cuda.device_count() > 0 else 'cpu')
        #     self.local_rank = 0
        self.n_gpus = torch.cuda.device_count()

        
        if self.fp16:
            from torch.cuda.amp import GradScaler as GradScaler
            self.scaler = GradScaler()
        else:
            self.scaler = None

        #TODO
        
        self.num_train = len(self.train_dataset)
        self.num_val = len(self.val_dataset)
        self._init_device()
        
        self._init_sampler()
        if self.local_rank ==0 :
            self.time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
            self.log_dir = os.path.join(self.save_dir, "loss_" + self.time_str)
            
    
    def _get_train_ds(self):
        return self._get_ds(True)
        #self.num_classes

    def _get_ds(self, train):
        ds_mgr = DataSetMgr()
        assert(self.model_info is not None)
        ds,_ = ds_mgr.get_dataset(self.ds_name, self.model_info, train=True)
        #print("leng of ds: {} ".format(len(ds))
        #print("num of classes: {}".format(ds.num_classes)
        print("num of ds: {}".format(len(ds)))
        return ds
    
    def _get_val_ds(self):
        return self._get_ds(False)


    def get_model_train(self, model):
        model_train = model.train()
        if self.sync_bn and self.n_gpus > 1 and self.distributed:
            model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
        elif self.sync_bn:
            print("Sync bn is not supported in single gpu training")
        
        if self.Cuda:
            if self.distributed:
                assert(self.local_rank is not None)
                model_train = model_train.cuda(self.local_rank)
                model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[self.local_rank], find_unused_parameters=True)
            else:
                #model_train = torch.nn.DataParallel(model)
                #cudnn.benchmark = True
                model_train = model_train.cuda()
        return model_train



    
    def get_optim(self, model, init_lr_fit, tc,  opt_type):
        opt = {
                'adam' : optim.Adam(model.parameters(), init_lr_fit, betas = (tc.momentum, 0.999), weight_decay =tc.weight_decay),
                'sgd'  : optim.SGD(model.parameters(), init_lr_fit, momentum=tc.momentum, nesterov=True, weight_decay=tc.weight_decay)
        }[opt_type]
        return opt
    
    def _init_device(self):
        if self.distributed:
            dist.init_process_group(backend="nccl")
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.rank = int(os.environ['RANK'])
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device('cuda' if torch.cuda.device_count() > 0 else 'cpu')
            self.local_rank = 0
            self.rank = 0
            

        if self.local_rank == 0:
        # show_config(
        #     classes_path = 
        # )
            #num_train = len(tc.train_dataset)
            self.wanted_step = 5e4  if self.opt_type == "sgd" else 1.5e4
            self.total_step = self.num_train // self.unfreeze_batch_size * self.unfreeze_epoch
            if self.total_step <= self.wanted_step:
                if self.num_train // self.unfreeze_batch_size < 0:
                    raise ValueError("Batch size is too small")
                print("wanted_step {}, num_train: {}, unfreeze_batch_size: {}".format(self.wanted_step, self.num_train, self.unfreeze_batch_size))
                self.wanted_epoch = self.wanted_step//(self.num_train // self.unfreeze_batch_size)
    
            
    
    def _init_sampler(self):
        if self.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True)
            self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False)
            self.shuffle = False
            # update batch_size as we will train on multiple gpus
            #self.batch_size = self.batch_size // self.n_gpus
        
        else:
            self.train_sampler = None
            self.val_sampler = None
            self.shuffle = True

    def get_train(self, bs):
        return DataLoader(self.train_dataset, shuffle=self.shuffle, batch_size=bs, 
                          num_workers=self.num_workers, pin_memory=True, 
                          drop_last=True, collate_fn=self.collate_fn,
                          sampler=self.train_sampler)
    

    def get_val(self, bs):
        
        return DataLoader(self.val_dataset, shuffle=self.shuffle, batch_size=bs,
                          num_workers=self.num_workers, pin_memory=True,
                          drop_last=True, collate_fn=self.collate_fn,
                          sampler=self.val_sampler)

    # @property
    # def train_num(self):
    #     return len(self.train_dataset)
    
    # @property
    # def val_num(self):
    #     return len(self.val_dataset)

    # def get_eval_cb(self, model):
    #     if self.local_rank == 0:
    #         self.eval_callback = EvalCallback(model, self.backbone_name, self.shapes[0], self.class_names, 
    #                                      self.num_classes, self.val_lines, self.log_dir, self.Cuda, \
    #                                         eval_flag=self.eval_flag, period=self.eval_period)
    #     else:
    #         self.eval_callback = None
        
        #return eval_callback
    
    def get_train_adapt_lr(self, bs, nbs=64):
        lr_limit_max= 5e-4 if self.opt_type == "adam" else 5e-2
        lr_limit_min = 2.5e-4 if self.opt_type == "adam" else 5e-4

        init_lr_fit = min(max(bs/ nbs * self.init_lr, lr_limit_min), lr_limit_max)
        min_lr_fit = min(max(bs / nbs * self.min_lr, lr_limit_min * 1e-2), lr_limit_max*1e-2)

        lr_scheduler_func = get_lr_scheduler(self.lr_decay_type, init_lr_fit, min_lr_fit, self.unfreeze_epoch)
        return init_lr_fit, min_lr_fit, lr_scheduler_func
        
    def check_dataset_size(self, bs):
        # get step 
        epoch_step = self.num_train // bs
        epoch_step_val = self.num_val // bs

        print("num_train: {} , num_val: {}, bs: {}, epoch_step: {}, epoch_val_step: {}".format(self.num_train, self.num_val, bs, epoch_step, epoch_step_val))
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Batch size is too small")

    def get_loss_history(self, model):
        if self.local_rank == 0:
            return LossHistory(self.log_dir, model)
        else:
            return None
        
    def get_train_model(self, model):

        model_train = model.train()
        if self.sync_bn and self.n_gpus > 1 and self.distributed:
            model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            pass
        
        if self.distributed:
            model_train = model_train.cuda(self.local_rank)
            #model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[self.local_rank], output_device=self.local_rank)
            model_train = torch.nn.prallel.DistributedDataParallel(model_train, device_ids=[self.local_rank], find_unused_parameters=True)

        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
        return model_train
