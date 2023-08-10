#!/usr/bin/env python3

import os
import torch
import numpy as np
from torch.hub import load_state_dict_from_url
from common.utils import singleton
import torch.distributed  as dist
from centernet.centernet_resnet50 import CenterNet_Resnet50
from ssd.ssd300 import SSD_VGG_300
from fcn.fcn32 import FCN32
from common import utils

from framework import dataset_mgr


# class CenterNet_Resnet50_Proxy(object):
#     WEIGHT_NAME = "centernet_resnet50_voc.pth"
    
#     def __init__(self, n_classes, pretrained) -> None:
#         self.n_classes = n_classes
#         self.pretrained = pretrained
#         self._init_weights()
        
        
#     def _init_weights(self):
#         full_path = os.path.join(ModelManager.LOCAL_MODEL_PATH, self.WEIGHT_NAME)
#         if os.path.isfile(full_path):
#             model = 

class CenterNet_Hourglass(object):
    def __init__(self, info_dict, pretrained) -> None:
        self.info_dict = info_dict
        self.pretrained = pretrained


    

class BackboneMgr(object):
    BACKBONE_URL_MAP = {
        "resnet50" : 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
        "hourglass"  :  "",
        "vgg16"    : "https://download.pytorch.org/models/vgg16-397923af.pth",
    }

    

    def __init__(self, tc) -> None:
        #self.name = tc.backbone_name
        #self.dataset_mgr = dataset_mgr.DataSetMgr()
        #self.num_classes = self.dataset_mgr.get_dataset(tc.dataset_name).num_classes
        #self.tc = tc
        self.backbone_name = tc.backbone_name
        assert(self.backbone_name is not None)
        print("backbone is: {}".format(self.backbone_name))
        assert(self.backbone_name in self.BACKBONE_URL_MAP)
        #self.backbones = {}

    def init_with_download(self):
        url = self.BACKBONE_URL_MAP[self.backbone_name]
        if not os.path.isdir(ModelManager.LOCAL_MODEL_PATH):
            os.makedirs(self.LOCAL_MODEL_PATH)
        load_state_dict_from_url(url, ModelManager.LOCAL_MODEL_PATH)
        #download_weights(backbone)

    def get_model(self, model_cfg):
        #if name == "resnet50":
        return self.backbones[model_cfg.name](model_cfg)
            #return CenterNet_Resnet50(self.num_classes, pretrained=self.tc.pretrained)
            
    # def init_locally(cls, name):
    #     if name == "resnet50":
    #         self.model = CenterNet_Resnet50(self.num_classes, pretrained=self.tc.pretrained)

   


class ModelManager(object):
    LOCAL_MODEL_PATH = "./models/"
    def __init__(self, tc) -> None:
        self.pretrained = tc.pretrained
        self.distributed = tc.distributed
        self.tc = tc
        self.model_name = tc.model_name
        self.backbone = tc.backbone_name
        self.backbone_mgr = BackboneMgr(tc)

    def _get_weight_name(self, m_name):
        if m_name == "centernet_resnet50":
            return "centernet_resnet50_voc.pth"
        else:
            raise NotImplementedError("Unknown model: {}".format(m_name))
    def init_backbone(self):
        """
        Initialize the model by download it from url.
        """
        if self.pretrained:
            print("pretrained: loading backbone weights...")
            if self.distributed:
                if self.tc.local_rank == 0:
                    #print("=> using pre-trained model '{}'".format(self.pretrained))
                    self.backbone_mgr.init_with_download()
                dist.barrier()
            else:
                self.backbone_mgr.init_with_download()

    def get_model(self):
        if self.model_name == "centernet" and self.backbone == "resnet50":
            
            model =  CenterNet_Resnet50(self.tc.num_classes, pretrained=self.tc.pretrained, 
                                        tc=self.tc)
            #model_filename = self._get_weight_name(args.model_name + "_" + self.backbone)
            self._init_model_with_weight( model)
            return model
        if self.model_name == "fcn32":
            model = FCN32(self.tc.num_classes)
            return model
        elif self.model_name == "centernet_hourglass" and self.backbone == "hourglass":
            model =  CenterNet_Hourglass({"hm": self.tc.num_classes, "wh": 2, "reg":2}, pretrained=self.tc.pretrained)
            self._init_model_with_weight( model)
            return model
        
        else:
            raise NotImplementedError("Unknown model: {}".format(args.model_name))
        
    def _init_model_with_weight(self,   model): 
        path = self._get_weight_name(self.model_name + "_" + self.backbone)
        print("ready to init model with weight file: {}".format( path))   
        
        full_path = os.path.join(self.LOCAL_MODEL_PATH, path)
        utils.load_model_from_path(model, full_path)

        
        

    @classmethod
    def get_model_infos(cls, model_name):
        print("model name: {}".format(model_name))
        if  "centernet" in model_name:
            return {
                "input_shape": [512, 512],
                "output_shape": [int(512/4), int(512/4)]
               # "collate_fn": collate_fn_centernet,
            }
        elif model_name == "fcn32":
            return {}
        elif model_name == "ssd":
            pass

        else:
            raise NotImplementedError("Unknown model: {}".format(model_name))