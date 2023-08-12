#!/usr/bin/env python3

import inspect
from common.utils import singleton
from torch.hub import load_state_dict_from_url

@singleton
class GlobalBackbones(object):
    def __init__(self) -> None:
        self.backbones = {}
        self.backbone_weight_urls = {}

    
    def RegisterBackbone(self,  backbone_cls):
        print("register backbone: ", backbone_cls)
        if not inspect.isclass(backbone_cls):
            raise TypeError("backbone_cls must be a class. {}".format(backbone_cls))
        
        self.backbones[backbone_cls.__BACKBONE__] = backbone_cls

        if hasattr(backbone_cls, "WEIGHT_URL"):
            self.backbone_weight_urls[backbone_cls.__BACKBONE__] = backbone_cls.WEIGHT_URL

        return backbone_cls

    def get_backbone(self, backbone_name, pretrained=False):
        print("get backbone: ", backbone_name)
        print("registered backbones: ", self.backbones.keys())
        if backbone_name in self.backbones:
            model =  self.backbones[backbone_name]()
            if pretrained:
                if backbone_name not in self.backbone_weight_urls:
                    raise NotImplementedError("backbone: {} has no weight url provided.".format(backbone_name))
                url = self.backbone_weight_urls[backbone_name]
                print("url is: ", url)
                state_dict = load_state_dict_from_url(url, model_dir = 'model_data/')
                #print(state_dict.keys())
                
                model.load_state_dict(state_dict)    
            else:
                print("not pretrained return raw model")
            return model
        else:
            raise NotImplementedError("Unknown backbone: {}  not registered.".format(backbone_name))

