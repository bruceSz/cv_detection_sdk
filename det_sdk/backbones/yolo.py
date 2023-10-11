#!/usr/bin/env python3

from torch import tensor
from torch import nn

from common.helper_block import v3parser_model_config as parse_model_config

def create_modules(module_defs: list) :
    """
        construct a model based on module_defs(list of nn block)
    """
    # the first module_def should be in [net] block
    # This contains hyperparameters for the network.
    paras = module_defs.pop(0)
    paras.update({
        'batch':int(paras['batch']),
        # parameters as divisor for batch size.
        'subdivisions':int(paras['subdivisions']),
        'width':int(paras['width']),
        'height':int(paras['height']),
        'channels':int(paras['channels']),
        'optimizer':paras['optimizer'],
        'momentum':float(paras['momentum']),
        'decay':float(paras['decay']),
        'learning_rate': float(paras['learning_rate']),
        # not widely used parameters in other work.
        # will increase the learning rate for the first few iterations.
        'burn_in': int(paras['burn_in']),
        'max_batches': int(paras['max_batches']),
        'policy': paras['policy'],
        'lr_steps':list(zip(map(int, paras['steps'].split(',')), 
                            map(float, paras['scales'].split(',')))),

    })


class Darknet(nn.Module):
    """
        yolov3 detection backbone
    """
    def __init__(self, conf_path):
        super(Darknet, self).__init__()
        self.module_dfs = parse_model_config(conf_path)

class Yolov5Base(nn.Module):

    """
        Base model class for yolov5
        inputs:
            ch: channels for detection
            inplace: inplace operation
            vis: visualize
    """
    def forward(self, x, profile=False, vis=False):


class Yolo5Detect(nn.Module):
    """
        Detection head for yolov5
        inputs:
            num_classes: number of classes
            anchors: anchors for detection
            ch: channels for detection
            inplace: inplace operation
    """
    stride = None
    dynamic = False
    export = False

    def __init__(self, num_classes = 80 , anchors=(), ch = (), inplace = True):
        super(Yolo5Detect, self).__init__()
        self.num_classes = num_classes
        # 4(coor) + 1(obj conf) + num_classes (conf)
        self.feat_out_ = num_classes + 5
        # each anchor will need a output layer.
        self.num_anchors = len(anchors)

        # anchor grid
        self.anchor_grid = [tensor.empty(0) for _ in range(self.num_anchors)]

        #self.m = nn.ModuleList(nn.Conv2d(x, num_anchors * (num_classes + 5), 1) for x in ch)
        #TODO.