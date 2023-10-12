#!/usr/bin/env python3

import torch
from torch import tensor
from torch import nn
from torch import functional as F
from itertools import chain
from typing import List, Tuple
from common.helper_block import v3parser_model_config as parse_model_config


class Upsample(nn.Module):
    """
    """
    def __init__(self, scale_factor, mode: str = "nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
    

class YoloLayer(nn.Module):
    def __init__(self, anchors: List[Tuple[int, int]], num_classes: int, new_coords: bool):
        super(YoloLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.new_coords = new_coords
        self.mse_loss = nn.MSELoss(reduction="none")
        self.bce_loss = nn.BCELoss(reduction="none")
        # number output per anchor
        self.no = num_classes + 5
        self.grid = torch.zeros(1)

        # chain here will iter all anchors and flatten them into a list
        # then view as a tensor with shape (num_anchors, 2)
        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2)
        self.register_buffer("anchors", anchors)
        self.register_buffer("anchor_grid", anchors.clone().view(1, -1, 1, 1, 2))
        self.stride = None

    def forward(self, x: torch.Tensor, img_size: int) -> torch.Tensor:
        # x is (batch_size, num_anchors * (num_classes + 5), grid_size, grid_size) or
        # (batch_size, num_anchors, num_classes + 5, grid_size, grid_size)
        # hence stride should be img_size / grid_size
        # each time we move a grid.
        stride = img_size / x.size(2)
        self.stride = stride
        bs, _, ny, nx = x.shape

        x = x.view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        if not self.training:
            if self.grid.shape[2:4] != x.shape[2:4]:
                # create new grid
                self.grid = self._make_grid(nx, ny).to(x.device)
            
            if self.new_coords:
                x[..., 0:2] = (x[..., 0:2]+ self.grid) * stride # xy
                x[..., 2:4] = x[..., 2:4] **2 * (4 * self.anchor_grid) # wh
            else:
                x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid) * stride # xy
                x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor_grid # wh
                x[..., 4:] = x[..., 4:].sigmoid() # p_conf, p_cls
            # flatten other dims except batch and output.
            x = x.view(bs, -1, self.no)
        return x



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

    # yolov3 check , input images should be hanled properly into square images.
    assert( paras["width"] == paras["height"], "width and height should be the same)")
    out_filters = [int(paras['channels'])]

    module_list = nn.ModuleList()

    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            # 
            pad = (kernel_size - 1) //2
            modules.add_module(
                f"conv_{i}",
                nn.Conv2d(
                    in_channels=out_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride = int(module_def["stride"]),
                    padding = pad,
                    bias = not bn
                )
            )
            if bn:
                modules.add_module(f"batch_norm_{i}", 
                                   nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{i}", nn.LeakyReLU(0.1))
            elif module_def["activation"] == "relu":
                modules.add_module(f"relu_{i}", nn.ReLU(inplace=True))
            elif module_def["activattion"] == "mish":
                modules.add_module(f"mish_{i}", nn.Mish())
            elif module_def["activation"] == "sigmoid":
                modules.add_module(f"sigmoid_{i}", nn.Sigmoid())
            elif module_def["activation"] == "silu":
                # s * sigmoid(x) also called swish
                modules.add_module(f"silu_{i}", nn.SiLU())
        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2  and stride ==1:
                # padding top, bottom, left, right
                modules.add_module(f"_debug_padding_{i}", nn.ZeroPad2d((0,1,0,1)))
            maxpool= nn.MaxPool2d(kernel_size=kernel_size, stride=stride, 
                                  padding= int((kernel_size -1)//2))
            modules.add_module(f"maxpool_{i}", maxpool)
        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode= "nearest")
            modules.add_module(f"upsample_{i}", upsample)
        
        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            #TODO?
            filters = sum([out_filters[1:][i] for i in layers]) // int(module_def.get("groups", 1))
            modules.add_module(f"route_{i}", nn.Sequential())

        elif module_def["type"] == "shortcut":
            filers = out_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{i}", nn.Sequential())
        elif module_def["type"] == "yolo":
            # mask are the index of anchors to use
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # anchor list
            # 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]

            num_classes = int(module_def["classes"])
            new_coords = bool(module_def.get("new_coords", False))


            # yolo layer detection layer
            yolo_layer = YoloLayer(anchors, num_classes, new_coords)
            modules.add_module(f"yolo_{i}", yolo_layer)
        module_list.append(modules)
        out_filters.append(filters)
    return paras, module_list







class Darknet(nn.Module):
    """
        yolov3 detection backbone
    """
    def __init__(self, conf_path):
        super(Darknet, self).__init__()
        self.module_dfs = parse_model_config(conf_path)
        self.hyperparas, self.module_list = create_modules(self.module_dfs)

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