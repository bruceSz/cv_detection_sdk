#!/usr/bin/env python3

import torch
from torch import nn


class Ensemble(nn.ModuleList):
    def __init__(self):
        super(Ensemble, self).__init__()
    
    def forward(self, x, augment=False, profile = False, vis = False):
        y = []
        for module in self:
            y.append(module(x, augment, profile, vis)[0])
        # cat along dim=1
        y = torch.cat(y, 1)
        # output: infer_output, train_output

        return y, None
    
def v3parser_model_config(path):
    """
        Parses the yolo-v3 layer configuration file and returns module definitions
        TODO change to yaml based parser.
    """
    with open(path, "r") as f:
        lines = f.read().split("\n")
        # skip empty line.
        lines = [x for x in lines if x and not x.startswith("#")]
        # strip leading and trailing spaces.
        lines = [x.rstrip().lstrip() for x in lines]
        module_defs = []
        for l in lines:
            if l.startswith("["):
                #new block
                module_defs.append({})
                # -1 here is the last one in module_defs.
                # aka the current block.
                # skip last charactor, which is ']'
                module_defs[-1]["type"] = l[1:-1].rstrip()
                if module_defs[-1]["type"] == "convolutional":
                    module_defs[-1]["batch_normalize"] = 0
            else:
                #print(l)
                k,v = l.split("=")
                v = v.strip()
                module_defs[-1][k] = v
        return module_defs




def attempt_load(model_w, device =None, inplace=True, fuse = True):
    from yolo import Model
    from yolo import Detect


if __name__ == "__main__":
    conf_path = "./model_defs/yolov3-tiny.cfg"
    c = v3parser_model_config(conf_path)
    print(c)