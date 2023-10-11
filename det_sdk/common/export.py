#!/usr/bin/env python3

import os
import argparse
import torch

from torch_utils import compatible_infer_mode

def get_device(prefer):
    if not prefer:        
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        return torch.device(prefer)

@compatible_infer_mode()
def export_to_onnx(model, out_path, device=None):
    assert (os.path.exists(out_path), 'model not exists: {}'.format(model))
    assert(not os.path.exists(out_path), 'out_path already exists: {}'.format(out_path))
    
    #TODO: check support for half precision
    dev = get_device(device)
    #TODO only use device:0
    mode = attempt_load(model, map_location=dev)

    



class TorchExporter(object):
    def __init__(self, conf):
        self.model_w = conf.model
        self.out_type = conf.out_type
        self.out_path = conf.out_path
        self.with_sim = conf.simplify
        self.check_type(self.out_type)
    
    def check_type(self, o_type):
        if o_type not in ['onnx', 'torchscript']:
            raise ValueError('out_type should be onnx or torchscript, provided: {}'.format(o_type))
        
    def _export_to_torchscript(self):
        pass

    def _export_to_onnx(self):
        

    def _export_to_onnx_sim(self):
        pass

    def export(self):
        if self.out_type == 'onnx':
            if self.with_sim:
                self._export_to_onnx_sim()
            else:
                self._export_to_onnx()
        elif self.out_type == 'torchscript':
            self._export_to_torchscript()
        else:
            raise ValueError('out_type should be onnx or torchscript')
        

def parse_args():
    parser = argparse.ArgumentParser(description='Export model')
    parser.add_argument('--model', type=str, default='yolov3.pt', help='model path')
    parser.add_argument('--out_type', type=str, default='onnx', help='export to format')
    parser.add_argument('--out_path', type=str, default='./yolov3.onnx', help='export out path')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--device', type=str, default='cuda', help='device to use')
    parser.add_argument('--half', action='store_true', help='half precision')
    return parser.parse_args()




