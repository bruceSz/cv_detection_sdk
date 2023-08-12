#!/usr/bin/env python3

import time
import torch
import numpy as np
from PIL import Image
import os

def pool_nms(hm, kernel=3):
    pad = (kernel - 1) // 2
    # n, c, h, w
    hmax = torch.nn.functional.max_pool2d(hm, (kernel, kernel), stride=1, padding=pad)
    # hmax.shape  ==  hm.
    # for hm points with local maximum
    keep = (hmax == hm).float()
    return hm * keep

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def resize_img(image, size, letterbox_img):
    iw, ih = image.size
    w, h = size
    if letterbox_img:
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw, nh), Image.BICUBIC)
def cvtColorGaurd(image):
    # hwc, where  c == 3
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

def singleton(cls):
    _instance = {}

    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]
    return inner


class MyTime(object):
    def __init__(self) -> None:
        pass

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        curr = time.time()
        print("elapsed time: ", curr - self.start)
        self.start = None


def weights_init(net, init_type='normal', init_gain=0.02):

    def init_func(m):

        class_name = m.__class__.__name__

        # conv layer
        # gain can be computed by torch.nn.init.calculate_gain
        # 1. in xavier uniform initializer, weight satisify (−a,a) uniform distribution
        #  where a = gain * sqrt(6/fan_in+fan_out)，

        if hasattr(m, 'weight') and class_name.find("Conv") != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, mean=0.0, std=init_gain)
            elif init_type == 'xavier':
                # 2. in xavier normal initializer, weight satisify normal,
                #   where mean=0, std = sqrt(2/fan_in+fan_out)
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                # 0-mean normal, std = sqrt(2/(1+a^2)*fan_in)
                torch.nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                # make the weight tensor orthogonal.
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
        elif class_name.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, mean=1, std=0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print("init network with %s" % init_type)

    # apply the initialization to each layer in the network
    net.apply(init_func)


def preprocess_input(img):
    # bgr -> rgb
    img = np.array(img, dtype=np.float32)[:, :, ::-1]
    # normalize
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    return (img / 255. - mean) / std

def resize_image(image, size, letterbox_img):
    iw, ih = image.size
    w, h = size
    if letterbox_img:
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def load_model_from_path(model, full_path):
    
    if os.path.isfile(full_path):
        model_dict = model.state_dict()
        device = model.parameters().__next__().device
        pretrained_dict = torch.load(full_path, map_location = device)
        load_key, no_load_key, tmp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                tmp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(tmp_dict)
        model.load_state_dict(model_dict)
    else:
        print("not existing weight file: ", full_path)