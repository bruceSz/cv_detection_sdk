#!/usr/bin/env python3

import torch
from utils import check_version



def compatible_infer_mode(torch_19 = check_version(torch.__version__, "1.9.0")):
    def dec(fn):
        return (torch.inference_mode if torch_19 else torch.no_grad())(fn)
    return dec