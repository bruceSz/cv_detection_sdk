#!/usr/bin/env python3

"""
Mask R-CNN
The main model implementation.

Copyright (c) 2023 X, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by bruceSz
"""

import math
import numpy as np


def compute_bb_shapes(config, img_shape):
    """
    Compute width and height of each stage of bb network
    Returns:
      [N,(height, width)]. Where N is the number of stages
    """

    if callable(config.BACKBONE):
        return config.BACKBONE(img_shape)
    
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array([
        [int(math.ceil(img_shape[0]/stride)),
         int(math.ceil(img_shape[1]/stride))]
        for stride in config.BACKBONE_STRIDES])

def main():
    pass
if  __name__ == "__main__":
    main()