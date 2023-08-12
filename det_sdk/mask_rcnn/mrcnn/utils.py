#!/usr/bin/env python3
"""
Mask R-CNN
The main model implementation.

Copyright (c) 2023 X, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by bruceSz
"""



def log(text, array = None):
    """Print a text message
        If a Numpy array is provided, the shape, min, max values are printed as well.
    """

    if array is not None:
        text = text.ljust(80)
        text += ("shape: {:20} ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f} max: {:10.5f} ".format(array.min(), array.max()))
        else:
            text += ("min: {:10.5f} max: {:10.5f} ".format("", ""))
        text += (" {} ".format(array.dtype))
    print(text)