#!/usr/bin/env python3

import numpy as np

class RecurrentLayer(object):
    def __init__(self, input_width, state_width, 
                 activator, lr):
        self.input_width = input_width
        self.state_width = state_width
        self.activator = activator
        self.lr = lr
        self.times = 0
        self.state_list = []
        