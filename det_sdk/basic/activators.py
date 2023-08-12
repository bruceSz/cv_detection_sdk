#!/usr/bin/env python3

import numpy as np

class ReluActivator(object):
    def forward(self, x):
        return max(0, x)
    
    def backward(self, x):
        return 1 if x > 0 else 0
    

class IdentityActivator(object):
    def forward(self, x):
        return x
    
    def backward(self, x):
        return 1
    

class SigmoidActivator(object):

    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def backward(self, output):
        return (1- output) * output
    

class TanhActivator(object):
    def forward(self, x):
        return 2.0 / (1.0 + np.exp(-2.0 * x)) - 1.0
    
    def backward(self, output):
        return 1 - output * output