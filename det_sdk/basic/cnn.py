#!/usr/bin/env python3

import numpy as np

from activators import IdentityActivator, ReluActivator

def get_patch(input, i, j, filter_width, filter_height, stride):
    '''
    return the patch to compute for this conn
       i: x axis step.
       j: y axis step.
       input: 
           2d: HW mat
           3d: CHW mat
    '''
    start_i = i * stride
    start_j = j * stride
    if input.ndim ==2:
        return input[start_i: start_i + filter_width, start_j: start_j + filter_height]
    elif input.ndim ==3:
        return input[:,start_i: start_i + filter_width, start_j: start_j + filter_height]
    
# return max value in
def get_max_index(array):
    max_i = 0
    max_j = 0
    max_val = array[0,0]
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i,j] > max_val:
                max_i = i
                max_j = j
                max_val = array[i,j]

    return max_i, max_j

def conv(input, kernel, output, stride, bias):
    #
    # 2d case.
    output_width = output.shape[-1]
    output_height = output.shape[-2]
    kernel_width = kernel.shape[-1]
    kernel_height = kernel.shape[-2]

    for i in range(output_height):
        for j in range(output_width):
            output[i][j] = (get_patch(input, i, j, kernel_width, kernel_height, stride) * kernel).sum()\
                + bias




def padding(input, zp):

    if zp == 0:
        return input
    
    else:
        if input.ndim == 3:
            input_width = input.shape[2]
            input_height = input.shape[1]
            input_depth = input.shape[0]
            padded_array = np.zeros((input_depth, 
                                     input_height + 2* zp, 
                                     input_width + 2*zp))
            
            padded_array[:,zp: zp + input_height,zp: zp + input_width] = input

            return padded_array
        elif input.ndim == 2:
            input_width = input.shape[1]
            input_height = input.shape[0]
            padded_array = np.zeros((input_height + 2 * zp, input_width + 2 * zp))

            padded_array[zp:zp + input_height, zp:zp + input_width] = input

            return padded_array

def element_wise(input, op):
    for i in np.nditer(input, op_flags=['readwrite']):
        i[...] = op(i)


class Filter(object):
    def __init__(self, width, height, depth):
        self.weights = np.random.uniform(-1e-4, 1e-4, (depth, height, width))
        self.bias = 0
        self.weights_grad = np.zeros(
            self.weights.shape)
        self.bias_grad = 0

    def __repr__(self) -> str:
        return f"Filter({self.weights.shape}, {self.bias})".format(self.weights.shape, self.bias)
    
    def get_weights(self) -> np.ndarray:
        return self.weights
    
    def get_bias(self) -> float:
        return self.bias
    
    def update(self, lr):
        self.weights += lr * self.weights_grad
        self.bias += lr * self.bias_grad

    
class ConvLayer(object):
    def __init__(self, in_width, in_height,
                 channels, filter_width, filter_height, filter_num,  
                 zero_padding, stride, activator, lr) -> None:
        self.in_width = in_width
        self.in_height = in_height
        self.channels = channels
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_num = filter_num
        self.zero_padding = zero_padding
        self.stride = stride
        self.output_width = ConvLayer.calculate_output_size(self.in_width, 
                                                            self.filter_width, 
                                                            self.zero_padding,
                                                            stride)
        self.output_height = ConvLayer.calculate_output_size(self.in_height, 
                                                            self.filter_height, 
                                                            self.zero_padding,
                                                            stride)
        
        self.output = np.zeros((self.filter_num, self.output_height, self.output_width))
        self.filters = []
        for i in range(filter_num):
            self.filters.append(Filter(filter_width, filter_height, channels))
        self.activator = activator
        self.lr = lr


    def forward(self, input):
        self.input_array = input
        self.padded_input_array = padding(input, self.zero_padding)
        for f in range(self.filter_num):
            filter = self.filters[f]
            conv(self.padded_input_array, filter.get_weights(), self.output[f],
                  self.stride, filter.get_bias())
        element_wise(self.output, self.activator.forward)

    def backward(self, input, sensitivity_arr, activator):
        self.forward(input)
        self.bp_sensitivity_map(sensitivity_arr, activator)
        self.bp_gradient(sensitivity_arr)

    
    def update(self):
        for filter  in self.filters:
            filter.update(self.lr)


    #
    # sensitivity map is also named as delta map from next layer, we need
    # to back-propagate the delta map to previous layer.
    #
    def bp_sensitivity_map(self, sensitivity_arr, activator):
        expanded_arr = self.expand_sensitivity_map(sensitivity_arr)
        expanded_width = expanded_arr.shape[2]
        zp = int((self.in_width + self.filter_width -1 - expanded_width) / 2)

        padded_arr = padding(expanded_arr, zp)
        self.delta_array = self.create_delta_array()

        for f in range(self.filter_num):
            filter = self.filters[f]
            # rotate the weights counter clockwise 180 = 90 * 2
            flipped_weights = np.array(list(map(
                lambda i: np.rot90(i,2),
                filter.get_weights()
            )))
            delta_array = self.create_delta_array()
            # shape[0] is channel_num
            for d in range(delta_array.shape[0]):
                conv(padded_arr[f], flipped_weights[d], delta_array[d],1, 0)
            # need to sum delta_array generated by this filter. (backward propagation)
            self.delta_array += delta_array

        derivative_array = np.array(self.input_array)
        element_wise(derivative_array, activator.backward)
        self.delta_array *= derivative_array


    def bp_gradient(self, sensitivity_arr):
        expanded_arr = self.expand_sensitivity_map(sensitivity_arr)
        for f in range(self.filter_num):
            filter = self.filters[f]
            for d in range(filter.weights.shape[0]):
                conv(
                    self.padded_input_array[d],
                    expanded_arr[f],
                    filter.weights_grad[d],
                    1, 0

                )           
            filter.bais_grad = expanded_arr[f].sum()
                
        
    def create_delta_array(self):
        return np.zeros((self.channels, self.in_height, self.in_width))

    def expand_sensitivity_map(self, sensitivity_arr):
        # TODO, figure it out.
        depth = sensitivity_arr.shape[0]
        #(h - k_h + 2p)/stripe + 1 ？
        expanded_width = (self.in_width - 
                          self.filter_width + 2 * self.zero_padding + 1)
        expanded_height = (self.in_height - 
                           self.filter_height + 2 * self.zero_padding + 1)
        expand_array = np.zeros((depth, expanded_height, expanded_width))

        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i* self.stride
                j_pos = j* self.stride
                expand_array[:, i_pos, j_pos] = sensitivity_arr[:, i, j]
        print("shape of sensitivity_arr:", sensitivity_arr.shape)
        print("shape of expand_array:", expand_array.shape)
        return expand_array


    @staticmethod
    def calculate_output_size(in_width, filter_width, zero_padding, stride):
        return int((in_width - filter_width + 2 * zero_padding)/stride + 1)
        

class MaxPoolingLayer(object):
    def __init__(self, in_width, in_height, channels, filter_width, filter_height, stride) -> None:
        self.input_width = in_width
        self.input_height = in_height
        self.channels = channels
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.stride = stride
        self.output_width = ((self.input_width - filter_width)/self.stride + 1)
        self.output_height = ((self.input_height - filter_height)/self.stride + 1)

        self.output_array = np.zeros((self.channels, self.output_height, self.output_width))


    def forward(self, input):
        for d in range(self.channels):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    self.output_array[d, i, j] = (get_patch(input[d], i, j, 
                                                            self.filter_width, 
                                                            self.filter_height,
                                                            self.stride).max())
                    

    def backward(self, input, sesitivity_arr):
        self.delta_array = np.zeros((input.shape))
        for d in range(self.channels):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    patch_array = get_patch(
                        input[d], i, j,
                        self.filter_width, self.filter_height,
                        self.stride
                    )
                    # TODO, memorize this to save computation.
                    k,l = get_max_index(patch_array)
                    self.delta_array[d,i * self.stride + k,
                                     j * self.stride + l] = sesitivity_arr[d,i,j]

                    



def init_test():
    a = np.array(
        [[[0,1,1,0,2],
          [2,2,2,2,1],
          [1,0,0,2,0],
          [0,1,1,0,0],
          [1,2,0,0,2]],
           [[1,0,2,2,0],
          [0,0,0,2,0],
          [1,2,1,2,1],
          [1,0,0,0,0],
          [1,2,1,1,1]],
         [[2,1,2,0,0],
          [1,0,0,1,0],
          [0,2,1,0,1],
          [0,1,2,2,2],
          [2,1,0,0,1]]])
    
    b = np.array(
        [[[0,1,1],
          [2,2,2],
          [1,0,0]],
          [[1,0,2],
           [0,0,0],
           [1,2,1]]
        ]
    )
                        
    cl = ConvLayer(in_width =5, in_height  = 5,
                   channels = 3, filter_width= 3, filter_height = 3,
                   filter_num = 2,
                   zero_padding = 1, stride = 2,activator = IdentityActivator(), lr =  0.001)
    cl.filters[0].weights = np.array( [[[-1,1,0],
          [0,1,0],
          [0,1,1]],
         [[-1,-1,0],
          [0,0,0],
          [0,-1,0]],
         [[0,0,-1],
          [0,1,0],
          [1,-1,-1]]], dtype=np.float64)
    
    cl.filters[0].bias=1
    cl.filters[1].weights = np.array(
        [[[1,1,-1],
          [-1,-1,1],
          [0,-1,1]],
         [[0,1,0],
         [-1,0,-1],
          [-1,1,0]],
         [[-1,0,0],
          [-1,0,1],
          [-1,0,0]]], dtype=np.float64)
    return a, b, cl



def test():
    a,b,cl = init_test()
    cl.forward(a)
    print(cl.output)



def test_bp():
    a,b,cl = init_test()
    cl.backward(a,b, IdentityActivator())
    cl.update()
    print(cl.filters[0])
    print(cl.filters[1])

if __name__ == '__main__':
    test()
    test_bp()

