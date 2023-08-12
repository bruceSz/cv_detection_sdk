#! --*- coding: utf-8 -*-
#!/usr/bin/env python3

import numpy as np

class FCLayer(object):
    def __init__(self, input_size, output_size, activator):
        '''
        intput_size: size of input layer
        output_size: size of output layer
        activator: activation function
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        
        self.W = np.random.uniform(-0.1, 0.1,(output_size,input_size))
        self.b = np.random.randn(output_size,1)
        # clear output 
        self.output = np.zeros((output_size,1))

    def forward(self, x):
        '''
        forward propagation
        x.shape = input_size, ignore batch size here.
        '''
        self.input = x
        # y = a(W * w + b)
        self.output = self.activator.forward(
            np.dot(self.W, x) + self.b)
        
    def backward(self, delta_array):
        """
        backward propagation
        delta.shape = output_size
        """
        print(delta_array.shape)
        print(self.activator.backward(self.input).shape)
        print(self.W.T.shape)
        # * here means hadmard product
        # self.delta here it to compute delta wrt x(input).
        self.delta = self.activator.backward(self.input) * np.dot(
            self.W.T, delta_array
        )
        # need activator.backward?
        # refer: https://zhuanlan.zhihu.com/p/61863634
        self.W_grad = np.dot(delta_array, self.input.T)
        # refer: https://zhuanlan.zhihu.com/p/61863634
        self.b_grad = delta_array

    def update(self, lr):
        '''
        sgd to update w,b
        '''
        self.W += lr * self.W_grad
        self.b += lr * ( self.b_grad)



class Sigmoid(object):
    def forward(self, w_input):
        return 1.0/(1.0+np.exp(-w_input))
    
    def backward(self, w_output):
        return w_output * (1-w_output)
    


class Network(object):
    def __init__(self,layers):
        self.layers = []
        for i in range(len(layers) -1):
            self.layers.append(
                FCLayer(layers[i], layers[i+1], Sigmoid)
            )

    def predict(self, x):
        output = x
        for l in self.layers:
            l.forward(output)
            output = l.output

        return output
    
    def train(self, labels, data_set, rate, epoch):
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one(labels[d], data_set[i], rate)

    def train_one(self, label, data, rate):
        self.predict(data)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):
        delta = self.layers[-1].activator.backward(self.layers[-1].output) * (label-self.layers[-1].output)
        for l in self.layers[:-1]:
            l.backward(delta)
            delta = l.delta

        return delta
    
    def update_weight(self, rate):
        for l in self.layers:
            l.update(rate)

    def dump(self):
        for l in self.layers:
            l.dump()
        
    def loss(self, out, label):
        return 0.5 * ((label - out) **2).sum()
    
    def gradient_check(self, sample_ft, sample_label):

        self.predict(sample_ft)
        self.calc_gradient(sample_label)
        eps = 1e-4
        for fc in self.layers:
            for i in range(fc.W.shape[0]):
                for j in range(fc.W.shape[1]):
                    fc.W[i,j] += eps
                    output = self.predict(sample_ft)
                    err1 = self.loss(sample_ft, sample_label)
                    fc.W[i,j] -= 2*eps
                    output = self.predict(sample_ft)
                    err2 = self.loss(sample_ft, sample_label)
                    expect_grad = (err1 - err2) / (2 * eps)
                    fc.W[i,j] += eps
                    print("weights {}, {} expected {} - actual {}".format(i, j, expect_grad, fc.W_grad[i,j]))


def transpose(args):
    return list(map(
        lambda arg: list(map(
            lambda line: np.array(line).reshape(len(line), 1), arg
        ))
        , args
    ))


def Normalizer(object):
    def __init__(self):
        self.mask = [
            0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80
        ]

    def norm(self, n):
        data = list(map(lambda m:0.9 if n & m else 0.1, self.mask))
        return np.array(data).reshape(8,1)
    
    def denorm(self, vec):
        binary = list(map(lambda i: 1 if i> 0.5 else 0, vec[:,0]))
        for i in range(len(self.mask)):
            binary[i] = binary[i] * self.mask[i]
        return reduce(lambda x,y: x + y, binary)
    

def train_data_set():
    norm = Normalizer()
    dataset = []
    labels = []
    for i in range(0,256):
        n = norm.norm(i)
        dataset.append(n)
        labels.append(n)
    return labels, dataset

def correct_ratio(net):
    norm = Normalizer()
    correct = 0
    for i in range(256):
        if norm.denorm(net.predict(norm.norm(i))) == norm.denorm(i):
            correct += 1
    print("correct ratio: {}".format(correct/256 * 100))


def test():
    labels, dataset = transpose(train_data_set())
    net = Network([8,3,8])
    rate = 0.5
    batch = 20
    epoch = 10
    for i in range(epoch):
        net.train(labels, dataset, rate, batch)
        print("epoch: {}, loss: {}".format(i, net.loss(net.predict(dataset))))
        #learning rate decay each epoch devided by 2
        rate /=2
    correct_ratio(net)

def gradient_check():
    labels, dataset = transpose(train_data_set())
    net = Network([8,3,8])
    net.gradient_check(dataset[0], labels[0])
    return net