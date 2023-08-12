#!/usr/bin/env python3

from functools import reduce

class Perceptron(object):
    def __init__(self, input_num, act):
        '''
        set input_num and activation
        '''
        self.activator = act
        print("input vector shape is: ", input_num)
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0

    def __str__(self) -> str:
        return "weights\t: {}\nbias\t: {}".format(self.weights, self.bias)
    
    def predict(self, input_data):
        #1. map two list and compute with lambda
        #2. reduce a list 
        #3. add bias
        #4. pass to activator 
        return self.activator(reduce(lambda a,b: a+b, list(map(lambda x,w:x*w, input_data, self.weights)), 0.0) + self.bias)
    
    def train(self, input_data, labels, iterations, rate):
        # perceptron train.
        for i in range(iterations):
            print("epoch {}/{}".format(i+1, iterations))
            
            self._one_iter(input_data, labels, rate)

    def _one_iter(self, input_data, labels, rate):
        samples = zip(input_data, labels)
        for (ind, label) in samples:
            output = self.predict(ind)
            self.update_weights(ind, output, label, rate)

    def update_weights(self, input_data, output, label, rate):
        delta = label - output
        # gradient descent for relu activation
        if output > 0:
            delta = delta
        else:
            delta = 0
        print("delta: {}".format(delta))
        print(self)
        # w is parameter to be updated
        # delta is learning rate parameter
        # delta is loss
        # delta_w = rate * delta * x
        
        # shape of weights will be updated.
        self.weights = list(map(lambda x,w: w+rate*delta*x, input_data, self.weights))
        self.bias += rate * delta


def f(x):
    return x if x > 0 else 0

def get_training_dataset():
    input_vecs = [
        [1,1],
        [0,0],
        [1,0],
        [0,1]
    ]
    labels = [1,0,0,0]
    return input_vecs, labels

def train_and_perception():

    p = Perceptron(10, f)
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    return p

if __name__ == '__main__':
    p = train_and_perception()
    print(p)
    print("1 and 1= {}".format(p.predict([1,1])))
    print("0 and 0= {}".format(p.predict([0,0])))
    print("1 and 0 = {}".format(p.predict([1,0])))
    print("0 and 1 = {}".format(p.predict([0,1])))
