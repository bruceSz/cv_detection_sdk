#!/usr/bin/env python3

from perceptron import Perceptron

def ident(x):
    return x

def relu(x):
    return x if x > 0 else 0

class Linear(Perceptron):
    def __init__(self, input_num):
        # relu here is not working if output is negative, activattion layer will
        # make it 0, thus no gradient al all.
        Perceptron.__init__(self, input_num, relu)

def get_train_dataset():
    input_vecs = [
        [5],
        [3],
        [8],
        [1.4],
        [10.1]
    ]
    labels = [-5500, -2300, -7600, -1800, -11400]
    return input_vecs, labels

def train_linear_unit():
    l = Linear(1)
    input_vecs, labels = get_train_dataset()
    l.train(input_vecs, labels, 5, 0.01)
    return l

if __name__ == '__main__':
    linear_unit = train_linear_unit()
    print(linear_unit)
     # 测试
    print('Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4]))
    print('Work 15 years, monthly salary = %.2f' % linear_unit.predict([15]))
    print('Work 1.5 years, monthly salary = %.2f' % linear_unit.predict([1.5]))
    print('Work 6.3 years, monthly salary = %.2f' % linear_unit.predict([6.3]))