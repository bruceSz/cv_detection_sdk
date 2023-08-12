#!/usr/bin/env python3

import random
from functools import reduce
from numpy import exp

def sigmoid(x):
    return 1.0 / (1.0 + exp(-inX))



class Node(object):
    def __init__(self, layer_idx, node_idx):

        self.layer_idx = layer_idx
        self.node_idx = node_idx
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        self.output = output

    def append_downstream_connection(self, conn):
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        self.upstream.append(conn)

    def calc_output(self):
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
        self.output = sigmoid(output)
    
    def calc_hidden_delta(self):
        down_strm_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight, self.downstream, 0
        )
        # output * ( 1- self.output) is gradient of sigmoid.
        self.delta = self.output * (1.0 - self.output) * down_strm_delta


    def calc_output_delta(self, label):
        self.delta = self.output * (1.0 - self.output) * (label - self.output)

    
    def __str__(self):
        '''
            node info.
        '''
        node_str = 'layer_idx: {}, node_idx: {}, output {}, delta: {}'.format(
            self.layer_idx, self.node_idx, self.output, self.delta
        )
        down_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        up_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n' + down_str + '\n' + up_str
    

class ConstNode(object):
    def __init__(self, layer_index, node_idx) -> None:
        self.layer_idx = layer_index
        self.node_idx = node_idx
        self.downstream = []
        self.output = 1

    def append_downstream_connection(self, conn):
        self.downstream.append(conn)

    def calc_hidden_delta(self):
        down_delta = reduce(lambda ret, conn: ret + conn.downstream_node.delta * conn.weight, self.downstream, 0)
        # TODO: why need self.output * ( 1- self.output) ?
        self.delta = self.output * ( 1 - self.output) * down_delta

    def __str__(self):
            '''
                node info.
                '''
            node_str = "layer_idx: {}, node_idx: {}, output 1".format(self.layer_idx, self.node_idx)
            down_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
            return node_str + '\n' + down_str


class Layer(object):
    def __init__(self, layer_idx, node_count):
        '''
        init a layer
        layer_idx: layer index
        node_count: node count in this layer
        '''

        self.layer_idx = layer_idx
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_idx, i))
        #ConstNode here is bias?
        self.nodes.append(ConstNode(layer_idx, node_count))

    
    def set_output(self, data):
        
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        # calculate each node's output(based on it's input)
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        for node in self.nodes:
            print(node)

    
class Connection(object):
    def __init__(self, upstream_node, downstream_node):
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1, 0.1)
        self.gradient = 0.0

    def calc_gradient(self):
        self.gradient = self.downstream_node.delta * self.upstream_node.output

    def get_gradient(self):
        return self.gradient
    
    def update_weight(self, learning_rate):
        self.calc_gradient()
        self.weight += learning_rate * self.gradient


    def __str__(self):
        return 'layer_idx: {}, node_idx: {}, to layer_idx: {}, node_idx: {}'.format(
            self.upstream_node.layer_idx, self.upstream_node.node_idx,
            self.downstream_node.layer_idx, self.downstream_node.node_idx)

class Connections(object):
    def __init__(self):
        self.connections = []
    def add_connection(self, conn):
        self.connections.append(conn)
    
    def dump(self):
        for c in self.connections:
            print(c)


class Network(object):
    def __init__(self, layers):
        self.connections = Connections()
        self.layers = []
        layer_c = len(layers)
        node_count  = 0
        for i in range(layer_c):
            self.layers.append(Layer(i, layers[i]))
        for layer in range(layer_c - 1):
            # for fc, connections are cartesian product of up nodes and down nodes.
            connections = [Connection(up_node, down_node)
                           for up_node in self.layers[layer].nodes
                           for down_node in self.layers[layer + 1].nodes[:-1]]
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)

        


    def train(self, labels, dataset, rate, iter):
        '''
        '''
        for i in range(iter):
            for d in range(len(dataset)):
                self.train_one_sample(labels[d], dataset[d], rate)

    
    def train_one_sample(self, label, data, rate):
        self.predict(data)
        self.calc_delta(label)
        self.update_weights(rate)

    def predict(self, data):
        self.layers[0].set_output(data)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return list(map(lambda node: node.output, self.layers[-1].nodes[:-1]))
    
    def dump(self):
        for l in self.layers:
            l.dump()

    def update_weights(self, rate):

        for l in self.layers[:-1]:
            for node in l.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def calc_gradient(self):
        for l in self.layers[:-1]:
            for node in l.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def get_gradient(self, label, sample):
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()



    def calc_delta(self, label):
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_delta(label[i])
        # compute from last layer, backward
        for l in self.layers[-2::-1]:
            for node in l.nodes:
                node.calc_hidden_delta()

# refer: https://cuijiahua.com/blog/2018/11/dl-9.html
def gradient_check(network, sample, label):
    '''
    '''

    network_error = lambda vec1, vec2: \
        0.5 * reduce(lambda a,b: a+b, map(lambda v:(v[0] - v[1]) * (v[0]-v[1]),zip(vec1, vec2)))
    
    network.get_gradient(label, sample)

    for conn in network.connections.connections:
        real_gradient = conn.get_gradient()
        eps = 0.0001
        conn.weight += eps

        err1 = network_error(network.predict(sample), label)

        conn.weight -= 2* eps
        err2 = network_error(network.predict(sample), label)

        expected_gradient = (err1 - err2) / (2 * eps)

        print("expected_gradient: {}, real_gradient: {}".format(expected_gradient, real_gradient))


if __name__ == '__main__':
    gradient_check()
              
    