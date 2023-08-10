#!/usr/bin/env python3

import argparse

def create_parser():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--faster_rcnn', help='train faster rcnn')
    #parser.add_argument('--centernet', help='train centernet')
    parser.add_argument('--model_name',default="fcn32",help='model: faster_rcnn|centernet|fcn32|ssd')
    parser.add_argument('--backbone', default="resnet50", help='backbone to be used')
    parser.add_argument('--dataset', default="voc_seg",help='dataset to be used: voc_centernet|voc_seg|voc_2007')
    parser.add_argument('--epochs', help='number of epochs')
    parser.add_argument('--batch_size', help='batch size')
    parser.add_argument('--multi_gpu', help='train with multi-gpu')
    parser.add_argument('--fp16', help='use fp16 training')
    parser.add_argument('--pretrained', help='pretrained model')
    return parser
