#!/usr/bin/env python3

import os
import torch
import scipy
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from matplotlib import pyplot as plt

class LossHistory(object):
    def __init__(self, log_dir, model ):#input_shape):
        self.log_dir  = log_dir
        self.losses = []

        self.val_loss = []

        os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir)

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir,"epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")

        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar("loss", loss, epoch)
        self.writer.add_scalar("val_loss", val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))
        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label = 'train loss')
        plt.plot(iters, self.val_loss, 'blue', linewidth=2, label = 'val loss')
        try:
            if len(self.losses) < 25:
                num = 15
            else:
                num = 15
            # use 3-order polynomial to fit num length window signal and filter out low
            # frequency signal.
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2, label='smoth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), 'purple', linestyle='--', linewidth=2, label='smoth val loss')


        except:
            pass

        plt.grid(True)      
        plt.xlabel('Epoch')      
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")
                






