#!/usr/bin/env python3

import time

import numpy as np
import torch.optim as optim
import torch

import torchvision.transforms as transforms
from torchvision.models import resnet50

#from common.config import Config
from torch_model import TrainContext

import dc_dataset

""" def pre_process_fns():
    transform_train = transforms.Compose([
        transforms.Resize(224, 224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
    ])
    transform_test  = transforms.Compose([
        transforms.Resize(224, 224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
    ])

    return transform_train, transform_test
 """



def accuracy(preds, gts):
    import numpy as np
    preds = [1 if preds[i] >= 0.5 else 0  for i in range(len(preds))]

    acc = [1 if preds[i] == gts[i] else 0 for i in range(len(preds))]

    assert(len(preds) == len(gts))

    acc = np.sum(acc) / len(preds)
    return acc * 100




class Trainer(object):
    def __init__(self) -> None:
        self.context = TrainContext()
        self.train_dl, self.valid_dl, self.test_dl = dc_dataset.get_dataloader()
        

    def train_one(self, train_loader):
        
        epoch_loss = []
        epoch_acc = []
        start_t = time.time()

        
        for imgs, labels in train_loader:
            # imgs: one batch of images
            # labels: one batch of labels
            imgs = imgs.to(self.context.device)
            labels = labels.to(self.context.device)
            labels = labels.reshape(labels.shape[0], 1)
            
            # reset grads
            self.context.optimizer.zero_grad()

            # forward
            preds = self.context.model(imgs)

            _loss = self.context.criterion(preds, labels)
            loss = _loss.item()
            epoch_loss.append(loss)

            # compute accuracy
            acc = accuracy(preds, labels)
            epoch_acc.append(acc)
            
            # backward
            _loss.backward()
            self.context.optimizer.step()
            print("acc this batch:", np.mean(epoch_acc))
            print("loss this batch:", np.mean(epoch_loss))
        end_t = time.time()
        total_t = end_t - start_t
        epoch_acc = np.mean(epoch_acc)
        epoch_loss = np.mean(epoch_loss)
        self.context.train_logs['loss'].append(epoch_loss)
        self.context.train_logs['acc'].append(epoch_acc)
        self.context.train_logs['time'].append(total_t)

        return epoch_loss, epoch_acc, total_t

    def val_one(self, val_loader, best_val_acc):
        epoch_loss = []
        epoch_acc = []
        start_t = time.time()

        for imgs, labels in val_loader:
            imgs = imgs.to(self.context.device)
            labels = labels.to(self.context.device)
            labels = labels.reshape(labels.shape[0], 1)

            # forward
            preds = self.context.model(imgs)

            # calculate accuracy and loss
            _loss = self.context.criterion(preds, labels)
            loss = _loss.item()
            epoch_loss.append(loss)

            # compute accuracy
            acc = accuracy(preds, labels)
            epoch_acc.append(acc)

        end_t = time.time()
        total_t = end_t - start_t

        epoch_acc = np.mean(epoch_acc)
        epoch_loss = np.mean(epoch_loss)

        self.context.val_logs['loss'].append(epoch_loss)
        self.context.val_logs['acc'].append(epoch_acc)
        self.context.val_logs['time'].append(total_t)

        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            torch.save(self.context.model.state_dict(), self.context.model_save_path)

        return epoch_loss, epoch_acc, total_t, best_val_acc


    def train(self):
        best_val_acc = 0
        for epoch in range(self.context.epochs):
            loss, acc, _t = self.train_one(self.train_dl)
            print("Training:")
            print("Epoch:", epoch, "Loss:", loss, "Accuracy:", acc, "Time:", _t)

            loss, acc, _t, best_val_acc = self.val_one(self.valid_dl, best_val_acc)
            print("Validation:")
            print("Epoch:", epoch, "Loss:", loss, "Accuracy:", acc, " best acc: ", best_val_acc, "Time:", _t)




if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()