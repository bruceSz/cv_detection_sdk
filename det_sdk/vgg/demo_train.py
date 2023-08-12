#!/usr/bin/env python

import torch
import os
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim

from framework.backbone_train_with_cifar import train_loop

from samples.pytorch.vgg.model import VGG16_CF
from dataset import get_loaders
from framework.train_config import CIFARConfig as Config


def main():
    conf = Config()
    train_loader, val_loader = get_loaders(conf)
    train_loop(conf, VGG16_CF(), train_loader, val_loader, 'vgg16_cf')


# def train_loop():
#     conf = Config()
#     train_loader, val_loader = get_loaders(conf)
    
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     model = VGG16_CF().to(device)

#     criterion = torch.nn.CrossEntropyLoss()

#     opt = optim.SGD(model.parameters(), lr=conf.lr, momentum=0.9, weight_decay=5e-4)
#     schedule = optim.lr_scheduler.StepLR(opt, step_size=conf.step_size, gamma=conf.gamma)


#     loss_list = []

#     for epoch in range(conf.epoch_n):
#         ww = 0.
#         r_loss = 0.
#         # train each epoch
#         for i , (inputs, labels) in enumerate(train_loader, 0):
#             inputs, labels = inputs.to(device), labels.to(device)

#             opt.zero_grad()

#             outputs = model(inputs)

#             loss = criterion(outputs, labels)

#             loss.backward()

#             opt.step()

#             r_loss += loss.item()
#             loss_list.append(loss.item())

#             if (i+1) % conf.num_print == 0:
#                 print("epoch: %d, batch: %d, loss: %.3f" % (epoch+1, i+1, r_loss/conf.num_print))
#                 r_loss = 0.

#         lr_1 = opt.param_groups[0]['lr']
#         print("epoch: %d, lr: %.6f" % (epoch+1, lr_1))
#         schedule.step()
    
#     torch.save(model.state_dict(), os.path.join(conf.model_dir, 'vgg16_epoch_' + str(conf.epoch_n + 1) + '.pth'))


if __name__ == "__main__":
    main()

    
