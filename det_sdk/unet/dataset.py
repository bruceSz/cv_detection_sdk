#!/usr/bin/env python3

import torch

import cv2
import os 
import glob

from torch.utils.data import Dataset

import random

class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        print("data root is: ", data_path)
        assert(os.path.exists(data_path))
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(self.data_path, 'image/*.png'))
        print("all images: ",len(self.imgs_path))

    def augment(self, img, flipCode):
        # use cv2.flip to flip the image
        #  1: flip horizontal
        #  0: flip vertical
        #  -1: flip both horizontally and vertically

        flip = cv2.flip(img, flipCode)
        return flip
    
    def __len__(self):
        return len(self.imgs_path)
    
    def __getitem__(self, idx):
        img_path = self.imgs_path[idx]

        label_path = img_path.replace('image','label')

        img = cv2.imread(img_path)
        label = cv2.imread(label_path)

        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])

        if label.max() > 1:
            label = label / 255.0

        flipCode = random.choice([-1,0,1,2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)

        return image, label
    
if __name__ == "__main__":
    isbi_dataset = ISBI_Loader('../data/ISBI/train')
    print("dataset no.",len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(isbi_dataset, batch_size=2, 
                                               shuffle=True)
    
    for image, label in train_loader:
        print(image.shape)
        print(label.shape)



