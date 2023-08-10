#!/usr/bin/env python3

import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from sklearn.model_selection import train_test_split

def pre_process_dc():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomCrop(204),
        transforms.ToTensor(),
        transforms.Normalize([0,0,0],[1,1,1])
    ])

    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0,0,0],[1,1,1])
    ])

    return transform_train, transforms_test

def get_dirs():
    train_dir = "/disk2/dataset/dogs_vs_cats/train"
    test_dir = "/disk2/dataset/dogs_vs_cats/test"
    return train_dir, test_dir

def get_dir_by_mode(mode):
    train_dir, test_dir = get_dirs()
    map = {'train': train_dir,'val':train_dir, 'test': test_dir}
    return map[mode]

class_to_int = {'cat': 0, 'dog': 1}
int_to_class = { 0: 'cat', 1: 'dog'}

class DogCatDataSet(Dataset):
    def __init__(self, imgs, class_to_int, mode='train', transforms=None):
        super(DogCatDataSet, self).__init__()
        self.imgs = imgs
        self.class_to_int = class_to_int
        self.mode = mode
        self.transforms = transforms


    def __getitem__(self, idx):
        image_name = self.imgs[idx]
        dir_path = get_dir_by_mode(self.mode)
        full_path = os.path.join(dir_path, image_name)
        img = Image.open(full_path)
        img = img.resize((224, 224))
        if self.mode == 'train' or self.mode == 'val':
            # img_name: cat.111.png
            label = self.class_to_int[image_name.split('.')[0]]
            label = torch.tensor(label, dtype=torch.float32)

            img = self.transforms(img)
            return img, label
        
        elif self.mode == 'test':
            img = self.transforms(img)
            return img
        
    def __len__(self):
        return len(self.imgs)
    


def get_dataloader():
    train_dir,test_dir = get_dirs()
    train_imgs = os.listdir(train_dir)
    test_imgs = os.listdir(test_dir)
    train_transform, test_transform = pre_process_dc()

    train_imgs, val_imags = train_test_split(train_imgs, test_size=0.2)
    train_dataset = DogCatDataSet(train_imgs, class_to_int, mode='train', transforms=train_transform)
    val_dataset = DogCatDataSet(val_imags, class_to_int, mode='val', transforms=test_transform)
    test_dataset = DogCatDataSet(test_imgs, class_to_int, mode='test', transforms=test_transform)


    train_data_loader = DataLoader(
        dataset = train_dataset, 
        num_workers = 4, 
        batch_size = 16,
        shuffle = True,
    )

    val_data_loader = DataLoader(
        dataset = val_dataset, 
        num_workers = 4, 
        batch_size = 16,
        shuffle = True,
    )

    test_data_loader = DataLoader(
        dataset = test_dataset, 
        num_workers = 4, 
        batch_size = 16,
        shuffle = True,

    )

    return train_data_loader, val_data_loader, test_data_loader