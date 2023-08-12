#!/usr/bin/env python3


import glob
import numpy as np
import torch
import os
import cv2
from model import UNet


def main():

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using dev: ", dev)
    net = UNet(n_channels=1, n_classes=1)

    net.to(device = dev)

    net.load_state_dict(torch.load("best_model.pth", map_location=dev))

    net.eval()

    tests_path = glob.glob('../data/ISBI/test/*.png')

    for test_path in tests_path:
        test_path = os.path.abspath(test_path)
        print("test_path: ", test_path)
        if ("_res.png" in test_path):
            continue
        save_res = test_path.split('.')[0] + '_res.png'
        print(save_res)
        img = cv2.imread(test_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("img shape: ", img.shape)
        img = img.reshape(1,1, img.shape[0], img.shape[1])
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.to(device = dev, dtype=torch.float32)

        pred = net(img_tensor)
        print("pred shape: ", pred.shape)
        pred = np.array(pred.data.cpu()[0])[0]

        print(pred.max(), pred.min())
        pred[pred >=0.5] = 100
        #pred[pred < 0.5] = 0
        cv2.imwrite(save_res, pred)
        


if __name__ == "__main__":
    main()