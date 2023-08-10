#!/usr/bin/env python3


import sys
import os
import time
import math
import numpy as np
from PIL import Image
import torch
import cv2
from torchvision import transforms
from torchvision import datasets

from det_sdk.common.utils import cvtColorGaurd
from det_sdk.common.utils import singleton
from det_sdk.common.utils import preprocess_input
from det_sdk.faster_rcnn.utils import get_classes

sys.path.append(os.path.dirname(os.path.abspath(sys.path[0])))


def gaussian2D(shape, sigma=1):
    m, n = [(ss -1 ) / 2. for ss in shape]
    #print("m: {}, n: {}".format(m,n))
    y, x = np.ogrid[-m:m+1, -n:n+1]
    #print("y: {} with shape: {}, x:{}, with shape: {}".format(y, y.shape,x, x.shape))

    #print(x+y)
    #h = 1/(2 * np.pi*sigma **2) *  np.exp(-(x*x + y*y)/(2 * sigma * sigma))  
    # here we don't multiply 1/(2 * np.pi*sigma **2) to make the max value of gaussian is 1
    h =  np.exp(-(x*x + y*y)/(2 * sigma * sigma))
    
    
    #print(h)
    old = h

    #print("all larger than 0",(h>0).all())
    
    #print("gaussian max is: ", h.max())
    h[h<np.finfo(h.dtype).eps * h.max()]  = 0
    #print("after filter with eps: ", h)
    #print("filter not working?",(old == h).all())
    return h



def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x,y = int(center[0]), int(center[1])
    #print(gaussian)

    height, width = heatmap.shape[0:2]
    # get real boundary around center, ideally it is (radius, radius)
    
    left, right = min(x,radius), min(width-x, radius+1)
    top, bottom = min(y, radius), min(height-y, radius +1)

    
    masked_hm = heatmap[y-top:y + bottom, x-left: x+right]
    masked_gaussian = gaussian[radius-top:radius + bottom, radius-left:radius+right]
    if min(masked_gaussian.shape) > 0 and min(masked_hm.shape) > 0:
        np.maximum(masked_hm, masked_gaussian*k, out=masked_hm)
    return heatmap



def centernet_gaussian_radius(det_size, min_overlap=0.7):
    """
        https://zhuanlan.zhihu.com/p/452632600
    """
    h, w = det_size

    a1 = 1
    b1 = (h + w)
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

def box_filter(box, dx, dy, nw, nh, iw, ih, w, h):
    np.random.shuffle(box)
    box[:, [0, 2]] = box[:, [0, 2]] * nw/iw + dx
    box[:,[1,3]] = box[:,[1,3]] * nh/ih + dy
    box[:,0:2][box[:,0:2]<0] = 0
    box[:,2][box[:,2] > w] = w
    
    
    idx = box[:, 3] > h
    #print("idx: ", idx)
    
    box[:, 3][idx] = h
    box_w = box[:,2] - box[:,0]
    box_h = box[:,3] - box[:,1]
    box = box[np.logical_and(box_w > 1, box_h > 1)]

    return box

class CIFar10_Dataset(object):
    def __init__(self) -> None:
        pass

    @classmethod
    def get_transform_loader(self):
        
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.226, 0.225, 0.224))
        ])

        transform_val = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406),(0.226, 0.225, 0.224))])
        
        return transform_train, transform_val
   
    #return train_dat, val_dat

    @classmethod
    def get_dataset(cls, info : dict, train_flag=True):
        # abstract into pre_process.py
        trans_train, trans_val = cls.get_transform_loader(info)

        train_dat = datasets.CIFAR10(root='./data', train=train_flag, 
                                     download=True, transform=trans_train)

        val_dat = datasets.CIFAR10(root='./data', train=train_flag,
                                     download=True, transform=trans_val)
    

        if train_flag:
            return train_dat
        else:
            return val_dat
      
        # train_loader = DataLoader(train_dat, batch_size=conf.batch_size, 
        #                         shuffle=True)
        
        # val_loader = DataLoader(val_dat, batch_size=conf.batch_size,
        #                         shuffle=False)
        
      

class VOC_SEG_Dataset(object):
    CLS_NAMES = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])
     
    VOC_COLORMAP = [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ]
    img_root = "/disk2/dataset/voc_11/VOCdevkit/VOC2012/JPEGImages"
    cls_label_root = "/disk2/dataset/voc_11/VOCdevkit/VOC2012/SegmentationClass"
    obj_label_root = "/disk2/dataset/voc_11/VOCdevkit/VOC2012/SegmentationObject"
    train_path = "/disk2/dataset/voc_11/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"
    val_path = "/disk2/dataset/voc_11/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"

    # bgr
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, model_info, train = False, is_cls_seg = True) -> None:
        if train:
            self.fname_flist = self.train_path
        else:
            self.fname_flist = self.val_path
        self.train = train
        
        if is_cls_seg:
            self.label_path = self.cls_label_root
        else:
            self.label_path = self.obj_label_root
        self.is_cls_seg = is_cls_seg

        self.class_names = self.CLS_NAMES
        self.num_classes = len(self.class_names)
        self.paths = []
        self.imgs = []
        self.model_info = model_info
        
        with open(self.fname_flist, "r") as f:
            for line in f.readlines():
                fname = line.strip() +".png"
                # seg label path
                fpath = os.path.join(self.label_path, fname)
                img = line.strip() + ".jpg"
                # raw image path.
                img_path = os.path.join(self.img_root, img)

                self.imgs.append(img_path)
                self.paths.append(fpath)
        self.length = len(self.paths)
        self.gray = False

    def __len__(self):
        return self.length
    


    def __getitem__(self, index):
        index = index % self.length

        image, label = self.get_seg_img_gt(
            self.imgs[index], self.paths[index], random=self.train)
        
        return image, label
    
    def convert_col_idx_map(self, label):
        h, w = label.shape[0:2]
        seg_mask  = np.zeros((h, w,len(self.VOC_COLORMAP)), dtype=np.float32)
        for l_idx, col in enumerate(self.VOC_COLORMAP):
            #l_idx is color index, label is color value [r,g,b]
            
            col = np.array(col, dtype=np.float32)
            tmp =np.all(label == col, axis=-1).astype(np.float32)
            if not np.all(tmp==0):
                print("col: ", col)
                print("seg mask check shape: ", tmp.shape)
                print("all o for tmp: ",np.all(tmp == 0))
            seg_mask[:,:,l_idx] = tmp
            print("all 0?:",np.all(seg_mask == 0))

        return seg_mask

    def get_seg_img_gt(self, img_path, label_path, random=False):
        assert(os.path.exists(img_path))
        print("img path: ", img_path)
        print("label path: ", label_path)
        in_shape  = self.model_info["input_shape"]
        img = Image.open(img_path)
        img = np.array(img, dtype=np.uint8)
        img = cv2.resize(img, (in_shape[1], in_shape[0]), interpolation=cv2.INTER_LINEAR)

        if self.gray:
            lb = Image.open(label_path)
            lb = np.array(lb, dtype=np.int32)
            lb[lb==255] = -1
        else:
            label = cv2.imread(label_path)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            label = cv2.resize(label, (in_shape[1], in_shape[0]), interpolation=cv2.INTER_NEAREST)
            lb = self.convert_col_idx_map(label)

            


        print("raw img shape: , ", img.shape)
        print("raw label shape: ", lb.shape)
        

        return self.transform(img, lb)



    def transform(self, img, lb):
        img = img[:,:, ::-1] # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        # transpose to chw.
        img = img.transpose(2,0,1)
        #print("img type: ", img.dtype)
        #print("label type: ", lb.dtype)
        img = img.astype(np.float32)
        lb = lb.astype(np.int32)
        
        return img, lb
    
    def untransform(self, img, lb):
        #img = img.numpy
        # transpose back to hwc
        img = img.transpose(1,2,0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:,:,::-1]
        #lb = lb.()
        return img, lb

def voc_fcn_dataset_collate(batch):
    imgs, labels = [], []

    for img, label in batch:
        print("shape of img: ", img.shape)
        imgs.append(img)
        labels.append(label)

    imgs = torch.from_numpy(np.array(imgs)).type(torch.FloatTensor)
    labels = torch.from_numpy(np.array(labels)).type(torch.FloatTensor)
    
    return imgs, labels



class VOC_center_Dataset(object):
    classes_path = "data/voc/classes_voc.txt"
    train_path = "data/voc/2007_train.txt"
    val_path = "data/voc/2007_val.txt"

    def __init__(self, model_info, train) -> None:

        input_shape, output_shape = model_info["input_shape"], model_info["output_shape"]
        # self.class_names, self.num_classes = self._get_class(self.classes_path)
        self.class_names, self.num_classes = get_classes(self.classes_path)
        # for centernet: output_shape = int(input_shape/4)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.train = train
        

        if self.train:
            with open(self.train_path, "r") as f:
                self.annotation_lines = f.readlines()[:100]
        else:
            with open(self.val_path, "r") as f:
                self.annotation_lines = f.readlines()[:100]

        self.length = len(self.annotation_lines)

        # with open(self.val_path,"r") as f:
        #     self.val_lines = f.readlines()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length

        image, box = self.get_random_data(
            self.annotation_lines[index], self.input_shape, random=self.train)

        #print("num_classes: ", type(self.num_classes))
        #print("output shape: ", self.output_shape)
        batch_hm = np.zeros(
            (self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)
        batch_wh = np.zeros(
            (self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_reg = np.zeros(
            (self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_reg_mask = np.zeros(
            (self.output_shape[0], self.output_shape[1]), dtype=np.float32)

        if len(box) != 0:
            #print("box shape: ", box.shape)
            boxes = np.array(box[:, :4], dtype=np.float32)

            #boxes[:, [0,2]]
            boxes[:, [0, 2]] = np.clip(
                boxes[:, [0, 2]] / self.input_shape[1] * self.output_shape[1], 0, self.output_shape[1]-1)
            boxes[:, [1, 3]] = np.clip(
                boxes[:, [1, 3]] / self.input_shape[0] * self.output_shape[0], 0, self.output_shape[0]-1)

        for i in range(len(box)):
            bbox = boxes[i].copy()
            cls_id = int(box[i, -1])

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

            if h > 0 and w > 0:
                radius = centernet_gaussian_radius(
                    (math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))

                ct = np.array([(bbox[0] + bbox[2]) / 2,
                              (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                batch_hm[:, :, cls_id] = draw_umich_gaussian(
                    batch_hm[:, :, cls_id], ct_int, radius)
                batch_wh[ct_int[1], ct_int[0]] = 1. * w , 1. * h

                batch_reg[ct_int[1], ct_int[0]] = ct - ct_int
                batch_reg_mask[ct_int[1], ct_int[0]] = 1

        image = np.transpose(preprocess_input(image), (2, 0, 1))
        return image, batch_hm, batch_wh, batch_reg, batch_reg_mask

    def rand(self, b=0, e=1):
        return np.random.rand() * (e - b) + b

    def get_random_data(self, anno_line, input_shape, jitter=.3, hue=.1, sat=.7, val=0.4,
                         random=True):
        line = anno_line.split()

        image = Image.open(line[0])
        image = cvtColorGaurd(image)

        iw, ih = image.size
        h, w = input_shape
        
        box = np.array([np.array(list(map(int,box.split(",")))) for box in line[1:]])
        

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, dtype=np.float32)


            if len(box) > 0:
                
                box = box_filter(box, dx, dy, nw, nh, iw, ih, w, h)

                

            return image_data, box
        
        new_ar = w/h * self.rand(1-jitter, 1+jitter) / self.rand(1-jitter, 1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        dx = int(self.rand(0,w-nw))
        dy = int(self.rand(0,h-nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image


        flip = self.rand() < .5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = np.array(image, dtype=np.uint8)
        if 0:
            r = np.random.uniform(-1, 1, 3) * [hue, sat, val]  + 1
            hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
            dtype = image_data.dtype

            #print("r: ", r.dtype)
            #print("dtype: {}".format(dtype))
            x = np.arange(0, 256,  dtype=r.dtype)
            lut_hue = ((x*r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x*r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
            
            assert(lut_hue.shape == lut_sat.shape == lut_val.shape )
            assert(lut_hue.shape[0] == 256)
            assert(lut_hue.data.contiguous)
            #h = 
            # s = 
            # v = cv2.LUT(val, lut_val)
            
            image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        if len(box) > 0:
            
            box = box_filter(box, dx, dy, nw, nh, iw, ih, w, h)
        
        return image_data, box

def centernet_dataset_collate(batch):
    imgs, batch_hms, batch_whs, batch_regs, batch_reg_masks = [], [], [], [], []

    for img, batch_hm, batch_wh, batch_reg, batch_reg_mask in batch:
        imgs.append(img)
        batch_hms.append(batch_hm)
        batch_whs.append(batch_wh)
        batch_regs.append(batch_reg)
        batch_reg_masks.append(batch_reg_mask)
    imgs = torch.from_numpy(np.array(imgs)).type(torch.FloatTensor)
    batch_hms = torch.from_numpy(np.array(batch_hms)).type(torch.FloatTensor)
    batch_whs = torch.from_numpy(np.array(batch_whs)).type(torch.FloatTensor)
    batch_regs = torch.from_numpy(np.array(batch_regs)).type(torch.FloatTensor)
    batch_reg_masks = torch.from_numpy(np.array(batch_reg_masks)).type(torch.FloatTensor)

    return imgs, batch_hms, batch_whs, batch_regs, batch_reg_masks


@singleton
class DataSetMgr(object):
    # VOC_DATASET_DIR = ""
    VOC_CENTERNET_DS =  "voc_centernet"
    VOC_SEG_DS = "voc_seg"
    CIFAR_10 = "cifar10"
    def __init__(self) -> None:
        self.date_ts = time.time()

    def get_dataset(self, name, info, train=False):
        if name == DataSetMgr.VOC_CENTERNET_DS:
            return VOC_center_Dataset(info, train), centernet_dataset_collate
        elif name == DataSetMgr.VOC_SEG_DS:
            return VOC_SEG_Dataset(info,train=train, is_cls_seg=True), voc_fcn_dataset_collate
        elif name == DataSetMgr.CIFAR_10:
            return  CIFar10_Dataset.get_dataset(info, train), None
        else:
            raise NotImplementedError("Unknown dataset: {}".format(name))
    

def main():
    dsm = DataSetMgr()
    print("dsm ts: ", dsm.date_ts)
    shapes = [[512,512], [int(512/4),int(512/4)]]
    voc_dataset = dsm.get_dataset("voc_centernet", shapes)
    print(voc_dataset.class_names)
    print(voc_dataset.num_classes)

def plot_gaussian():
    mat = np.zeros((7, 7),dtype=np.float64)
    x = gaussian2D((10, 10)) * 256
    x = x.astype(np.int32)
    print((x>0).all())
    mat = draw_umich_gaussian(heatmap=mat, center=(3,3), radius=3, k=1)
    #print(x)
    print(mat)

    #from framework import dataset_mgr 
# from framework.dataset_mgr import draw_umich_gaussian
# from framework.dataset_mgr import gaussian2D
# from IPython.display import display
# from PIL import Image
# import numpy as np
# mat = np.zeros((512, 512),dtype=np.int8)#
# print(mat.shape)
# x = gaussian2D((20,20))
# #print(x.shape)
# print(x)

def fcn_dataset_get():
    dsm = DataSetMgr()
    print("dsm ts: ", dsm.date_ts)
    mod_info = {
        "input_shape":[366, 500],
         "output_shape": [256, 256]
         }

    voc_dataset, _ = dsm.get_dataset("voc_seg", mod_info)
    print(voc_dataset.class_names)
    print(voc_dataset.num_classes)
    return voc_dataset




if __name__ == "__main__":
    #main()
    #plot_gaussian()#
    fcn_dataset_get()