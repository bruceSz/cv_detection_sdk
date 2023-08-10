#!/usr/bin/env python3

import sys
import os
import math
import numpy as np
import torch
from tqdm import tqdm
from torchvision.ops import nms
from torch import nn
from PIL import Image
import shutil
sys.path.append(os.path.dirname(os.path.abspath(sys.path[0])))

from matplotlib import pyplot as plt
from backbones.resnet50 import resnet50
from centernet.resnet50_adaptor import resnet50_Decoder, resnet50_Head
from common.utils import cvtColorGaurd
from common.utils import  preprocess_input
from common.utils import resize_image
from common.utils import pool_nms

class EvalCallback(object):
    def __init__(self, net, backbone, input_shape, class_names,
                 num_class,val_lines, log_dir, cuda,
                 map_out = "./tmp_map_out", max_boxes = 100, confidence=0.05,
                 nms=True, nms_iou = 0.5, letterbox_img = True,
                 MINOVERLAP=0.5, eval_flag = True, period=1) -> None:
        
        super(EvalCallback, self).__init__()
        
        self.net = net
        self.backbone = backbone
        self.input_shape = input_shape
        self.class_names = class_names
        self.num_class = num_class
        self.val_lines = val_lines
        self.log_dir = log_dir
        self.cuda = cuda
        self.map_out = map_out
        self.max_boxes = max_boxes
        self.confidence = confidence
        self.nms = nms
        self.nms_iou = nms_iou
        self.letterbox_img = letterbox_img
        self.MINOVERLAP = MINOVERLAP
        self.eval_flag = eval_flag
        self.period = period
        
        self.maps = [0]
        self.epoches = [0]
        if self.eval_flag:
            with open(self.log_dir + "/epoch_map.txt", "a") as f:
                f.write(str(0))
                f.write("\n")

    def centernet_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        box_yx = box_xy[...,::-1]
        box_hw = box_wh[...,::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            #TODO？
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset = (input_shape - new_shape)/2./input_shape
            scale = input_shape/new_shape

            box_yx = (box_yx - offset) * scale
            box_hw *= scale
        
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[...,0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        # TODO?
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)

    def postprocess(self, prediction, need_nms, img_shape, input_shape, letterbox_img,  nms_iou):
        output = [None for _ in range(len(prediction))]

        for i, image_pred  in enumerate(prediction):
            detections = prediction[i]
            if len(detections) == 0:
                continue
            # 0:4 x1,y1,x2,y2, 4:5 score, 5:6 class
            unique_labels = detections[:, -1].cpu().unique()
            if detections.is_cuda:
                unique_labels = unique_labels.cuda()
                #TODO ? do we realy need this?
                detections = detections.cuda()

            for c in unique_labels:
                det_class = detections[detections[:, -1] == c]
                if need_nms:
                    keep = nms(
                        det_class[:,:4],
                        det_class[:,4],
                        nms_iou
                    )
                    max_det = det_class[keep]
                else:
                    max_det = det_class
                output[i] = max_det if output[i] is None else torch.cat((output[i], max_det))
        
            if output[i] is not None:
                output[i] = output[i].cpu().numpy()
                box_xy, box_wh = (output[i][:,:2] + output[i][:,2:4]) / 2, output[i][:,2:4] - output[i][:,:2]
                output[i][:,:4] = self.centernet_correct_boxes(box_xy, box_wh, input_shape, img_shape, letterbox_img)
        
        return output


    def on_epoch_end(self, epoch, model_val):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_val
            if not os.path.exists(self.map_out):
                os.makedirs(self.map_out)
            if not os.path.exists(os.path.join(self.map_out,'gt')):
                os.makedirs(os.path.join(self.map_out,'gt'))
            if not os.path.exists(os.path.join(self.map_out,'dets')):
                os.makedirs(os.path.join(self.map_out,'dets'))

            print("gen map")

            for ann in tqdm(self.val_lines):
                line = ann.split()
                image_id = os.path.basename(line[0]).split('.')[0]

                image = Image.open(line[0])

                gt = np.array([np.array(list(map(int, box.split(",")))) for box in line[1:]])
                self.get_map_txt(image_id=image_id, image=image, class_names=self.class_names, map_out_path=self.map_out)

                with open(os.path.join(self.map_out, "gt/"+image_id + ".txt"), "w") as new_f:
                    for box  in gt:
                        left, top, right, bottom, class_id = box
                        obj_name = self.class_names[class_id]
                        new_f.write("%s %s %s %s %s\n"%(obj_name, left, top, right, bottom))
            
            print("calc map")

            # ignore coco map.
            #TODO?
            #temp_map = get_map(self.MINOVERLAP, False, path = self.map_out)
            #self.maps.append(temp_map)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                #f.write(str(temp_map))
                f.write("\n")
            return
            plt.figure()
            plt.plot(self.epoches, self.maps, 'red', linewidth = 1.0, label='train map')

            plt.grid(True)
            plt.xlabel('epoch')
            plt.ylabel('map %s '%str(self.MINOVERLAP))
            plt.title("a map curve")
            plt.legend(loc="upper right")
            
            plt.savefig(os.path.join(self.log_dir, "map.png"))
            plt.cla()
            plt.close("all")

            print("get map done")
            shutil.rmtree(self.map_out)





    def decode_box(self, hm, wh, reg, conf_thresh):
        pred_hms = pool_nms(hm)
        b,c,output_h, output_w = pred_hms.shape
        dets = []
        for batch in range(b):
            # heat_map:  128 * 128, num_classes
            # pred_wh:   128 * 128, 2
            # pred_reg:  128 * 128, 2

            # c,h,w ->h,w,c -> h*w,c
            heat_map = pred_hms[batch].permute(1,2,0).view([-1, self.num_class])
            pred_wh = wh[batch].permute(1,2,0).view([-1, 2])
            pred_off = reg[batch].permute(1,2,0).view([-1, 2])

            yv, xv  = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))

            xv, yv = xv.flatten().float(),  yv.flatten().float()
            if self.cuda:
                xv = xv.cuda()
                yv = yv.cuda()
            # max_val, max_idx
            class_conf, class_pred = torch.max(heat_map, dim=-1)
            mask = class_conf > conf_thresh
            pred_wh_mask = pred_wh[mask]
            pred_off_mask = pred_off[mask]

            if len(pred_wh_mask) ==0:
                dets.append([])
                continue
            xv_mask = torch.unsqueeze(xv[mask] + pred_off_mask[...,0], -1)
            yv_mask = torch.unsqueeze(yv[mask] + pred_off_mask[...,1], -1)
            half_w, half_h = pred_wh_mask[...,0:1] / 2, pred_wh_mask[...,1:2] / 2
            bboxes = torch.cat([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h], dim=-1)
            bboxes[:, [0,2]] /= output_w
            bboxes[:, [1,3]] /= output_h
            det = torch.cat([bboxes, torch.unsqueeze(class_conf[mask], -1).float(), torch.unsqueeze(class_pred[mask].float(), -1)], dim=-1)
            dets.append(det)
        return dets


    def get_map_txt(self, image_id, image, class_names, map_out_path):
        return
        f = open(os.path.join(map_out_path, image_id + ".txt"), "w")
        img_shape = np.array(np.shape(image)[0:2])
        img = cvtColorGaurd(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_img)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype=np.float32)),(2,0,1)), axis=0)
        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            if self.backbone == "hourglass":
                outputs = [outputs[-1]["hm"].sigmoid(), outputs[-1]["wh"], outputs[-1]["reg"]]

            
            outputs = self.decode_box(outputs[0], outputs[1], outputs[2], self.confidence)

            results = self.postprocess(outputs, self.nms, img_shape, self.input_shape, self.letterbox_img, self.nms_iou)

            if results[0] is None:
                return
            
            top_label = np.array(results[0][:,5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]
        top_100 = np.argsort(top_conf)[::-1][:self.max_boxes]
        top_boxes = top_boxes[top_100]
        top_conf = top_conf[top_100]
        top_label = top_label[top_100]

        for i, c in list(enumerate(top_label)):
            pred_class = self.class_names[c]
            box = top_boxes[i]
            score =  top_conf[i]

            t, l, b, r  = box
            if pred_class not in class_names:
                continue
            
            f.write("%s %s %s %s %s %s\n"%(pred_class, score, str(int(l)), str(int(t)), str(int(r)), str(int(b))))
        f.close()
        return




class CenterNet_Resnet50(nn.Module):
    __BACKBONE__ = "resnet50"
    def __init__(self, n_class = 20, pretrained = False, tc=None) -> None:
        super(CenterNet_Resnet50, self).__init__()
        self.pretrained = pretrained
        # 512,512,3 -> 16 * 16 * 2048
        self.backbone = resnet50(pretrained= pretrained)

        # 16, 16, 2048 -> 128 , 128, 64
        self.decoder = resnet50_Decoder(2048)

        self.tc = tc
        assert(self.tc is not None)

        self.head = resnet50_Head(channel=64, num_classes= n_class)

        self._init_weights()
        self.input_shape = [512,512]

    def _init_weights(self):
        if not self.pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    # init with MSRC initializer/kaiming
                    m.weight.data.normal_(0, math.sqrt(2./n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        #TODO: last conv2d?
        self.head.cls_head[-1].weight.data.fill_(0)
        self.head.cls_head[-1].bias.data.fill_(0)


    def get_eval_cb(self):
        if self.tc.local_rank == 0:
            self.eval_callback = EvalCallback(self, self.tc.backbone_name, 
                                              self.tc.shapes[0], self.tc.class_names, 
                                         self.tc.num_classes, self.tc.val_lines, 
                                         self.tc.log_dir, self.tc.Cuda, 
                                            eval_flag=self.tc.eval_flag, 
                                            period=self.tc.eval_period)
        else:
            self.eval_callback = None
        return self.eval_callback
        
    def forward(self, x):
        print("under cuda: ", torch.cuda.is_available())
        print("forward backbone.")
        feat = self.backbone(x)
        print("forward decoder.")
        tmp = self.decoder(feat)
        print("forward head.")
        yhat = self.head(tmp)
        print("forward done.")
        return yhat
    

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
    
class CenterNet_HourglassNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        raise NotImplementedError("HourglassNet based centernet is not implemented yet.")