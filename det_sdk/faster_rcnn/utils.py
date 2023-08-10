#!/usr/bin/env python3

import os
import torch
import scipy

from torch.utils.tensorboard import SummaryWriter

from matplotlib import pyplot as plt



def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def normal_init( m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

def loc2box(anchor, loc):
    """
        anchor: (tlx, tly, brx, bry)
        loc: (dx, dy, dw, dh) , prediction result.
    """
    src_width = torch.unsqueeze(anchor[:,2] - anchor[:,0], -1)
    src_height = torch.unsqueeze(anchor[:,3] - anchor[:,1], -1)
    src_c_x = torch.unsqueeze(anchor[:,0], -1) + 0.5 * src_width
    src_c_y = torch.unsqueeze(anchor[:,1], -1) + 0.5 * src_height

    print("loc shape under loc2box: ", loc.shape)
    dx = loc[:, 0::4]
    dy = loc[:, 1::4]
    dw = loc[:, 2::4]
    dh = loc[:, 3::4]

    ctr_x = dx * src_width + src_c_x

    ctr_y = dy * src_height + src_c_y

    w = torch.exp(dw) * src_width
    h = torch.exp(dh) * src_height

    dst_bbox = torch.zeros_like(loc)
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h

    return dst_bbox



import numpy as np



def get_new_img_size(height, width, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)
    return resized_height, resized_width

class DecodeBox(object):
    def __init__(self, std, num_classes):
        self.std = std
        self.num_classes = num_classes

    def frcnn_correct_boxes(self, box_xy, box_wh, input_shape, image_shape):

        # here we put y axis in front of x axis
        # in that it is more convenient to multi pred-box's width-height
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        input_shape  = np.array(input_shape)
        image_shape = np.array(image_shape)

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)

        boxes = np.concatenate([box_mins[..., 0:1], 
                    box_mins[..., 1:2], 
                    box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)

        return boxes

    def forward(self, roi_cls_locs, roi_scores, rois, image_shape,
                input_shape, nms_iou = 0.3, confidence = 0.5):
        results = []

        # return outer-most dimension.
        bs = len(roi_cls_locs)

        # batch_size, num_rois, 4
        rois = rois.view(bs, -1, 4)

        for i in range(bs):
            # reshape regression parameters
            roi_cls_loc = roi_cls_locs[i] * self.std

            # [num_rois, num_classes, 4]
            roi_cls_loc = roi_cls_loc.view([-1, self.num_classes, 4])


            roi = rois[i].view((-1, 1, 4)).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(roi.contiguous().view((-1,4)), roi_cls_loc.contiguous().view(-1,4))
            cls_bbox = cls_bbox.view([-1, (self.num_class), 4])

            # normalize pred-bbox
            # input_shape-> height-width?
            cls_bbox[..., [0,2]] = (cls_bbox[...,[0,2]])/input_shape[1]
            cls_bbox[...,[1,3]] = (cls_bbox[...,[1,3]])/input_shape[0]


            roi_score = roi_scores[i]
            prob = F.softmax(roi_score, dim=-1)

            results.append([])
            for c in range(1, self.num_classes):
                # threshold check

                c_confs = prob[:,c]
                c_confs_m = c_confs > confidence

                if len(c_confs[c_confs_m]) > 0:
                    boxes_to_process = cls_bbox[c_confs_m, c]
                    confs_to_process = c_confs[c_confs_m]

                    keep = nms(
                        bboxes_to_process,
                        confs_to_process,
                        nms_iou
                    )


                    # get good box from results of nms
                    good_boxes = boxes_to_process[keep]
                    # why we need None?
                    confs = confs_to_process[keep][:,None]
                    if confs.is_cuda:
                        labels = (c-1)  * torch.ones((len(keep), 1)).cuda()  
                    else:
                        labels = ( c-1) * torch.ones((len(keep),1))

                    c_pred = torch.cat((good_boxes, confs, labels), dim=1).cpu().numpy()
                    results[-1].append(c_pred)

            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                box_xy, box_wh = (results[-1][:,0:2] + results[-1][:,2:4])/2,\
                                    results[-1][:,2:4] - results[-1][:,0:2]
                results[-1][:,:4] = self.frcnn_correct_boxes(box_xy, box_wh, input_shape, image_shape)
        return results


def init_model( model_path, model, device):
    import os
    assert(os.path.exists(model_path))
    print("Loading model from {}".format(model_path))

    model_dict = model.state_dict()
    #  pre-trained weights
    # this is dict from parameter name to tensor
    pretrained_dict = torch.load(model_path, map_location=device)
    
    # for k in model_dict.keys():
    #     print("model: ",k)

    # for k in pretrained_dict.keys():
    #     print("pretrained: ",k)
    
    
    load_key, no_load_key, temp_dict = [], [],{}

    for k,v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            # k exist in model_dict and shape of v is same
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)

    model_dict.update(temp_dict)
    # reload model_state.
    model.load_state_dict(model_dict)
    #print("updated dict: ",temp_dict.keys())
    print("Loaded pre-trained weights from {}".format(model_path))
    print("No pre-trained weights to load: {}".format(no_load_key))


