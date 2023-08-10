#!/usr/bin/env python3

# refer: https://github.com/bubbliiiing/faster-rcnn-pytorch/blob/6cf39bf75d95975210e9b6f94cad691b330fee90/utils/callbacks.py#L79
import sys
import math
import os
from tqdm import tqdm
from functools import partial
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.nn import functional as F




import numpy as np
import datetime

from torch import nn
from dataset import FasterRCNNDataSet
from dataset import frcnn_dataset_collate
from utils import get_classes
from framework.train_helper import LossHistory
#from samples.pytorch.framework.train_helper import EvalCallback
from utils import init_model

from framework import scheduler

from common.utils import weights_init
from faster_rcnn.utils import DecodeBox

from frcnn import FasterRCNN

class EvalCallbackFaster(object):
    def __init__(self, net, input_shape, class_names, num_classes, val_lines, log_dir, cuda\
                 ,map_out="./tmp_out", max_boxes = 100, confi=0.05, nms_iou=0.5\
                 , letterbox=True, MIN_OVERLAP=0.5, eval_flag=True, period=1 ):
        super(EvalCallbackFaster, self).__init__()

        self.net = net
        self.input_shape = input_shape
        self.class_names = class_names
        self.num_classes = num_classes
        self.val_lines = val_lines
        self.log_dir = log_dir
        self.cuda = cuda
        self.map_out = map_out
        self.max_boxes = max_boxes
        self.confidence = confi
        self.nms_iou = nms_iou
        self.letterbox_img = letterbox
        self.MIN_OVERLAP = MIN_OVERLAP
        self.eval_flag = eval_flag
        self.period = period


        self.std = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(self.num_classes + 1)
        if self.cuda:
            self.std = self.std.cuda()
        self.bbox_util = DecodeBox(self.std, self.num_classes)

        self.maps = [0]
        self.epoches = [0]
        
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def get_map_txt(self, img_id, img, class_names, map_out):
        f = open(os.path.join(map_out, "det-results/"+img_id+".txt"), 'w')

        # compute image height and width
        img_shape = np.array(np.shape(img)[0:2])
        # change to (600, width), or (height, 600)
        input_shape = get_new_img_size(img_shape[0],  img_shape[1])

        # cvt to rgb
        img = cvtColor(img)

        # resize  to width == 600
        img_data = resize_img(img, [input_shape[1], input_shape[0]])

        # add batch dimension
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(img_data, dtype='float32')) , (2,0,1)),0)


        # ready to predict.
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            
            roi_cls_locs, roi_scores, rois, _ = self.net(images)
            # decode predictions
            results = self.bbox_util.forward(roi_cls_locs, roi_scores,
                                            image_shape,input_shape,
                                            nms_iou=self.nms_iou,confidence=self.confidence)
            if len(results[0]) <= 0:
                return
            top_label = np.array(results[0][:,5], dtype='int32')
            top_conf = results[0][:,4]
            top_boxes = results[0][:,:4]

        top_100 = np.argsort(top_conf)[::-1][:self.max_boses]
        top_boxes = top_boxes[top_100]
        top_confi =   top_conf[top_100]
        top_label = top_label[top_100]

        for i, c in list(enumerate(top_label)):
            pred_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_confi[i])

            top, left, bottom, right = box
            if pred_class not in class_names:
                continue
            f.write("%s %s %s %s %s %s\n" % (pred_class,score[:6], str(int(left)), str(int(top)),str(int(right)), str(int(bottom))))

        f.close()

        return

    def on_epoch_end(self, epoch):
        if epoch % self.period ==0 and self.eval_flag:
            if not os.path.exists(self.map_out):
                os.makedirs(self.map_out)
            if not os.path.exists(os.path.join(self.map_out, "ground-truth")):
                os.makedirs(os.path.join(self.map_out, "ground-truth"))
            if not os.path.exists(os.path.join(self.map_out, "det-results")):
                os.makedirs(os.path.join(self.map_out, "det-results"))

            print("gget map.")

            for ann_line in tqdm(self.val_lines):
                line = ann_line.strip().split()
                img_id = os.path.basename(line[0])

                img = Image.open(line[0])

                gt_bboxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

                self.get_map_txt(img_id, img, self.class_names, self.map_out)

                with open(os.path.join(self.map_out, "ground-truth/"+img_id+".txt"), 'w') as f:
                    for box in gt_bboxes:
                        left, top, right, bottom, obj = box
                        obj_name = self.class_names[obj]
                        f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

            print("calculate map.")
        
            try:
                temp_map = get_coco_map(class_names = self.class_names, path = self.map_out)[1]
            except:
                temp_map = get_map(self.MINOVERLAP, False, path=self.map_out)

            self.maps.append(temp_map)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(temp_map))
                f.write("\n")
            plt.grid(True)
            plt.xlabel("Epoch")
            plt.ylabel("Map %s"%str(self.MINOVERLAP))
            plt.title("A Map Curve")
            plt.legend(loc='upper right')
            
            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")

            print("save map.")
            shutil.rmtree(self.map_out)


def bbox_iou(anchor, bbox):
    if anchor.shape[1] != 4 or bbox.shape[1] != 4:
        print(anchor, bbox)
        print('anchor shape:should be (n * 4) while it is:', anchor.shape)
        print('bbox shape:should be (n * 4) while it is:', bbox.shape)
        raise IndexError
    # based on broadcast mechanism, anchor will compare with each bbox
    # 1. for tl  , we select maximum of each of anchor and bbox
    # 2. for br , we select minimum of each of anchor and bbox

    #print("shape of anchor: ", anchor.shape)
    #print("shape of bbox: ", bbox.shape)
    tl = np.maximum(anchor[:,None, :2], bbox[:,:2])
    br = np.minimum(anchor[:,None, 2:], bbox[:,2:])


    #print("shape of tl: ", tl.shape)
    #print("shape of br: ", br.shape)
    # br.x - tl.x, br.y - tl.y,
    # prod of axis==2 will, reduce axis==2 and get the area.
    # select with `all` 
    area_inter = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_anchor = np.prod(anchor[:,2:] - anchor[:,:2], axis=1)
    area_bbox = np.prod(bbox[:,2:] - bbox[:,:2], axis=1)
    return area_inter / (area_anchor[:,None] + area_bbox - area_inter)


def bbox2loc(src_bbox, dst_bbox):
    # br.x - tl.x
    width = src_bbox[:,2] - src_bbox[:,0]
    height = src_bbox[:,3] - src_bbox[:,1]
    ctr_x = src_bbox[:,0] + 0.5 * width
    ctr_y = src_bbox[:,1] + 0.5 * height


    base_width = dst_bbox[:,2] - dst_bbox[:,0]
    base_height = dst_bbox[:,3] - dst_bbox[:,1]
    base_ctr_x = dst_bbox[:,0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:,1] + 0.5 * base_height

    eps  = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)

    # compute center distance scaled by width and height
    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)

    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc



class AnchorTargetCreator(object):
    def __init__(self, n_samples=256, pos_iou_thresh=0.7, 
                 neg_iou_thresh=0.3, pos_ratio=0.5):
        self.n_samples = n_samples
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor):
        argmax_ious, label = self._create_label(anchor, bbox)
        if (label > 0).any():
            loc = bbox2loc(anchor, bbox[argmax_ious])
            return loc, label
        else:
            return np.zeros_like(anchor), label
        

    

        
    def _calc_ious(self, anchor, bbox):
        # compute ious between anchor and bbox
        # output shape: [num_anchors, num_gt]

        ious = bbox_iou(anchor, bbox)
        if len(bbox) == 0:
            return np.zeros(len(anchor), np.int32), np.zeros(len(anchor)), np.zeros(len(bbox))
        # get gt/pred-box with max iou compared with each anchor box
        argmax_ious = ious.argmax(axis=1)

        max_ious = np.max(ious, axis=1)

        # get anchor box with max iou compared with each gt/pred box
        gt_argmax_ious = ious.argmax(axis=0)

        # len of argmax_ious is equal to len(anchors)
        # each contains index of bbox with max iou

        # while len of gt_argmax_ious is equal to len(bbox)
        # each contains index of anchor with max iou
        
        #below assignment will make anchor related to pred/gt bbox.
        for i in range(len(gt_argmax_ious)):
            argmax_ious[gt_argmax_ious[i]] = i
        return argmax_ious, max_ious, gt_argmax_ious
    
    def _create_label(self, anchor, bbox):
        # 1: positive sample
        # 0: negative sample
        # -1: default value,ignore.

        label = np.empty((len(anchor),), dtype=np.int32)
        label.fill(-1)
        # argmax_ious (num_anchors,)
        # max_ious (num_anchors,)
        # gt_argmax_ious (num_gt,)

        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox)

        label[max_ious < self.neg_iou_thresh] = 0
        label[max_ious >= self.pos_iou_thresh] = 1
        # mark label ==1 for each pred/gt bbox (make it at least 1 corresponding anchor)
        if len(gt_argmax_ious) > 0:
            label[gt_argmax_ious] = 1

        n_pos = int(self.pos_ratio * self.n_samples)
        # np.where will return (array(inx, inx, ...),)
        pos_index = np.where(label ==1)[0]

        # mark as -19(default value)  by randomly chosing from pos_index
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace = False)
            label[disable_index] = -1

        n_neg = self.n_samples - np.sum(label ==1)
        neg_index = np.where(label==0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1


        return argmax_ious, label

        
class ProposalTargetCreator(object):
    def __init__(self, n_sample=128, pos_ratio=0.5, pos_iou_threshold=0.5,
                 neg_iou_threshold_high=0.5,neg_iou_threshold_low=0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_roi_per_img = np.round(self.n_sample * self.pos_ratio)
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold_high = neg_iou_threshold_high
        self.neg_iou_threshold_low = neg_iou_threshold_low

    def __call__(self, roi, bbox, label, loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        roi = np.concatenate((roi.detach().cpu().numpy(), bbox), axis=0)
        iou = bbox_iou(roi, bbox)

        if len(bbox) == 0:
            gt_idx = np.zeros(len(roi), np.int32)
            max_iou = np.zeros(len(roi))
            gt_roi_label = np.zeros(len(roi))
        else:
            # [num_roi, ] 
            # get index of `most corresponding gt bbox` for each roi/proposed
            gt_idx = iou.argmax(axis=1)
            # [num_roi, ]
            # get iou of `most corresponding gt bbox` for each roi/proposed
            max_iou = iou.max(axis=1)
            # [num_roi, ]
            # `+1` here is that backgroud label is 0.
            gt_roi_label = label[gt_idx] + 1

        pos_index = np.where(max_iou >= self.pos_iou_threshold)[0]
        pos_roi_per_this_img = int(min(self.pos_roi_per_img, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_img, replace=False)

        neg_index = np.where((max_iou < self.neg_iou_threshold_high) & (max_iou >= self.neg_iou_threshold_low))[0]

        neg_roi_per_this_img = self.n_sample - pos_roi_per_this_img
        neg_roi_per_this_img = int(min(neg_roi_per_this_img, neg_index.size))

        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size = neg_roi_per_this_img, replace = False)

        # append will return 1-d array index.
        keep_index = np.append(pos_index, neg_index)

        sample_roi = roi[keep_index]
        #
        # sample_roi [n_sample, ]
        # gt_roi_loc [n_sample, ]
        # gt_roi_label [n_sample, ]
        #

        if len(bbox) ==0:
            return sample_roi, np.zeros_like(sample_roi), gt_roi_label[keep_index]
        
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_idx[keep_index]])
        gt_roi_loc = (gt_roi_loc/ np.array(loc_normalize_std, np.float32))

        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_img:] = 0
        return sample_roi, gt_roi_loc, gt_roi_label





class FasterRCNNTrainer(nn.Module):
    def __init__(self, model,  optimizer):
        super(FasterRCNNTrainer, self).__init__()
        self.model = model
        self.optimizer = optimizer

        self.rpn_sigma = 1
        self.roi_sigma = 1

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_std = [0.1, 0.1, 0.2, 0.2]



    def _fast_rcnn_loc_loss(self, pred_loc, gt_loc, gt_label,sigma):
        """
           loss: `smoth - l1` 
        """
        # where gt_label is 1
        pred_loc = pred_loc[gt_label > 0]
        gt_loc = gt_loc[gt_label > 0]
        sigma_squared = sigma ** 2#torch.pow(sigma, 2)

        # for all with right gt_label
        reg_diff = (gt_loc - pred_loc)
        # l1 loss
        #TODO? understand this.
        reg_diff = reg_diff.abs().float()
        reg_loss = torch.where(
            reg_diff < (1./sigma_squared),
            0.5 * sigma_squared * reg_diff **2,
            reg_diff - 0.5 / sigma_squared            
        )
        regression_loss = reg_loss.sum()
        num_pos = (gt_label> 0).sum().float()

        # max normalization
        regression_loss /= torch.max(num_pos, torch.ones_like(num_pos))

        return regression_loss

    def forward(self, imgs, bboxes, labels, scale):
        n = imgs.shape[0]
        img_size = imgs.shape[2:]
        print("--- img size is: ", img_size)

        assert(self.model is not None)
        print("type of model", type(self.model))

        base_ft = self.model(imgs, mode='extract')

        # get 
        assert(base_ft is not None)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.model(x = [base_ft, img_size], scale=scale, mode='rpn')

        rpn_loc_loss_all, rpn_cls_loss_all, roi_loc_loss_all, roi_cls_loss_all = \
            0, 0, 0, 0
    
        sample_rois, sample_indexes,  gt_roi_locs, gt_roi_labels = [], [], [], []

        for i in range(n):
            bbox = bboxes[i]
            label = labels[i]
            rpn_loc = rpn_locs[i]
            rpn_score = rpn_scores[i]
            roi = rois[i]

            # compute gt box and rpn box  to get pred result.

            # gt_rpn_loc [num_anchors, 4]
            # gt_rpn_label [num_anchors, ]

            #
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(bbox, anchor[0].cpu().numpy())
            gt_rpn_loc = torch.from_numpy(gt_rpn_loc).type_as(rpn_locs)
            gt_rpn_label = torch.from_numpy(gt_rpn_label).type_as(rpn_locs).long()

            # compute proposal box 's regression and classification score.
            rpn_loc_loss = self._fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc,gt_rpn_label, self.rpn_sigma)
            rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1) 


            rpn_loc_loss_all += rpn_loc_loss
            rpn_cls_loss_all += rpn_cls_loss


            # use gt box and proposal box to get classification results
            # sample_roi = [n_sample,]
            # gt_roi_loc = [n_sample,4]
            # gt_roi_label = [n_sample,]

            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi, bbox, label, self.loc_normalize_std)
            sample_rois.append(torch.Tensor(sample_roi).type_as(rpn_locs))
            sample_indexes.append(torch.ones(len(sample_roi)).type_as(rpn_locs) * roi_indices[i][0])
            gt_roi_locs.append(torch.Tensor(gt_roi_loc).type_as(rpn_locs))
            gt_roi_labels.append(torch.Tensor(gt_roi_label).type_as(rpn_locs).long())

        sample_rois = torch.stack(sample_rois,dim=0)
        sample_indexes = torch.stack(sample_indexes,dim=0)

        roi_cls_locs, roi_scores = self.model([base_ft, sample_rois,sample_indexes, img_size], mode='head')

        for i in range(n):
            # fetch regression result according to proposal box's type
            n_sample = roi_cls_locs.size()[1]

            roi_cls_loc = roi_cls_locs[i]
            roi_score = roi_scores[i]
            gt_roi_loc = gt_roi_locs[i]
            gt_roi_label = gt_roi_labels[i]

            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            print(gt_roi_label.dtype)
            roi_loc = roi_cls_loc[torch.arange(0, n_sample), gt_roi_label]


            roi_loc_loss = self._fast_rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label.data, self.roi_sigma)
            roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label)

            roi_loc_loss_all += roi_loc_loss
            roi_cls_loss_all += roi_cls_loss

        losses = [rpn_loc_loss_all/n, rpn_cls_loss_all/n, roi_loc_loss_all/n, roi_cls_loss_all/n]
        losses = losses + [sum(losses)]

        return losses

    def train_step(self, imgs, bboxes, labels, scale, fp16=False, scaler = None):
        self.optimizer.zero_grad()
        if not fp16:
            losses = self.forward(imgs, bboxes, labels, scale)
            losses[-1].backward()
            self.optimizer.step()

        else:
            from torch.cuda.amp import autocast
            with autocast():
                losses = self.forward(imgs, bboxes, labels, scale)
            
            scaler.scale(losses[-1]).backward()
            scaler.step(self.optimizer)
            scaler.update()
        return losses


                               
        



# def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iter, warmup_iter_ratio=0.05, 
#                      warmup_lr_ratio=0.1, no_aug_iter_ratio=0.05, step_num =10):


#     print("get lr scheduler: decay_type: %d, lr: %d , min_lr: %d, total_iter : %d"%(lr_decay_type, lr, min_lr, total_iter))
#     def yolox_warm_cos_lr(lr, min_lr, total_iter, warmup_total_iter, warmup_lr_start,
#                           no_aug_iter, iters):
#         if iters <= warmup_total_iter:
#             lr = (lr - warmup_lr_start) / pow(iters/float(warmup_total_iter),2) + warmup_total_iter*warmup_lr_start
#         elif iters >= total_iter - no_aug_iter:
#             lr = min_lr

#         else:
#             lr = min_lr + 0.5 * (lr - min_lr) * (
#                 1.0 + math.cos(math.pi * (iters - warmup_total_iter)/(total_iter - warmup_total_iter - no_aug_iter - no_aug_iter)))
#         return lr
    
#     def step_lr(lr, decay_rate, step_size, iters):
#         if step_lr < 1:
#             raise ValueError("step_lr must be greater than 1")
#         n = iters// step_size
#         out_lr = lr * decay_rate **n
#         return out_lr
    
#     if lr_decay_type == 'cos':
#         warmup_total_iters = min(max(warmup_iter_ratio * total_iter, 1), 3)
#         warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)

#         no_aug_iter = min(max(no_aug_iter_ratio * total_iter, 1), 15)

#         func = partial(yolox_warm_cos_lr, lr, min_lr, total_iter,warmup_total_iters,warmup_lr_start, no_aug_iter)

#     else:
#         decay_rate = (min_lr/ lr ) ** (1/(step_num -1))
#         step_size = total_iter//step_num
#         func = partial(step_lr, lr, decay_rate, step_size)
    
#     return func
#             #lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iter, eta_min=min_lr)




def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func( epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(model, train_util, loss_history, eval_callback,
                optimizer, epoch, epoch_step, epoch_step_val, 
                gen, gen_val, Epoch , cuda, fp16, scaler, save_period, save_dir):
    total_loss = 0
    rpn_loc_loss = 0
    rpn_cls_loss = 0
    roi_loc_loss = 0
    roi_cls_loss = 0


    val_loss = 0
    print("start train")

    with tqdm(total = epoch_step, desc =f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3 ) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, boxes, labels = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    images = images.cuda()

            
            rpn_loc, rpn_cls, roi_loc, roi_cls, total = train_util.train_step(images, boxes, labels, 1, fp16, scaler)
            total_loss += total.item()
            rpn_loc_loss += rpn_loc.item()
            rpn_cls_loss += rpn_cls.item()
            roi_loc_loss += roi_loc.item()
            roi_cls_loss += roi_cls.item()

            pbar.set_postfix(**{
                'total_loss': total_loss / (iteration + 1),
                'rpn_loc' : rpn_loc_loss / (iteration + 1),
                'rpn_cls' : rpn_cls_loss / (iteration + 1),
                'roi_loc' : roi_loc_loss / (iteration + 1),
                'roi_cls' : roi_cls_loss / (iteration + 1),
                'lr' : get_lr(optimizer)
            })

            pbar.update(1)
    print("end train")
    print("start eval")
    with tqdm(total= epoch_step_val, desc =f'Epoch {epoch + 1}/{Epoch}', postfix=dict,mininterval=0.3) as pbar:
        for iter, batch in enumerate(gen_val):
            if iter >= epoch_step_val:
                break
            images, boxes, labels = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    images = images.cuda()
                optimizer.zero_grad()
                _, _, _, _, val_total = train_util.forward(images, boxes, labels, 1)
                val_loss += val_total.item()

                pbar.set_postfix(**{
                    'val_loss': val_loss / (iter + 1)
                })
                pbar.update(1)
    print("finish validation")
    loss_history.append_loss(epoch+1, total_loss/epoch_step, val_loss/epoch_step_val)
    eval_callback.on_epoch_end(epoch+1 )
    print("Epoch: " + str(epoch+1) + "/" + str(Epoch))
    print("Total Loss: %.3f || Val loss: %.3f" % (total_loss/epoch_step, val_loss/epoch_step_val))

    if (epoch+1) % save_period == 0 or epoch+1 == Epoch:
        
        torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-valloss%.3f.pth"%(epoch+1, total_loss/epoch_step, val_loss/epoch_step_val)))
    
   # if (len(loss_history.val_loss) <= 1):
    if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
        # 
        print("save best model to best_epoch_weights.pth")
        torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

    torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))


def train():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # multi-gpu training
    Cuda = False

    fp16 = False

    classes_path = "data/voc/classes_voc.txt"


    model_path = "model_checkpoints/voc_weights_resnet.pth"

    input_shape = [600, 600]

    backbone = "resnet50"

    #only valid when model_path is None
    pretrained=False

    # prior set anchor size.
    # custom according to use case.
    anchor_size = [ 8, 16, 32]

    init_epoch = 0
    freeze_epoch = 50
    freeze_batch_size = 4

    unfreeze_epoch = 100
    unfreeze_batch_size = 2

    freeze_train = True
    # for adam, set init_lr to 1e-4
    # for sgd, set init_lr to 1e-2
    init_lr = 1e-4
    min_lr = init_lr * 0.01
    
    optimizer_type = "adam"
    momentum = 0.9
    # for adam, set weight_decay = 0
    weight_decay = 0

    lr_decay_type = 'cos'

    save_period = 5

    save_dir = 'data/voc/logs'

    eval_flag = True
    eval_period = 5

    num_workers = 1

    train_anno_path = "data/voc/2007_train.txt"
    val_anno_path = "data/voc/2007_val.txt"

    class_names, num_classes = get_classes(classes_path)
    model = FasterRCNN(num_classes, anchor_scales=anchor_size, backbone=backbone, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path is not None:
        init_model(model_path, model, device)
        
        
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" +time_str)
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    if fp16:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None


    model_train  = model.train()
    print("model type: ", type(model))
    print('model_train type: ', type(model_train))

    # for cuda, using this branch.
    if Cuda:
        model_train = torch.nn.DataParallel(model_train)

        cudnn.benchmark  = True
        model_train = model_train.cuda()

    
    with open(train_anno_path, 'r') as f:
        train_lines = f.readlines()

    with open(val_anno_path, 'r') as f:
        val_lines = f.readlines()

    num_train = len(train_lines)
    num_val = len(val_lines)

    # show_config(
    #     classes_path = classes_path, 
    #     model_path = model_path, 
    #     input_shape = input_shape, 
    #     anchor_size = anchor_size, 
    #     init_epoch = init_epoch, 
    #     freeze_epoch = freeze_epoch,
    # )

    wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
    total_step = num_train //  unfreeze_batch_size  * unfreeze_epoch

    if total_step < wanted_step:
        if num_train // unfreeze_batch_size == 0 :
            raise ValueError("dataset is too  small for training")
        wanted_epoch = wanted_step //(num_train//unfreeze_batch_size) + 1

        print("it is adviced that total_step should be larger than %d , when using %s optimizer"%(wanted_step, optimizer_type))
        print("In this run, total train set has %d samples, unfreeze_batch_size is %d"%(num_train, unfreeze_batch_size))
        print("All together %d steps and % d epochs"%(total_step, wanted_epoch))
        print("All train step is %d, less than %s all step, set all epoch to %d"%(total_step, wanted_step, wanted_epoch))



    if True:
        UnFreeze_flag = False
        # freeze extractor.
        if freeze_train:
            for p in model.extractor.parameters():
                p.requires_grad = False


        model.freeze_bn()

        batch_size = freeze_batch_size if freeze_train else unfreeze_batch_size

        nbs = 16

        lr_limit_max = 1e-4 if optimizer_type == "sgd" else 5e-2
        lr_limit_min = 1e-4 if optimizer_type == "sgd" else 5e-4

        Init_lr_fit = min(max(batch_size/nbs * init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size/nbs * min_lr, lr_limit_min * 1e-2),  lr_limit_max * 1e-4)


        optimizer = {
            'adam' : optim.Adam(model.parameters(), lr=Init_lr_fit, betas=(momentum, 0.99), weight_decay=weight_decay),
            'sgd' : optim.SGD(model.parameters(), lr=Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
        }[optimizer_type]


        lr_scheduler_func = scheduler.get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, unfreeze_epoch)

        epoch_step = num_train//batch_size
        epoch_step_val = num_val//batch_size


        if epoch_step ==0 or epoch_step_val ==0:
            raise ValueError("dataset is too  small for training")
        
        train_dataset = FasterRCNNDataSet(train_lines, input_shape, train=True)
        val_dataset = FasterRCNNDataSet(val_lines, input_shape, train=False)


        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,drop_last =True, collate_fn=frcnn_dataset_collate)
        
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,drop_last =True, collate_fn=frcnn_dataset_collate)
        
        print("train model is: ", type(model_train))
        train_util = FasterRCNNTrainer(model_train, optimizer)
        eval_callback  = EvalCallbackFaster(model_train, input_shape, class_names, num_classes,
                                      val_lines, log_dir, Cuda, eval_flag=eval_flag, period=eval_period)
        

        print("init epoch is: ", init_epoch, " and unfreeze_epoch is: ", unfreeze_epoch)    
        
    
        for epoch in range(init_epoch, unfreeze_epoch):
            if epoch >= freeze_epoch and not UnFreeze_flag and freeze_train:
                batch_size = unfreeze_batch_size

                nbs = 16
                lr_limit_max = 1e-4 if optimizer_type == "sgd" else 5e-2
                lr_limit_min = 1e-4 if optimizer_type == "sgd" else 5e-4

                Init_lr_fit = min(max(batch_size/nbs * init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size/nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)


                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, unfreeze_epoch)
                assert(lr_scheduler_func is not None)

                for param in model.extractor.parameters():
                    param.requires_grad = True

                model.freeze_bn()
                epoch_step = num_train//batch_size
                epoch_step_val = num_val//batch_size

                if epoch_step ==0 or epoch_step_val == 0:
                    raise ValueError("dataset is too  small for training")
                gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=frcnn_dataset_collate)

                gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,ping_memory=True,
                                    drop_last=True, collate_fn=frcnn_dataset_collate)


                UnFreeze_flag = True

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            print("begin fit on epoch: ")
            
            fit_one_epoch(model, train_util, loss_history, eval_callback, optimizer, epoch, epoch_step ,epoch_step_val, gen, gen_val, unfreeze_epoch, Cuda, fp16, scaler, save_period, save_dir)

        loss_history.writer.close()

    


if __name__ == '__main__':
    train()