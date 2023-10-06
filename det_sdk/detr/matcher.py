#!/usr/bin/env python3

import torch

from torch import nn
from torchvision.ops.boxes import box_area

def box_cxcywh_to_xyxy(box):
    # remove the last dim, the bbox dim
    # return tuple of each slice based on last dim.
    x_c, x_y, w, h = box.unbind(-1)
    # b: lt and br
    b = [(x_c - 0.5 * w), (x_y - 0.5 * h), 
         (x_c + 0.5 * w), (x_y + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # for each box in boxes1,
    # broadcast to boxes2 and do max computation.
    # turn out, each box will be compared with each box in boxes2 max res it recored.
    # res shape: [N, M, 2]
    lt = torch.max(boxes1[:,None, :2], boxes2[:,:2])
    # same as above , but it refers to bottom right point.
    rb = torch.min(boxes1[:,None, 2:], boxes2[:,2:])

    # set low bound to 0
    wh = (rb - lt).clamp(min=0)

    # res shape: [N, M]
    # compute intersection area
    inter = wh[:, :, 0] * wh[:, :, 1]
    # compute union area
    # do broadcast for each box-area of area1
    # add each area1 with area2, then minus inter area.
    # res shape: [N, M]
    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    Input: tow boxes with shape: [N, 4] and [M,4], each box is [x1, y1, x2, y2]
    output: [N, M] pairwise matrix, where N = len(boxes1), M = len(boxes2)
    """
    assert(boxes1.shape[-1] == 4)
    assert(boxes2.shape[-1] == 4)
    # in [tl , br] format
    assert(boxes1[:,2:] >= boxes1[:,:2]).all()
    assert(boxes2[:,2:] >= boxes2[:,:2]).all()

    iou, union = box_iou(boxes1, boxes2)

    # this is to compute closure.
    # min of lt compared between each box pair(box1 from boxes1
    #  and box2 from boxes2) 
    # will get the upper-left point of the closure box(box1 and box2).
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)

    area = wh[:, :, 0] * wh[:, :, 1]
    # below is the giou formula.
    # A-area is closure area.
    # giou = iou - (A-area - union) / A-area
    return iou - (area - union) / area

class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions 
    of the network.

    For efficiency reasons, the targets don't include the no_object.
    (there are more predictions than targets).
    There are more predictions than targets. In this case, we do a 1-to-1
    matching of the best predictions,

    Leaving others unmatched.
    """
    def __init__(self, cost_class : float=1, cost_bbox : float=1, cost_giou :float = 1):
        """
        Creates the matcher: 
            cost_class: relative classification error weight in cost matrix.
            cost_bbox: relative L1 weight (of bbox coordinates) in cost matrix.
            cost_giou: relative giou weight (between gt and preds bounding box) in cost matrix.
        """
        super(HungarianMatcher, self).__init__()
        self.cost_class = cost_class 
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert(cost_class != 0)
        assert(cost_bbox != 0)
        assert(cost_giou != 0)


    @torch.no_grad()
    def forward(self, outputs, targets):
        """
            Perform matching
            Parameters:
                outputs: dict of tensors, output of the model, etc:
                    { 
                        "pred_logits": x(Tensor of dim [batch_size, num_queries, num_classes] with classification logits),
                        "pred_boxes": y(Tensor of dim [batch_size, num_queries, 4] with predicted box coordinates)
                    }
                targets: list of targets (len(targets) == batch_size), each target is
                a dict: 
                    {
                        "labels": x(Tensor of dim [num_target_boxes], ),
                        "boxes": y(Tensor of dim [num_target_boxes, 4])
                    }
            Returns:
                A list of size batch_size, (index_i, index_j), where
                 - index_i is the indices of the selected predictions (in order)
                 - index_j is the indeces of the corresponding selected targets (in order too)
                For each batch element, it holds:
                    len(index_i) = len(index_j) = min(num_queries, num_target_boxes)        
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # compute prob for last dim, the class dim
        # res shape: [batch_size * num_queries, num_classes]
        out_prob = outputs["pred_logits"].flatten(0,1).softmax(-1)

        # res shape: [batch_size * num_queries, 4]
        out_bbox = outputs["pred_boxes"].flatten(0,1)


        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # retrospect definition: 
        # 
        # [Entroy]: 
        # The shortest encode method, log(p) is the encode length of 
        # information set S(i). p(i) is the probability/frequence of S(i) in the whole set S. 
        # Total encode length is sigma(-1 * p(i) * log(p(i)))
        # 
        # [Cross Entropy]
        # the encode method, formula is sigma(-1 * p(i) * log(q(i)))
        # p(i) is the probability/frequence of S(i) in the whole set S, (targets)
        # q(i) is the probability/frequence of S(i) in the whole set S, (outputs)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
        cost_class = -out_prob[:, tgt_ids]

        # compute p-norm distance(p default is 2,here it is 1, manhattan distance)
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), 
                                         box_cxcywh_to_xyxy(tgt_bbox))
        
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        # separate bs and num_quiies dim
        C = C.view(bs, num_queries, -1).cpu()

        # for each batch, get batch's gt boxes num.
        sizes = [len(v["boxes"]) for v in targets]
        # split boxes of each batch and do linear_sum_assignment upon it.
        #TODO: figure it out.
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64),
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]



def build_matcher(args):
    return HungarianMatcher(cost_class = args.set_cost_class,
                            cost_bbox = args.set_cost_bbox,
                            cost_giou = args.set_cost_giou)