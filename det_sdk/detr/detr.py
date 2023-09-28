#!/usr/bin/env python3

from torch import nn
import torch
from utils import NestedTensor
import torch.nn.functional as F
import torchvision
from torch import Tensor
from typing import List

from misc import _onnx_nested_tensor_from_tensor_list

class Joiner(nn.Sequential):

    def __init__(self, backbone, pos_emb):
        super(Joiner, self).__init__(backbone, pos_emb)
    
    def forward(self, tensor_list: NestedTensor):
        # forward with backbone
        xs = self[0](tensor_list)
        out : List(NestedTensor) = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos

class MLP(nn.Module):
    """
        One vanilla MLP with hidden layer and output layer.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super(MLP, self).__init__()
        self.n_layers = num_layers
        h = [hidden_dim] * (num_layers - 1 )
        # layer 0: input layer in_dim -> h 
        # layer 1: h -> h
        # ...
        # layer n: h -> out_dim
        self.layers = nn.ModuleList(nn.Linear(n,k) for n,k in zip([in_dim] + h, h+ [out_dim]))



    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = F.relu(l(x)) if i < self.n_layers - 1 else l(x)
        return x


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublib in the_list[1:]:
        for idx, item in enumerate(sublib):
            maxes[idx] = max(maxes[idx], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            return _onnx_nested_tensor_from_tensor_list(tensor_list)
        
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size

        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device

        tensor = torch.zeros(batch_shape, dtype=dtype, device = device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device = device)

        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError("not supported tensor_list shape")
    return NestedTensor(tensor, mask)

        
class DETR(nn.Module):
    """
        DETR model.

        backbone: backbone network (ResNet50?)
        transformer: transformer network.
        n_classes: number of classes.
        n_queries: number of queries.
        aux_loss: whether to use auxiliary decoding losses(loss computed at each decoder layer).
    """
    def __init__(self, backbone, transformer, n_classes, n_queries, aux_loss = False):
        super(DETR, self).__init__()
        self.n_queries = n_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        # for output of decoder, we add a linear layer to get the class
        self.class_emb = nn.Linear(hidden_dim, n_classes + 1)

        self.bbox_emb = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_emb = nn.Embedding(n_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size = 1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """
            samples: NestedTensor
                - tensor: batched images, of shape [batch_size x C x H x W], C == 3
                - mask: binary mask of shape [batch_size x H x W], padding pixels are ignored by loss.

        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        feats, pos = self.backbone(samples)
        src, mask = feats[-1].decompose()
        assert(mask is not None)

        hs = self.transformer(self.input_proj(src), mask, self.query_emb.weight, pos[-1])[0]
            
        out_class = self.class_emb(hs)
        out_coord = self.bbox_emb(hs).sigmoid()

        out = {"pred_logits": out_class[-1], "pred_boxes": out_coord[-1]}
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(out_class, out_coord)
        return out
    
    @torch.jit.unused
    def _set_aux_loss(self, out_class, out_coord):
        # compute auxiliary loss at each decoder layer
        return [{
            "pred_logits" : a, 
            "pred_boxes": b
        } for a, b  in zip(out_class[:-1], out_coord[:-1])]
    
def build_backbone(args):
    xxx = 1

def build_transformer(args):


def build(args):
    n_class = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == 'coco_panoptic':
        n_class = 250
    device = torch.device(args.device)
    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(backbone, transformer, n_class, args.num_queries, aux_loss = args.aux_loss)
    if args.masks:
        model = DETRsegm(model, freeze_detr = (args.freeze_detr is not None) )
    matcher = build_matcher(args)
    weight_dict = {"loss_ce": 1, "loos_box": args.bbox_loss_coef}
    weight_dict["loss_giou"] = args.giou_loss_coef

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers -1):
            aux_weight_dict.update({k + f'_i': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["labels", "boxes", "cardinality"]
    if args.masks:
        losses += ["masks"]
    
    criterion = SetCriterion(n_class, matcher = matcher, weight_dict = weight_dict, 
                             eos_coef = args.eos_coef, losses = losses)
    
    criterion.to(device)
    postprocessors = {
        "bbox": PostProcess()
    }

    if args.masks:
        postprocessors["segm"] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i : i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold = 0.85)

    return model, criterion, postprocessors