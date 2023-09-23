#!/usr/bin/env python3

from torch import nn
from typing import Optional, List
from torch import Tensor

class TransformerEncoder(nn.Module):
    def __init__(self, d_model,heads, dim_forward=2048, dropout = 0.1,
                 act='relu', normalize): 
        super(TransformerEncoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout)
        self.liear1 = nn.Linear(d_model, dim_forward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_forward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = _get_act_fn(act)
        self.normalize = normalize


    def with_pos_emb(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward_post(self,
                     src, src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # firstly we add tensor and pos_emb together.
        q = k = self.with_pos_emb(src, pos)
        # output of self_attn is a tuple of (output, attn_weights)
        src2 = self.self_attn(q, k, value = src, attn_mask=src_mask,
                              key_padding_mask =src_key_padding_mask)[0]
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
            src2 = self.norm1(src)
            q = k = self.with_pos_emb(src2, pos)
            src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.act(self.linear1(src2))))
            src = src + self.dropout2(src2)
            return src
    def forward(self, src, 
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
         if self.normalize:
              return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
         return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class Transformer(nn.Module):
    def __init__(self, d_model = 512, heads=8, num_encoder=6,
                 num_dec=6, dim_forward=2048, dropout=0.1,
                 act='relu', normalize=False, return_intermediate_dec=False):
        super(Transformer, self).__init__()

        encoder = TransformerEncoder(d_model, heads, dim_forward,
                                     dropout, act, normalize)
        