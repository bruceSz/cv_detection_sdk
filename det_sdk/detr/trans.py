#!/usr/bin/env python3

from torch import nn
from typing import Optional, List
from torch import Tensor
import torch.nn.functional as F

def _get_act_fn(act):
    """
    Return an act according to input string
    """
    act_map = {
         "relu": F.relu,
         "gelu": F.gelu,
         "glu": F.glu
    }
    if not act in act_map:
        raise ValueError("act must be one of {}".format(act_map.keys()))
    else:
        return act_map[act]

class TransformerEncoder(nn.Module):
    def __init__(self, d_model,heads, dim_forward=2048, dropout = 0.1,
                 act='relu', normalize=False): 
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

def _get_clones(module, N):
    import copy
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
class TransformerEncoderLayer(nn.Module):
    def __init__(self, encoder, n_layers, norm=None):
        super(TransformerEncoderLayer, self).__init__()
        self.layers = _get_clones(encoder, n_layers)
        self.norm = norm

    def forward(self, src, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        out = src

        for layer in self.layers:
             out = layer(out, src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask, pos=pos)
            
        if self.norm is not None:
            out = self.norm(out)    
        return out
    

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, heads, dim_forward=2048,dropout=0.1,
                    act="relu", normalize = False):
        super(TransformerDecoder, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_forward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_forward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.act = _get_act_fn(act)
        self.normalize = normalize

    def with_pos_emb(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
        
    def forward_post(self, tgt, memory, 
                        tgt_mask: Optional[Tensor] = None,
                        memory_mask: Optional[Tensor] = None,
                        tgt_key_padding_mask: Optional[Tensor] = None,
                        memory_key_padding_mask: Optional[Tensor] = None,
                        pos: Optional[Tensor] = None,
                        query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_emb(tgt, query_pos)

        # self attn q and k <- tgt + pos(if not None)
        tgt2 = self.attn(q, k, value=tgt, attn_mask=tgt_mask,
                            key_padding_mask= tgt_key_padding_mask)[0]
        

        
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(query=self.with_pos_emb(tgt, query_pos),
                                    key = self.with_pos_emb(memory, pos),
                                    value = memory, attn_mask=memory_mask,
                                    key_padding_mask= memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_emb(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                                key_padding_mask = tgt_key_padding_mask)[0]
        
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query = self.with_pos_emb(tgt2, query_pos),
                                    key = self.with_pos_emb(memory, pos),
                                    value = memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]
        
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.act(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                tgt_key_padding_mask, memory_key_padding_mask,
                                pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask,
                                    pos, query_pos)
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self, decoder_layer, n_layers, norm=None, return_intermediate_dec = False):
        super(TransformerDecoderLayer, self).__init__()
        self.layers = _get_clones(decoder_layer, n_layers)
        self.n_layers = n_layers
        self.norm = norm
        self.return_intermediate_dec = return_intermediate_dec

    def forward(self, tgt, memory, 
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        out = tgt
        intermediate = []


        for l in self.layers:
            out = l(out, memory, tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    pos=pos, query_pos=query_pos)
            if self.return_intermediate_dec:
                intermediate.append(self.norm(out))
        
        if self.norm is not None:
            out = self.norm(out)
            if self.return_intermediate_dec:
                intermediate.pop()
                intermediate.append(out)

        if self.return_intermediate_dec:
            return torch.stack(intermediate)
        return out.unsqueeze(0)
    
    
class Transformer(nn.Module):
    def __init__(self, d_model = 512, heads=8, num_encoder=6,
                 num_dec=6, dim_forward=2048, dropout=0.1,
                 act='relu', normalize=False, return_intermediate_dec=False):
        super(Transformer, self).__init__()

        encoder = TransformerEncoder(d_model, heads, dim_forward,
                                     dropout, act, normalize)
        encoder_norm = nn.LayerNorm(d_model) if normalize else None

        
        self.encoder = TransformerEncoderLayer(encoder, num_encoder, encoder_norm)
        decoder = TransformerDecoder(d_model, heads, dim_forward, 
                                     dropout, act, normalize)
        decoder_norm = nn.LayerNorm(d_model) if normalize else None

        self.decoder = TransformerDecoderLayer(decoder, num_dec, decoder_norm,
                                               return_intermediate_dec=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.heads = heads

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                # uniform initialization
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_emb, pos_emb):
        bs, c, h, w = src.shape
        # flatten the image from dim 2.
        # src: [bs, c, h, w] -> [bs, c, h*w]
        # permute: [bs, c, h*w] -> [h*w, bs, c]
        # figure it out.
        src = src.flatten(2).permute(2, 0, 1)
        pos_emb = pos_emb.flatten(2).permute(2, 0, 1)
        query_emb = query_emb.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_emb)
        memory = self.encoder(src, src_key_padding_mask = mask, pos = pos_emb)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask, 
                          pos = pos_emb, query_pos = query_emb)
        return hs.transpose(1,2), memory.permute(1,2,0).view(bs, c, h, w)

    
