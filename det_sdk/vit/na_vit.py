#!/usr/bin/env python3

from functools import partial
from torch.nn.utils.rnn import pad_sequence as orig_pad_sequence
from torch.nn import functional as F
import torch
from torch import nn
from torch import Tensor
from einops.layers.torch import Rearrange
from einops import rearrange
from einops import repeat

from typing import Union, List


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(t, divisor):
    return (t % divisor) == 0

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)
       

class RMSNorm(nn.Module):
    def __init__(self, heads, dim):
        super(RMSNorm, self).__init__()
        self.scale = dim ** 0.5
        # to align with input shape of [ b, h, n, dim ]
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        # x: b, heads, n, dim
        normed = F.normalize(x, dim=-1)
        return normed * self.scale * self.gamma

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class Attention(nn.Module):
    """ 
        dim: embedding dimension
        heads: number of heads
        dim_head: dimension of each head
    """
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.norm = LayerNorm(dim)

        self.q_norm = RMSNorm(heads, dim_head)
        self.k_norm = RMSNorm(heads, dim_head)

        self.atten = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential([
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        ])



    def forward(self, x, context=None, mask=None, attn_mask=None):
        """ 
            Compared with simpleVit, the differences here is mask and attn_mask.
            mask: mask for the input x, with shape [b, n], where n is the number of patches
                (mask input x means to skip that patch)
            mask: mask for the attention, with shape [b, h, n, n]
        """
        # x : b, n, dim
        x = self.norm(x)
        kv_input = default(context, x)
        # q: b, n, heads, dim_head
        # k: b, n, heads, dim_head
        # v: b, n, heads, dim_head

        qkv = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1))
        q, k, v = map(lambda t:rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # q: (b, n, heads, n,  dim_head)
        # k: (b, n, heads, n, dim_head)
        dots = torch.matmul(q, k.transpose(-1, -2))

        # dots: (b, heads, n, n)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            # for masked position, set to -inf
            dots = dots.masked_fill(~mask, -torch.finfo(dots.dtype).max)
        
        if exists(attn_mask):
            dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max)

        # activate it .
        attn = self.atten(dots)
        attn = self.dropout(attn)



        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

def always(val):
    return lambda *args: val

def grouping_img_by_max_seq_len(imgs :List[Tensor], 
                                patch_size : int,
                                calc_token_dropout = None,
                                  max_seq_len =2048) -> List[List[Tensor]]:
    """ 
        Input: imgs not grouped by max_seq_len,
        Output: imgs grouped by max_seq_len
    """
    calc_token_dropout = default(calc_token_dropout, lambda h, w: always(0.))

    groups = []
    group = []
    seq_len = 0

    if isinstance(calc_token_dropout, (float, int)):
        calc_token_dropout = always(calc_token_dropout)
    
    for img in imgs:
        assert(isinstance(img, Tensor))
        #img: c, h, w
        img_dim = img.shape[-2:]

        ph, pw = map(lambda t: t// patch_size, img_dim)

        img_seq_len = (ph * pw)
        img_seq_len = int(img_seq_len * (1-calc_token_dropout(*img_dim)))

        assert(img_seq_len <= max_seq_len, "max_seq_len is too small,  img seq len exceed it.")

        if (seq_len + img_seq_len) > max_seq_len:
            groups.append(group)
            group = []
            seq_len = 0
        group.append(img)
        seq_len += img_seq_len

    if len(group) > 0:
        groups.append(group)

    return groups

    


def FeedForward(dim, hidden_dim, dropout = 0.5):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout)
    )


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super(Transformer,self).__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
        self.norm = LayerNorm(dim)

    def forward(self, x, mask=None, attn_mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask, attn_mask=attn_mask) + x
            x = ff(x) + x
        return self.norm(x)
    
class NaVit(nn.Module):
    def __init__(self, *, img_size,patch_size, n_class, dim, depth, heads, mlp_dim, 
                 channels=3, dim_head=64, dropout=0., emb_dropout = 0., token_dropout_prob=None):
        super().__init__()
        h, w = pair(img_size)

        self.calc_token_dropout = None

        if callable(token_dropout_prob):
            self.calc_token_dropout = token_dropout_prob
        elif isinstance(token_dropout_prob, (float, int)):
            assert(0. < token_dropout_prob < 1.)
            token_dropout_prob = float(token_dropout_prob)
            self.calc_token_dropout = lambda h, w: token_dropout_prob

        assert(divisible_by(h, patch_size) and divisible_by(w, patch_size), "Image dimension must be divisible by the patch size")

        patch_h, patch_w = h // patch_size, w // patch_size

        patch_dim = channels * patch_size ** 2

        self.channels = channels
        self.patch_size = patch_size

        self.to_patch_emb = nn.Sequential([
            LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            LayerNorm(dim),
        ])

        self.pos_emb_h = nn.Parameter(torch.randn(patch_dim, dim))
        self.pos_emb_w = nn.Parameter(torch.randn(patch_dim, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim , depth, 
                                       heads, dim_head, mlp_dim,
                                       dropout=dropout)
        
        self.attn_pool_queries = nn.Parameter(torch.randn(dim))
        self.attn_pool = Attention(dim = dim, dim_head = dim_head, 
                                   heads = heads)
        
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential([
            LayerNorm(dim),
            nn.Linear(dim, n_class, bias=False)
        ])

    @property
    def device(self):
        return next(self.parameters()).device
    

    def forward(self,  
                # either list of Tensor with same shape, or
                # list of list of Tensor with same shape (same within inner list, different across inner list)))
                batched_imgs: Union[List[Tensor], List[List[Tensor]]],
                group_imgs = False, 
                group_max_seq_len = 2048):
        p, c , device, has_token_dropput = self.patch_size, self.channels, self.device, exists(self.calc_token_dropout)
        arange = partial(torch.arange, device = device)
        pad_seq = partial(orig_pad_sequence, batch_first = True)

        if group_imgs:
            batched_imgs = grouping_img_by_max_seq_len(
                batched_imgs, 
                patch_size = self.patch_size,
                calc_token_dropout = self.calc_token_dropout,
                max_seq_len = group_max_seq_len
            )
        
        num_imgs = []
        batched_seq = []
        batched_pos = []
        batched_img_ids = []

        for imgs in batched_imgs:
            num_imgs.append(len(imgs))

            seqs = []
            poss = []
            img_ids = torch.empty((0,), device=device, dtype=torch.long)

            #for this batch of imgs,
            # total seq len is max_seq_len
            for img_id, img  in enumerate(imgs):
                assert(img.ndim==3 and img.shape[0] == c)
                #img shape: c, h, w
                img_dim = img.shape[-2:]
                assert(all([divisible_by(dim,p) for dim in img_dim]), "height and width must be divisible by dim")

                ph, pw = map(lambda dim: dim//p, img_dim)


                # for torch 2.0 indexing is not supported anymore.
                pos = torch.stack(torch.meshgrid(
                    (arange(ph), arange(pw))
                ), indexing="ij", dim=-1)

                # this will flatten height and width to 1d array, each with a 
                # c: postion of each patch(i,j)
                pos = rearrange(pos, "h w c -> (h w) c")
                seq = rearrange(img, "c (h p1) (w p2) -> (h w) (c p1 p2)")
               
                seq_len = seq.shape[-2]

                if has_token_dropput:
                    token_dropout = self.calc_token_dropout(*img_dim)
                    num_keep = max(1, int(seq * (1 -token_dropout)))
                    keep_indices = torch.randn((seq_len)).topk(num_keep, dim=-1)

                    seq = seq[keep_indices]
                    pos = pos[keep_indices]
                # all pos in second dim is set to img_id
                img_ids = F.pad(img_ids, (0, seq.shape[-2]), value=img_id)
                seqs.append(seq)
                poss.append(pos)

            batched_img_ids.append(img_ids)
            batched_seq(torch.cat(seqs, dim=0))
            batched_pos(torch.cat(poss, dim=0))

        
        # TODO figure it out.
        lengths = torch.tensor([seq.shape[-2] for seq in batched_seq], device=device, dtype=torch.long)
        max_len = arange(lengths.amax().item())
        key_pad_mask = rearrange(lengths, 'b -> b 1') <= rearrange(max_len, 'n -> 1 n')

        # TODO figure it out.
        batched_img_ids = pad_seq(batched_img_ids)
        attn_mask = rearrange(batched_img_ids, 'b i -> b 1 i 1') == rearrange(batched_img_ids, 'b j -> b 1 1 j')
        attn_mask = attn_mask & rearrange(key_pad_mask, 'b j -> b 1 1 j')


        patches = pad_seq(batched_seq)
        patch_positions = pad_seq(batched_pos)

        num_imgs = torch.tensor(num_imgs, device=device, dtype = torch.long)

        x = self.to_patch_emb(patches)
        
        h_indices, w_indices = patch_positions.unbind(dim=-1)

        h_pos = self.pos_emb_h[h_indices]
        w_pos = self.pos_emb_w[w_indices]

        x = x + h_pos + w_pos


        x = self.dropout(x)

        x = self.transformer(x, attn_mask = attn_mask)

        max_queries = num_imgs.amax().item()

        queries = repeat(self.attn_pool_queries, 'd -> b n d', n = max_queries, b = x.shape[0])


        img_id_arange = arange(max_queries)
        attn_pool_mask = rearrange(img_id_arange, 'i -> i 1') == rearrange(batched_img_ids, "b j -> b 1 j")
        attn_pool_mask = attn_pool_mask & rearrange(key_pad_mask, 'b j -> b 1 j')

        attn_pool_mask = rearrange(attn_pool_mask, 'b i j -> b 1 i j')


        # attention pool
        x = self.attn_pool(queries, content = x, attn_mask = attn_pool_mask) + queries

        x = rearrange(x, 'b n d -> (b n ) d')


        is_images = img_id_arange < rearrange(num_imgs, 'b -> b 1')
        x = x[is_images]

        x = self.to_latent(x)

        return self.mlp_head(x)









        

