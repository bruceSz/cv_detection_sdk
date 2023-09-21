#!/usr/bin/env python3


import torch
import torch.nn as nn
from einops import rearrange
from einsops.layers.torch import Rearrange

def posemb_sin_cos_2d(h, w, dim, temperature:  int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim% 4 == 0, "dimension must be divisible by 4")
    omega = torch.arange(dim//4) / (dim//4- 1)
    emega = 1.0/(temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)

        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)


    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # inner dim = dim_head * heads, 
        # n == number of patches
        # qkv = [b, n, heads, dim_head] -> [b, heads, n, dim_head]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d ', h = self.heads), qkv)
        # dots = [b, heads, n, n]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # transform to 0~1
        attn = self.atten(dots)

        # out : [b, heads, n, dim_head] 
        out = torch.matmul(attn, v)
        # change back to [b, n, heads, dim_head]
        out = rearrange(out, 'b h n d -> b n (h d)')

        # result shape of to_out : [b, n, dim]
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super(Transformer,self).__init__()

        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
    
class VIT(nn.Module):
    """
        dim: dimension of embedding
    """
    def __init__(self, *, img_size, patch_size, n_classes, dim, depth, heads, mlp_dim,
                 channels = 3, dim_head = 64):
        
        super(VIT, self).__init__()
        h, w = img_size
        patch_h, patch_w = patch_size
        assert(h % patch_h == 0 and w % patch_w == 0, "img_size must be divisible by patch_size")

        patch_dim = channels * patch_h * patch_w

        self.to_patch_emb = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_h, p2 = patch_w),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_emb = posemb_sin_cos_2d(
            h = h // patch_h,
            w = w // patch_w,
            dim = dim,
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, n_classes)


    def forward(self, img):
        device = img.device
        x = self.to_patch_emb(img)
        x += self.pos_emb.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)