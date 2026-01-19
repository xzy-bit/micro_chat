import torch
import torch.nn as nn


def precompute_rotate_matrix(dim: int, end: int=int(32*1024),base:float=1e6):
    # 每一个维度对应的都有一个位置编码, 这里的角度wi 对应的是 (base) ** (2i//dim)
    # 由于在旋转编码中需要将原有的空间划分为多个维度为2的子空间
    # 因此对应的i为 [0,0,1,1,...]
    freqs = 1.0/(base**(torch.arange(0,dim,2)[:dim//2].float()/dim))
    pos = torch.arange(end,device=freqs.device)
    freqs = torch.outer(pos,freqs).float() # [end, dim/2]
    freqs_cos = torch.cat([torch.cos(freqs),torch.cos(freqs)],dim=-1) # [end, dim]
    freqs_sin = torch.cat([torch.sin(freqs),torch.sin(freqs)],dim=-1) # [end, dim]
    return freqs_cos,freqs_sin

def apply_rotary_pos_emd(q,k,cos,sin):
    def rotate_half(x):
        return torch.cat((-x[...,x.shape[-1]//2:],x[...,:x.shape[-1]//2]),dim=-1)

    # q = [B, T, H, D]
    # cos = [T, D]
    # 对每个batch , 每个head都生效
    q_embed = q * cos.unsqueeze(dim=1) + rotate_half(q) * sin.unsqueeze(dim=1)
    k_embed = k * cos.unsqueeze(dim=1) + rotate_half(k) * sin.unsqueeze(dim=1)
    return q_embed,k_embed