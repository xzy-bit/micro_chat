import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self,dim_in, dim_q, dim_v,bias=False):
        super(SelfAttention,self).__init__()
        self.dim_q = dim_q
        self.W_q = nn.Linear(dim_in,dim_q)
        self.W_k = nn.Linear(dim_in,dim_q)
        self.W_v = nn.Linear(dim_in,dim_v)

    def forward(self,x,mask = None):
        # bsz, seq_len, hidden_dim = x.shape

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        attn_logits = (q@k.transpose(-1,-2)/math.sqrt(self.dim_q))
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn_logits.masked_fill_(mask,float('-inf'))

        attn_score = F.softmax(attn_logits,dim=-1)

        return attn_score @ v

class MultiHeadAttention(nn.Module):
    def __init__(self,hidden_dim, head_num, dim_q, dim_o,bias=False):
        super(MultiHeadAttention,self).__init__()
        self.dim_q = dim_q
        self.head_num = head_num
        assert hidden_dim == head_num * dim_q
        self.W_q = nn.Linear(hidden_dim,head_num*dim_q,bias=bias)
        self.W_k = nn.Linear(hidden_dim,head_num*dim_q,bias=bias)
        self.W_v = nn.Linear(hidden_dim,head_num*dim_q,bias=bias)
        self.W_o = nn.Linear(head_num*dim_q,dim_o,bias=bias)


    def forward(self,x,mask=None):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        batch_size, seq_len, hidden_dim = x.shape
        # reshape只能够切块，不能够在语义上单独计算
        q = Q.reshape(batch_size,seq_len,self.head_num,self.dim_q).transpose(1,2)
        k = K.reshape(batch_size,seq_len,self.head_num,self.dim_q).transpose(1,2)
        v = V.reshape(batch_size,seq_len,self.head_num,self.dim_q).transpose(1,2)

        attn_logits = (q@k.transpose(-1,-2)/math.sqrt(self.dim_q))

        if mask is not None:
            # mask_shape = [B, L]
            # attn_score = [B, H, L, L]
            mask = mask.reshape(batch_size,1,1,seq_len)
            attn_logits.masked_fill_(mask,float("-inf"))

        attn_score = F.softmax(attn_logits, dim=-1)
        v_out = attn_score @ v
        # v_out [B, H, L, D]
        v_out = v_out.transpose(1,2).reshape(batch_size,seq_len,-1)
        return self.W_o(v_out)

class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_dim, head_q,head_kv, head_dim,dim_o,bias=False):
        super().__init__()
        assert head_q % head_kv == 0
        self.head_q = head_q
        self.head_kv = head_kv
        self.head_dim = head_dim
        self.q_per_kv = head_q // head_kv
        self.W_q = nn.Linear(hidden_dim, head_q * head_dim, bias)
        self.W_k = nn.Linear(hidden_dim, head_kv * head_dim, bias)
        self.W_v = nn.Linear(hidden_dim, head_kv * head_dim, bias)
        self.W_o = nn.Linear(head_q * head_dim, dim_o, bias)

    def forward(self, x, mask=None):
        B, L, _ = x.shape

        Q = self.W_q(x)                # [B, L, head_q * head_dim]
        K = self.W_k(x)                # [B, L, head_kv * head_dim]
        V = self.W_v(x)                # [B, L, head_kv * head_dim]

        # q: [B, head_q, L, head_dim]
        q = Q.reshape(B,L,self.head_q,self.head_dim).transpose(1,2)
        k = K.reshape(B,L,self.head_kv,self.head_dim).transpose(1,2)
        v = V.reshape(B,L,self.head_kv,self.head_dim).transpose(1,2)

        # Expand KV heads to match Q heads: repeat each kv head q_per_kv times
        # k,v: [B,Hkv,L,D] -> [B,Hq,L,D]
        if self.head_kv>1:
            k = k.repeat_interleave(self.q_per_kv,dim=1)
            v = v.repeat_interleave(self.q_per_kv,dim=1)

        # attn_logits [B,Hq,L,L]
        attn_logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn_logits, dim=-1)

        v = attn @ v                  # [B, Hq, L, head_dim]
        o = v.transpose(1, 2).reshape(B, L, -1)
        return self.W_o(o)

# if __name__ == "__main__":
#     self_attn = SelfAttention(10,20,20)
#     x = torch.ones((2,5,10))
#
#     mask = torch.ones((2,5),dtype=bool)
#     print(self_attn(x).shape)
#     print(self_attn(x,mask).shape)
#
#     multi_head_attn = MultiHeadAttention(24,2,12,36)
#     x = torch.ones((2,5,24))
#     print(multi_head_attn(x).shape)
#     mask = torch.ones((2,5),dtype=bool)
#     print(multi_head_attn(x,mask).shape)
#
#     group_query_attn = GroupedQueryAttention(hidden_dim=24,head_q=12,head_kv=4,head_dim=2,dim_o=36)
#     x = torch.ones((2,5,24))
#     print(group_query_attn(x).shape)