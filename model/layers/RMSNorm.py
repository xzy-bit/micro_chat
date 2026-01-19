import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int):
        super(RMSNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.eps = 1e-5

    def forward(self, x):
        return self.gamma * x * torch.rsqrt(torch.sqrt(x.pow(2).mean(dim=-1,keepdim=True))+self.eps)
