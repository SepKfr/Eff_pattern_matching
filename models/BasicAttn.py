import torch
import torch.nn as nn


class BasicAttn(nn.Module):

    def __init__(self, d_k, device):

        super(BasicAttn, self).__init__()
        self.device = device
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):

        scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) / np.sqrt(self.d_k)
        attn = torch.softmax(scores, -1)
        context = torch.einsum('bhqk,bhvd->bhqd', attn, V)
        return context, attn