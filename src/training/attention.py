import torch
import torch.nn as nn
import torch.nn.functional as F  # ✅ cần thêm dòng này

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, feat_dim)
        weights = self.attn(x)  # (batch, seq_len, 1)
        weights = F.softmax(weights, dim=1)
        context = torch.sum(weights * x, dim=1)  # (batch, feat_dim)
        return context, weights
