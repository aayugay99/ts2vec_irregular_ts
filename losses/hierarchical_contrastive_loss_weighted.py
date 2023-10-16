import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hierarchical_contrastive_loss import instance_contrastive_loss, temporal_contrastive_loss


def hierarchical_contrastive_loss_weighted(z1, z2, t, h=1, sqrt=True, alpha=0.5, temporal_unit=0):
    B, T = z1.size(0), z1.size(1)
    
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1

        if t.size(1) % 2 != 0:
            t = t[:, :-1]

        delta = torch.clip(t.reshape(B, -1, 2)[:, :, 1] - t.reshape(B, -1, 2)[:, :, 0], min=0)
                
        t = t.reshape(B, -1, 2).float().mean(dim=2).reshape(B, -1)

        mult = np.sqrt(d) if sqrt else d

        weights = torch.exp(-delta / (h*mult)).unsqueeze(2)

        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2) * weights
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2) * weights

    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d


class HierarchicalContrastiveLossWeighted(nn.Module):
    def __init__(self, h=1, sqrt=True, alpha=0.5, temporal_unit=0):
        super().__init__()

        self.sqrt = sqrt
        self.h = h
        self.alpha = alpha
        self.temporal_unit = temporal_unit

    def forward(self, embeddings, _):
        out1, out2, t = embeddings
        return hierarchical_contrastive_loss_weighted(out1, out2, t, self.h, self.sqrt, self.alpha, self.temporal_unit)
    