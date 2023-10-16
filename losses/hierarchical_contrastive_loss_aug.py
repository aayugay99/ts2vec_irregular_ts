import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from hierarchical_contrastive_loss import temporal_contrastive_loss


def hierarchical_contrastive_loss_aug(z1, z2, t, alpha=0.5, temporal_unit=0):
    B, T = z1.size(0), z1.size(1)
    
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss_aug(z1, z2, t)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1

        if t.size(1) % 2 != 0:
            t = t[:, :-1]
                
        t = t.reshape(B, -1, 2).float().mean(dim=2).reshape(B, -1)

        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss_aug(z1, z2, t)
        d += 1
    return loss / d


def instance_contrastive_loss_aug(z1, z2, t):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0., requires_grad=True)
    
    if t is not None:
        diff = abs(t.reshape(B*T, 1) - t.reshape(1, B*T)) 
        seq_num = torch.arange(B).unsqueeze(1).expand(B, T).reshape(-1, 1)

        mask = (seq_num - seq_num.t()) == 0
        diff[mask] += diff.max() + 1e-3

        indices = diff.topk(k=B, dim=1, largest=False).indices

        negative_pairs = torch.stack(
            [
                torch.arange(0, B*T, dtype=indices.dtype, device=indices.device).repeat_interleave(B),
                torch.cat(indices.unbind(dim=0))
            ]).t()      
        
        z1_flat, z2_flat = z1.reshape(B*T, -1), z2.reshape(B*T, -1)

        logits = torch.cat(
            [
                (z1 * z2).sum(dim=-1).reshape(B*T, -1),
                (z1_flat[negative_pairs[:, 0]] * z1_flat[negative_pairs[:, 1]]).sum(dim=1).reshape(B*T, -1),
                (z1_flat[negative_pairs[:, 0]] * z2_flat[negative_pairs[:, 1]]).sum(dim=1).reshape(B*T, -1),
            ],
            dim=1
        ) 
        
        return -F.log_softmax(logits, dim=1)[:, 0].mean()
    

class HierarchicalContrastiveLossAug(nn.Module):
    def __init__(self, alpha=0.5, temporal_unit=0):
        super().__init__()

        self.alpha = alpha
        self.temporal_unit = temporal_unit

    def forward(self, embeddings, _):
        out1, out2, t = embeddings
        return hierarchical_contrastive_loss_aug(out1, out2, t, self.alpha, self.temporal_unit)
    