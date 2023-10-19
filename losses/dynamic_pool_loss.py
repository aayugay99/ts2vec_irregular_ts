import torch
import torch.nn as nn
import torch.nn.functional as F


def pool_and_filter_by_time(z, time):
    # z: [batch_size, seq_len, dim]
    # time: [batch_size, seq_len]
    batch_size, seq_len, dim = z.shape
    pooled = F.max_pool1d(z.transpose(1,2), kernel_size=2, stride=1).transpose(1,2)
    pooled_time = (time[:, :-1] + time[:, 1:]) // 2
    
    deltas = time.diff(dim=-1)
    indices = torch.argsort(deltas, dim=-1)
    distance_below_median_mask = indices < (seq_len // 2)

    pooled = pooled[distance_below_median_mask].reshape(batch_size, seq_len // 2, dim)
    pooled_time = pooled_time[distance_below_median_mask].reshape(batch_size, seq_len // 2)

    return pooled, pooled_time

def dynamic_pool_hierarchical_contrastive_loss(z1, z2, event_time, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1, _ = pool_and_filter_by_time(z1, event_time)
        z2, event_time = pool_and_filter_by_time(z2, event_time)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d

def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss


class DynamicPoolLoss(nn.Module):
    def __init__(self, alpha=0.5, temporal_unit=0):
        super().__init__()

        self.alpha = alpha
        self.temporal_unit = temporal_unit

    def forward(self, inputs, _):
        out1, out2, deltas = inputs
        return dynamic_pool_hierarchical_contrastive_loss(out1, out2, deltas, self.alpha, self.temporal_unit)
    