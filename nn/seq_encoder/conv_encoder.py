import numpy as np

import torch
import torch.nn as nn

from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder
from ptls.data_load.padded_batch import PaddedBatch

from dilated_conv import DilatedConvEncoder


def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class ConvEncoder(AbsSeqEncoder):
    def __init__(self,
                 input_size=None,
                 hidden_size=None,
                 num_layers=10,
                 mask_mode='binomial',
                 dropout=0,
                 is_reduce_sequence=False,  
                 reducer='maxpool'
                 ):
        
        super().__init__(is_reduce_sequence=is_reduce_sequence)

        self.hidden_size = hidden_size
        self.mask_mode = mask_mode

        self.feature_extractor = DilatedConvEncoder(
            input_size,
            [hidden_size] * num_layers + [hidden_size],
            kernel_size=3
        )

        self.repr_dropout = nn.Dropout(dropout)

        self.reducer = reducer

    def forward(self, x: PaddedBatch, mask=None):  # x: B x T x input_dims
        shape = x.payload.size()

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(shape[0], shape[1]).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(shape[0], shape[1]).to(x.device)
        elif mask == 'all_true':
            mask = x.payload.new_full((shape[0], shape[1]), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.payload.new_full((shape[0], shape[1]), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.payload.new_full((shape[0], shape[1]), True, dtype=torch.bool)
            mask[:, -1] = False
        
        x_masked = x.payload
        x_masked[~mask] = 0
        
        # conv encoder
        x_masked = x_masked.transpose(1, 2)  # B x Ch x T
        out = self.repr_dropout(self.feature_extractor(x_masked))  # B x Co x T
        out = x_masked.transpose(1, 2)  # B x T x Co
        
        out = PaddedBatch(out, x.seq_lens)
        if self.is_reduce_sequence:
            out = out.payload.max(dim=1).values

        return out


class ConvSeqEncoder(SeqEncoderContainer):
    def __init__(self,
                trx_encoder=None,
                input_size=None,
                is_reduce_sequence=False,
                **seq_encoder_params,
                ):
        super().__init__(
            trx_encoder=trx_encoder,
            seq_encoder_cls=ConvEncoder,
            input_size=input_size,
            seq_encoder_params=seq_encoder_params,
            is_reduce_sequence=is_reduce_sequence,
        )
