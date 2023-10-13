import torch
import torch.nn as nn

from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder
from ptls.data_load.padded_batch import PaddedBatch

from .cont_dilated_conv import ContConv1d, Kernel

class DilatedContConvEncoder(nn.Module):
    def __init__(self, in_channels, nb_filters, nb_layers, kernel, kernel_size):
        super().__init__()

        self.in_channels = [in_channels] + [nb_filters] * nb_layers
        include_zero_lag = [True] + [True] * nb_layers
        self.dilation_factors = [2 ** i for i in range(0, nb_layers)]

        self.nb_layers = nb_layers
        self.nb_filters = nb_filters

        self.convs = nn.ModuleList(
            [
                ContConv1d(
                    kernel.recreate(self.in_channels[i]),
                    kernel_size,
                    self.in_channels[i],
                    nb_filters,
                    self.dilation_factors[i],
                    include_zero_lag[i],
                )
                for i in range(nb_layers)
            ]
        )
        
    def forward(self, times, features, non_pad_mask):
        for conv in self.convs:
            features = torch.nn.functional.leaky_relu(
                conv(times, features, non_pad_mask), 0.1
            )
        return features


class ContConvEncoder(AbsSeqEncoder):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=10,
                 kernel_hiddens=[8, 4, 8],
                 kernel_size=5,
                 dropout=0,
                 is_reduce_sequence=False,  
                 reducer='maxpool'
                 ):
        
        super().__init__(is_reduce_sequence=is_reduce_sequence)

        self.hidden_size = hidden_size

        self.kernel = Kernel(
            hidden1=kernel_hiddens[0],
            hidden2=kernel_hiddens[1],
            hidden3=kernel_hiddens[2],
            in_channels=input_size,
            out_channels=hidden_size
        )

        self.feature_extractor = DilatedContConvEncoder(
            in_channels=input_size,
            nb_filters=hidden_size,
            nb_layers=num_layers,
            kernel=self.kernel,
            kernel_size=kernel_size
        )

        self.repr_dropout = nn.Dropout(dropout)

        self.reducer = reducer

    def forward(self, x: PaddedBatch):  # x: B x T x input_dims   
        # cont conv encoder
        times = x.payload["event_time"].float()
        features = x.payload["embeddings"].float()

        non_pad_mask = features.ne(0)[:, :, 0]

        out = self.repr_dropout(self.feature_extractor(times, features, non_pad_mask))  # B x Co x T
        
        out = PaddedBatch(out, x.seq_lens)
        if self.is_reduce_sequence:
            out = out.payload.max(dim=1).values

        return out


class ContConvSeqEncoder(SeqEncoderContainer):
    def __init__(self,
                trx_encoder=None,
                input_size=None,
                is_reduce_sequence=False,
                **seq_encoder_params,
                ):
        super().__init__(
            trx_encoder=trx_encoder,
            seq_encoder_cls=ContConvEncoder,
            input_size=input_size,
            seq_encoder_params=seq_encoder_params,
            is_reduce_sequence=is_reduce_sequence,
        )