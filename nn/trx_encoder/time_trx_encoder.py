from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn import TrxEncoder
from ptls.data_load import PaddedBatch


class TimeTrxEncoder(TrxEncoder):
    def __init__(self, col_time="event_time", **trx_encoder_params):
        super().__init__(**trx_encoder_params)

        self.col_time = col_time

    def forward(self, x: PaddedBatch):
        embeddings = super().forward(x).payload
        timestamps = x.payload[self.col_time]
        return PaddedBatch({"embeddings": embeddings, self.col_time: timestamps}, x.seq_lens)
