import torch
import numpy as np

from ptls.frames.abs_module import ABSModule
from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.head import Head

from torchmetrics import MeanMetric
from losses.dynamic_pool_loss import DynamicPoolLoss


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


def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]


def mask_input(x, mask):
    shape = x.size()

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

    x[~mask] = 0

    return x


def pool_fixed_span(stepwise_embeds, timestamps, span: int, stride: int):
    batch_size, dim, seq_len = stepwise_embeds.shape
    
    pooled_embeds = []
    pooled_time = []

    for sample_i in range(batch_size):
        left = 0
        right = 0

        pooled_embeds_i = []
        pooled_time_i = []

        while right < seq_len:
            if timestamps[sample_i, right] - timestamps[sample_i, left] < span:
                right += 1
            else:
                pooled_embeds_i.append(stepwise_embeds[sample_i, :, left:right].mean(axis=-1))
                pooled_time_i.append(timestamps[sample_i, left:right].mean(axis=-1))
                left += stride
        
        pooled_embeds = torch.stack(pooled_embeds_i).reshape(1,2,0)
        pooled_time = torch.stack(pooled_time_i).reshape(1,2,0)
        
        return pooled_embeds, pooled_time

# TODO: merge this with the main pipeline
class TS2VecDynamicPool(ABSModule):
    '''The TS2Vec model'''
    
    def __init__(
        self,
        seq_encoder,
        mask_mode="binomial",
        head=None,
        loss=None,
        validation_metric=None,
        optimizer_partial=None,
        lr_scheduler_partial=None
    ):
        ''' Initialize a TS2Vec model.
        
        Args:
        '''
        if head is None:
            head = Head(use_norm_encoder=True)
        
        if loss is None:
            loss = DynamicPoolLoss(alpha=0.5, temporal_unit=0)

        self.temporal_unit = loss.temporal_unit
        self.mask_mode = mask_mode
        
        super().__init__(validation_metric,
                         seq_encoder,
                         loss,
                         optimizer_partial,
                         lr_scheduler_partial)

        self._head = head
        self.valid_loss = MeanMetric()

    def shared_step(self, x, y):
        trx_encoder = self._seq_encoder.trx_encoder
        seq_encoder = self._seq_encoder.seq_encoder 

        event_time = x.payload["event_time"]
        time_deltas = x.payload["time_delta"]
        seq_lens = x.seq_lens

        time_deltas[x.seq_len_mask == 0] = time_deltas.max()
        adjusted_event_time = torch.cumsum(time_deltas, dim=-1) + event_time[:, :1]

        x = trx_encoder(x).payload

        ts_l = x.size(1)
        crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))

        input1 = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
        input2 = take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)
        times = take_per_row(adjusted_event_time, crop_offset + crop_left, crop_right - crop_left)

        input1_masked = mask_input(input1, self.mask_mode)
        input2_masked = mask_input(input2, self.mask_mode)
        
        out1 = seq_encoder(PaddedBatch(input1_masked, seq_lens)).payload
        out1 = out1[:, -crop_l:]

        out2 = seq_encoder(PaddedBatch(input2_masked, seq_lens)).payload
        out2 = out2[:, :crop_l]
        
        if self._head is not None:
            out1 = self._head(out1)
            out2 = self._head(out2)

        return (out1, out2, times), y

    def validation_step(self, batch, _):
        augmented_sample_with_timestamps, y = self.shared_step(*batch)
        loss = self._loss(augmented_sample_with_timestamps, y)
        self.valid_loss(loss)

    def validation_epoch_end(self, outputs):
        self.log(f'valid_loss', self.valid_loss, prog_bar=True)
        # self._validation_metric.reset()

    @property
    def is_requires_reduced_sequence(self):
        return False
    
    @property
    def metric_name(self):
        return "valid_loss"
