import torch
import numpy as np

from ptls.frames.abs_module import ABSModule
from ptls.data_load.padded_batch import PaddedBatch

from torchmetrics import MeanMetric
from losses.hierarchical_contrastive_loss import HierarchicalContrastiveLoss


def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]


class TS2Vec(ABSModule):
    '''The TS2Vec model'''
    
    def __init__(
        self,
        seq_encoder,
        loss=None,
        validation_metric=None,
        optimizer_partial=None,
        lr_scheduler_partial=None
    ):
        ''' Initialize a TS2Vec model.
        
        Args:
        '''
        
        if loss is None:
            loss = HierarchicalContrastiveLoss(alpha=0.5, temporal_unit=0)

        self.temporal_unit = loss.temporal_unit
        
        # if validation_metric is None:

        super().__init__(validation_metric,
                         seq_encoder,
                         loss,
                         optimizer_partial,
                         lr_scheduler_partial)

        self.valid_loss = MeanMetric()

    def shared_step(self, x, y):
        trx_encoder = self._seq_encoder.trx_encoder
        seq_encoder = self._seq_encoder.seq_encoder 

        seq_lens = x.seq_lens
        x = trx_encoder(x).payload

        ts_l = x.size(1)
        crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))

        input1 = PaddedBatch(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft), seq_lens)
        input2 = PaddedBatch(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left), seq_lens)
        
        out1 = seq_encoder(input1).payload
        out1 = out1[:, -crop_l:]
                
        out2 = seq_encoder(input2).payload
        out2 = out2[:, :crop_l]
        
        return (out1, out2), y

    def validation_step(self, batch, _):
        y_h, y = self.shared_step(*batch)
        loss = self._loss(y_h, y)
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
