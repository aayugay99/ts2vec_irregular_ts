from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.trainer import Trainer

from ptls.preprocessing import PandasDataPreprocessor
from ptls.nn import TrxEncoder
from ptls.frames import PtlsDataModule
    
from nn.seq_encoder import ConvSeqEncoder
from modules.ts2vec_module import TS2Vec
from datasets import TS2VecDataset
from utils.encode import encode_data
from utils.evaluation import bootstrap_eval


df = pd.read_parquet("data/preprocessed_new/default_date.parquet")

preprocessor = PandasDataPreprocessor(
    col_id="user_id",
    col_event_time="timestamp",
    event_time_transformation="dt_to_timestamp",
    cols_category=["mcc_code"],
    cols_first_item=["global_target"]
)

data = preprocessor.fit_transform(df[(df["time_delta"] >= 0)])

val_size = 0.1
test_size = 0.1

y = [data[i]["global_target"] for i in range(len(data))] 

val_size = 0.1
test_size = 0.1

train, val_test, y_train, y_val_test = train_test_split(data, y, stratify=y, test_size=test_size+val_size, random_state=42)
val, test, y_val, y_test = train_test_split(val_test, y_val_test, stratify=y_val_test, test_size=test_size/(test_size+val_size), random_state=42)

train_ds = TS2VecDataset(train, min_seq_len=15)
val_ds = TS2VecDataset(val, min_seq_len=15)
test_ds = TS2VecDataset(test, min_seq_len=15)

datamodule = PtlsDataModule(
    train_data=train_ds,
    valid_data=val_ds,
    train_batch_size=16,
    valid_batch_size=16,
    train_num_workers=8,
    valid_num_workers=8
)

from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn import TrxEncoder
from ptls.data_load import PaddedBatch


class TimeTrxEncoder(TrxEncoder):
    def __init__(self, col_time="event_time", **trx_encoder_params):
        super().__init__(**trx_encoder_params)

        self.col_time = col_time

    def forward(self, x: PaddedBatch):
        embeddings = super().forward(x).payload
        timestamps = (x.payload[self.col_time] - 1444734866) / (60*60*24) * x.seq_len_mask
        return PaddedBatch({"embeddings": embeddings, self.col_time: timestamps}, x.seq_lens)

import torch
import torch.nn as nn
import torch.nn.functional as F


def hierarchical_contrastive_loss(z1, z2, t, alpha=0.5, temporal_unit=0):
    B, T = z1.size(0), z1.size(1)
    
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2, t)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

        if t.size(1) % 2 != 0:
            t = t[:, :-1]
            
        t = t.reshape(B, -1, 2).float().mean(dim=2).reshape(B, -1)

    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2, t)
        d += 1
    return loss / d

def instance_contrastive_loss(z1, z2, t):
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


class HierarchicalContrastiveLossNew(nn.Module):
    def __init__(self, alpha=0.5, temporal_unit=0):
        super().__init__()

        self.alpha = alpha
        self.temporal_unit = temporal_unit

    def forward(self, embeddings, _):
        out1, out2, t = embeddings
        return hierarchical_contrastive_loss(out1, out2, t, self.alpha, self.temporal_unit)
    
import torch
import numpy as np

from ptls.frames.abs_module import ABSModule
from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.head import Head

from torchmetrics import MeanMetric
from losses.hierarchical_contrastive_loss import HierarchicalContrastiveLoss


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


class TS2Vec(ABSModule):
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
            loss = HierarchicalContrastiveLoss(alpha=0.5, temporal_unit=0)

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

        seq_lens = x.seq_lens
        encoder_out = trx_encoder(x).payload

        x = encoder_out["embeddings"]
        t = encoder_out["event_time"]

        ts_l = x.size(1)
        crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))

        input1 = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
        input2 = take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)
        
        t = take_per_row(t, crop_offset + crop_eleft, crop_right - crop_eleft)
        
        input1_masked = mask_input(input1, self.mask_mode)
        input2_masked = mask_input(input2, self.mask_mode)
        
        out1 = seq_encoder(PaddedBatch(input1_masked, seq_lens)).payload
        out1 = out1[:, -crop_l:]

        out2 = seq_encoder(PaddedBatch(input2_masked, seq_lens)).payload
        out2 = out2[:, :crop_l]
        
        t = t[:, -crop_l:]
        
        if self._head is not None:
            out1 = self._head(out1)
            out2 = self._head(out2)

        return (out1, out2, t), y

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
    

for run in range(5):
    trx_encoder = TimeTrxEncoder(
        col_time="event_time",
        embeddings={
            "mcc_code": {"in": 345, "out": 24}
        },
        numeric_values={
            "amount": "identity"
        },
        use_batch_norm_with_lens=True,
        norm_embeddings=False,
        embeddings_noise=0.0003
    )

    seq_encoder = ConvSeqEncoder(
        trx_encoder,
        hidden_size=1024,
        num_layers=10,
        dropout=0.1,
    )

    lr_scheduler_partial = partial(torch.optim.lr_scheduler.ReduceLROnPlateau, factor=.99, patience=50)
    optimizer_partial = partial(torch.optim.Adam, lr=1e-3)

    model = TS2Vec(
        seq_encoder,
        loss=HierarchicalContrastiveLossNew(),
        optimizer_partial=optimizer_partial,
        lr_scheduler_partial=lr_scheduler_partial
    )

    checkpoint = ModelCheckpoint(
        monitor="valid_loss", 
        mode="min"
    )

    trainer = Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=[1],
        callbacks=[checkpoint],
    )

    trainer.fit(model, datamodule)

    model.load_state_dict(torch.load(checkpoint.best_model_path)["state_dict"])
    torch.save(model.seq_encoder.state_dict(), f"ts2vec_default_new_instance_{run}.pth")
