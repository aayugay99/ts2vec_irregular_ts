import numpy as np

import torch
from ptls.data_load.utils import collate_feature_dict


def encode_data(seq_encoder, data, batch_size=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seq_encoder.to(device)
    seq_encoder.eval()
    seq_encoder.is_reduce_sequence = True

    features, targets = [], []
    with torch.no_grad():

        for i in range(0, len(data), batch_size):
            batch = data[i: i + batch_size]
            collated_batch = collate_feature_dict(batch).to(device)

            out = seq_encoder(collated_batch).detach().cpu().numpy()
            
            features += [*out]
            targets += list(collated_batch.payload["global_target"].cpu().numpy())

    return np.array(features), np.array(targets)
