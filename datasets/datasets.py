from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import SampleSlices

from ptls.data_load.datasets.memory_dataset import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.data_load.utils import collate_feature_dict

from torch.utils.data import Dataset


class TS2VecDataset(Dataset):
    def __init__(self, data, min_seq_len=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = MemoryMapDataset(data, [SeqLenFilter(min_seq_len)])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        feature_arrays = self.data[item]
        return feature_arrays
    
    @staticmethod
    def collate_fn(batch):
        return collate_feature_dict(batch), None


class CustomColesDataset(ColesDataset):
    def __init__(self, data, split_count, cnt_min, cnt_max, min_seq_len=None):
        super().__init__(
            data=MemoryMapDataset(data, [SeqLenFilter(min_seq_len)]),
            splitter=SampleSlices(split_count, cnt_min, cnt_max)
        )
