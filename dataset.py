import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MovingMNIST(Dataset):
    def __init__(self, data_path, total_length=20, input_length=10,
                 is_train=True, transform=None):
        self.total_length = total_length
        self.input_length = input_length
        self.is_train = is_train
        self.transform = transform

        data = np.load(data_path)
        if is_train:
            self.data = data['train']
        else:
            self.data = data['test']

        self.num_samples = self.data.shape[1]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = self.data[:, idx]
        seq = seq[:self.total_length]
        seq = seq.astype(np.float32) / 255.0
        seq = torch.from_numpy(seq).unsqueeze(1)
        return seq


class RadarEcho(Dataset):
    def __init__(self, data_path, total_length=20, input_length=10,
                 is_train=True, patch_size=1):
        self.total_length = total_length
        self.input_length = input_length
        self.is_train = is_train
        self.patch_size = patch_size

        data = np.load(data_path)
        if is_train:
            self.data = data['train']
        else:
            self.data = data['test']

        self.num_samples = self.data.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = self.data[idx]
        seq = seq[:self.total_length]
        seq = seq.astype(np.float32)
        if seq.ndim == 3:
            seq = seq[:, np.newaxis, :, :]
        seq = torch.from_numpy(seq)
        return seq


def get_dataloader(dataset, batch_size, shuffle=True, num_workers=0,
                   prefetch_factor=2):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = prefetch_factor
    return DataLoader(dataset, **kwargs)
