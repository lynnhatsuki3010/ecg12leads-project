# training/dataset_loader.py
import numpy as np
import torch
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    def __init__(self, file_paths, labels, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        signal = np.load(self.file_paths[idx]).astype(np.float32)
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

        if self.augment:
            signal = self.augment_signal(signal)

        signal = torch.tensor(signal, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return signal, label

    def augment_signal(self, sig):
        shift = np.random.randint(-50, 50)
        sig = np.roll(sig, shift, axis=1)
        noise = np.random.normal(0, 0.01, sig.shape)
        sig += noise.astype(np.float32)
        sig *= np.random.uniform(0.9, 1.1)
        sig = np.clip(sig, -5, 5)
        return sig
