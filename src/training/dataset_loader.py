# training/dataset_loader.py
# training/dataset_loader.py
import numpy as np
import torch
from torch.utils.data import Dataset
from training.augmentations import jitter, scaling, time_shift

class ECGDataset(Dataset):
    def __init__(self, file_paths, labels, augment=False, rare_class_indices=None, 
                 augment_strength="normal"):
        """
        Args:
            file_paths: Array of file paths to .npy files
            labels: Labels array (n_samples, n_classes)
            augment: Whether to apply augmentation
            rare_class_indices: List of rare class indices for stronger augmentation
            augment_strength: "normal" or "aggressive"
        """
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment
        self.rare_class_indices = rare_class_indices if rare_class_indices is not None else []
        self.augment_strength = augment_strength
        
        # Pre-identify rare samples for faster lookup
        if len(self.rare_class_indices) > 0:
            rare_mask = labels[:, self.rare_class_indices].sum(axis=1) > 0
            self.is_rare = rare_mask
        else:
            self.is_rare = np.zeros(len(labels), dtype=bool)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        signal = np.load(self.file_paths[idx]).astype(np.float32)
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

        if self.augment:
            # Apply stronger augmentation for rare classes
            if self.is_rare[idx]:
                signal = self.augment_signal_aggressive(signal)
            else:
                signal = self.augment_signal(signal)

        signal = torch.tensor(signal, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return signal, label
    
    def augment_signal(self, sig):
        """Normal augmentation for common classes"""
        sig = jitter(sig, sigma=0.01)
        sig = scaling(sig, sigma=0.1)
        sig = time_shift(sig, max_shift=0.05)
        sig = np.clip(sig, -5, 5)
        return sig
    
    def augment_signal_aggressive(self, sig):
        """
        Aggressive augmentation for rare classes
        Apply multiple augmentations with stronger parameters
        """
        if self.augment_strength == "aggressive":
            # Apply 2-3x augmentation
            num_augments = np.random.randint(2, 4)
        else:
            num_augments = 1
        
        for _ in range(num_augments):
            # 1. Stronger jittering
            sig = jitter(sig, sigma=np.random.uniform(0.01, 0.03))
            
            # 2. More aggressive scaling
            sig = scaling(sig, sigma=np.random.uniform(0.1, 0.2))
            
            # 3. Larger time shift
            sig = time_shift(sig, max_shift=np.random.uniform(0.05, 0.1))
            
            # 4. Random dropout (simulate missing data)
            if np.random.rand() > 0.5:
                dropout_mask = np.random.rand(*sig.shape) > 0.05  # 5% dropout
                sig = sig * dropout_mask
            
            # 5. Random sign flip for some channels (rare)
            if np.random.rand() > 0.8:
                flip_channels = np.random.choice(sig.shape[1], 
                                                size=max(1, sig.shape[1]//3), 
                                                replace=False)
                sig[:, flip_channels] *= -1
        
        sig = np.clip(sig, -5, 5)
        return sig


class FastECGDataset(Dataset):
    """
    Lightweight version without heavy augmentation (for speed)
    Use this if training is too slow
    """
    def __init__(self, file_paths, labels, augment=False, rare_class_indices=None):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment
        self.rare_class_indices = rare_class_indices if rare_class_indices is not None else []
        
        if len(self.rare_class_indices) > 0:
            rare_mask = labels[:, self.rare_class_indices].sum(axis=1) > 0
            self.is_rare = rare_mask
        else:
            self.is_rare = np.zeros(len(labels), dtype=bool)
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        signal = np.load(self.file_paths[idx]).astype(np.float32)
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.augment:
            signal = self.fast_augment(signal, self.is_rare[idx])
        
        signal = torch.tensor(signal, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return signal, label
    
    def fast_augment(self, sig, is_rare=False):
        """Fast augmentation - only essential operations"""
        # Noise
        noise_level = 0.03 if is_rare else 0.01
        noise = np.random.normal(0, noise_level, sig.shape)
        sig = sig + noise
        
        # Scaling
        scale = np.random.uniform(0.9, 1.1)
        sig = sig * scale
        
        # Time shift
        shift = np.random.randint(-50, 50)
        sig = np.roll(sig, shift, axis=0)
        
        sig = np.clip(sig, -5, 5)
        return sig



# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from training.augmentations import jitter, scaling, time_shift

# class ECGDataset(Dataset):
#     def __init__(self, file_paths, labels, augment=False):
#         self.file_paths = file_paths
#         self.labels = labels
#         self.augment = augment

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx):
#         signal = np.load(self.file_paths[idx]).astype(np.float32)
#         signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

#         if self.augment:
#             signal = self.augment_signal(signal)

#         signal = torch.tensor(signal, dtype=torch.float32)
#         label = torch.tensor(self.labels[idx], dtype=torch.float32)

#         return signal, label

#     """def augment_signal(self, sig):
#         shift = np.random.randint(-50, 50)
#         sig = np.roll(sig, shift, axis=1)
#         noise = np.random.normal(0, 0.01, sig.shape)
#         sig += noise.astype(np.float32)
#         sig *= np.random.uniform(0.9, 1.1)
#         sig = np.clip(sig, -5, 5)
#         return sig"""
    
#     def augment_signal(self, sig):
#         sig = jitter(sig, sigma=0.01)
#         sig = scaling(sig, sigma=0.1)
#         sig = time_shift(sig, max_shift=0.05)
#         sig = np.clip(sig, -5, 5)
#         return sig






# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from training.augmentations import jitter, scaling, time_shift

# class ECGDataset(Dataset):
#     def __init__(self, file_paths, labels, augment=False):
#         """
#         Args:
#             file_paths: Array of file paths to .npy files
#             labels: Labels array (n_samples, n_classes)
#             augment: Whether to apply augmentation
#         """
#         self.file_paths = file_paths
#         self.labels = labels
#         self.augment = augment
        
#         # Auto-detect shape from first file
#         first_signal = np.load(file_paths[0]).astype(np.float32)
#         self.original_shape = first_signal.shape
        
#         if first_signal.shape[0] < first_signal.shape[1]:
#             # Shape is (n_leads, seq_len) like (6, 5000)
#             self.needs_transpose = True
#             print(f"⚠️ Detected shape: {first_signal.shape} -> Will transpose to (seq_len, n_leads)")
#         else:
#             # Shape is (seq_len, n_leads) like (5000, 6)
#             self.needs_transpose = False
#             print(f"✅ Detected shape: {first_signal.shape} -> Correct format (seq_len, n_leads)")

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx):
#         # Load signal from .npy file
#         signal = np.load(self.file_paths[idx]).astype(np.float32)
#         signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        
#         # Transpose if needed: (n_leads, seq_len) -> (seq_len, n_leads)
#         if self.needs_transpose:
#             signal = signal.T  # (6, 5000) -> (5000, 6)
        
#         # Apply augmentation
#         if self.augment:
#             signal = self.augment_signal(signal)
        
#         # Convert to tensors
#         signal = torch.tensor(signal, dtype=torch.float32)
#         label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
#         return signal, label
    
#     def augment_signal(self, sig):
#         """Apply data augmentation (expects shape: seq_len, n_leads)"""
#         sig = jitter(sig, sigma=0.01)
#         sig = scaling(sig, sigma=0.1)
#         sig = time_shift(sig, max_shift=0.05)
#         sig = np.clip(sig, -5, 5)
#         return sig



