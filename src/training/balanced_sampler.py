import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

class BalancedMultiLabelSampler(Sampler):
    """
    Oversample rare classes while maintaining all original samples
    Enhanced with adaptive oversampling based on class rarity
    """
    def __init__(self, labels, oversample_factor=3.0, adaptive=True):
        """
        Args:
            labels: (n_samples, n_classes) binary array
            oversample_factor: Base oversampling factor for rare classes
            adaptive: If True, very rare classes get even more oversampling
        """
        self.labels = labels
        self.num_samples = len(labels)
        self.num_classes = labels.shape[1]
        
        # Calculate sample frequency per class
        class_counts = labels.sum(axis=0)
        
        # Identify rare classes (less than 10% of dataset)
        rare_threshold = self.num_samples * 0.1
        self.rare_classes = np.where(class_counts < rare_threshold)[0]
        
        # Identify VERY rare classes (less than 5% of dataset)
        very_rare_threshold = self.num_samples * 0.05
        very_rare_classes = np.where(class_counts < very_rare_threshold)[0]
        
        # Find samples containing rare classes
        rare_sample_mask = labels[:, self.rare_classes].sum(axis=1) > 0
        self.rare_indices = np.where(rare_sample_mask)[0]
        self.common_indices = np.where(~rare_sample_mask)[0]
        
        # Adaptive oversampling
        if adaptive and len(very_rare_classes) > 0:
            # Very rare classes get 2x more oversampling
            very_rare_mask = labels[:, very_rare_classes].sum(axis=1) > 0
            very_rare_indices = np.where(very_rare_mask)[0]
            
            # Normal rare samples
            normal_rare_indices = np.setdiff1d(self.rare_indices, very_rare_indices)
            
            n_oversample_normal = int(len(normal_rare_indices) * (oversample_factor - 1))
            n_oversample_very_rare = int(len(very_rare_indices) * (oversample_factor * 2 - 1))
            
            # Create final indices
            self.indices = np.concatenate([
                np.arange(self.num_samples),  # All original samples
                np.random.choice(normal_rare_indices, size=max(0, n_oversample_normal), replace=True),
                np.random.choice(very_rare_indices, size=max(0, n_oversample_very_rare), replace=True)
            ])
            
            print(f"📊 Adaptive Balanced Sampler Stats:")
            print(f"   Total samples: {len(self.indices)} (original: {self.num_samples})")
            print(f"   Rare classes: {self.rare_classes}")
            print(f"   Very rare classes: {very_rare_classes}")
            print(f"   Very rare oversampling: {oversample_factor * 2:.1f}x")
            print(f"   Normal rare oversampling: {oversample_factor:.1f}x")
        else:
            # Standard oversampling
            n_oversample = int(len(self.rare_indices) * (oversample_factor - 1))
            
            self.indices = np.concatenate([
                np.arange(self.num_samples),
                np.random.choice(self.rare_indices, size=n_oversample, replace=True)
            ])
            
            print(f"📊 Balanced Sampler Stats:")
            print(f"   Total samples: {len(self.indices)} (original: {self.num_samples})")
            print(f"   Rare classes: {self.rare_classes}")
            print(f"   Rare samples: {len(self.rare_indices)} → oversampled {n_oversample} times")
        
        np.random.shuffle(self.indices)
    
    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)

class AugmentedRareClassDataset(Dataset):
    """
    Apply MORE augmentation to rare class samples
    """
    def __init__(self, base_dataset, labels, rare_class_indices, augment_multiplier=2):
        """
        Args:
            base_dataset: Original ECGDataset
            labels: (n_samples, n_classes)
            rare_class_indices: List of rare class indices
            augment_multiplier: Apply augmentation this many times for rare samples
        """
        self.base_dataset = base_dataset
        self.labels = labels
        self.rare_class_indices = rare_class_indices
        self.augment_multiplier = augment_multiplier
        
        # Identify rare samples
        rare_mask = labels[:, rare_class_indices].sum(axis=1) > 0
        self.is_rare = rare_mask
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        signal, label = self.base_dataset[idx]
        
        # Apply STRONGER augmentation for rare class samples
        if self.is_rare[idx] and self.base_dataset.augment:
            for _ in range(self.augment_multiplier):
                signal_np = signal.numpy()
                signal_np = self.base_dataset.augment_signal(signal_np)
                signal = torch.tensor(signal_np, dtype=torch.float32)
        
        return signal, label

def create_balanced_dataloader(dataset, labels, batch_size=32, 
                                use_sampler=True, oversample_factor=3.0,
                                adaptive=True):
    """
    Create a balanced dataloader with oversampling
    
    Args:
        dataset: ECGDataset instance
        labels: (n_samples, n_classes) binary array
        batch_size: Batch size
        use_sampler: Whether to use balanced sampler
        oversample_factor: Base oversampling factor for rare classes
        adaptive: If True, very rare classes get 2x more oversampling
    
    Usage:
        train_loader = create_balanced_dataloader(
            train_dataset, y_train, batch_size=32, 
            use_sampler=True, oversample_factor=5.0, adaptive=True
        )
    """
    if use_sampler:
        sampler = BalancedMultiLabelSampler(
            labels, oversample_factor=oversample_factor, adaptive=adaptive
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, 
            sampler=sampler, num_workers=0, pin_memory=True
        )
    else:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, 
            shuffle=True, num_workers=0, pin_memory=True
        )
    
    return loader

# === Example Usage ===
"""
from training.dataset_loader import ECGDataset
from training.balanced_sampler import create_balanced_dataloader

# Create dataset
train_dataset = ECGDataset(train_files, y_train, augment=True)

# Create balanced dataloader
train_loader = create_balanced_dataloader(
    train_dataset, y_train, 
    batch_size=32, 
    use_sampler=True, 
    oversample_factor=3.0  # Oversample rare classes 3x
)
"""



# import numpy as np
# import torch
# from torch.utils.data import Dataset, Sampler

# class BalancedMultiLabelSampler(Sampler):
#     """
#     Oversample rare classes while maintaining all original samples
#     """
#     def __init__(self, labels, oversample_factor=3.0):
#         """
#         Args:
#             labels: (n_samples, n_classes) binary array
#             oversample_factor: How many times to oversample rare classes
#         """
#         self.labels = labels
#         self.num_samples = len(labels)
#         self.num_classes = labels.shape[1]
        
#         # Calculate sample frequency per class
#         class_counts = labels.sum(axis=0)
        
#         # Identify rare classes (less than 10% of dataset)
#         rare_threshold = self.num_samples * 0.1
#         self.rare_classes = np.where(class_counts < rare_threshold)[0]
        
#         # Find samples containing rare classes
#         rare_sample_mask = labels[:, self.rare_classes].sum(axis=1) > 0
#         self.rare_indices = np.where(rare_sample_mask)[0]
#         self.common_indices = np.where(~rare_sample_mask)[0]
        
#         # Calculate oversampling ratio
#         n_oversample = int(len(self.rare_indices) * (oversample_factor - 1))
        
#         # Create final indices
#         self.indices = np.concatenate([
#             np.arange(self.num_samples),  # All original samples
#             np.random.choice(self.rare_indices, size=n_oversample, replace=True)  # Extra rare samples
#         ])
        
#         np.random.shuffle(self.indices)
        
#         print(f"📊 Balanced Sampler Stats:")
#         print(f"   Total samples: {len(self.indices)} (original: {self.num_samples})")
#         print(f"   Rare classes: {self.rare_classes}")
#         print(f"   Rare samples: {len(self.rare_indices)} → oversampled {n_oversample} times")
    
#     def __iter__(self):
#         return iter(self.indices)
    
#     def __len__(self):
#         return len(self.indices)

# class AugmentedRareClassDataset(Dataset):
#     """
#     Apply MORE augmentation to rare class samples
#     """
#     def __init__(self, base_dataset, labels, rare_class_indices, augment_multiplier=2):
#         """
#         Args:
#             base_dataset: Original ECGDataset
#             labels: (n_samples, n_classes)
#             rare_class_indices: List of rare class indices
#             augment_multiplier: Apply augmentation this many times for rare samples
#         """
#         self.base_dataset = base_dataset
#         self.labels = labels
#         self.rare_class_indices = rare_class_indices
#         self.augment_multiplier = augment_multiplier
        
#         # Identify rare samples
#         rare_mask = labels[:, rare_class_indices].sum(axis=1) > 0
#         self.is_rare = rare_mask
        
#     def __len__(self):
#         return len(self.base_dataset)
    
#     def __getitem__(self, idx):
#         signal, label = self.base_dataset[idx]
        
#         # Apply STRONGER augmentation for rare class samples
#         if self.is_rare[idx] and self.base_dataset.augment:
#             for _ in range(self.augment_multiplier):
#                 signal_np = signal.numpy()
#                 signal_np = self.base_dataset.augment_signal(signal_np)
#                 signal = torch.tensor(signal_np, dtype=torch.float32)
        
#         return signal, label

# def create_balanced_dataloader(dataset, labels, batch_size=32, 
#                                 use_sampler=True, oversample_factor=3.0):
#     """
#     Create a balanced dataloader with oversampling
    
#     Usage:
#         train_loader = create_balanced_dataloader(
#             train_dataset, y_train, batch_size=32, 
#             use_sampler=True, oversample_factor=3.0
#         )
#     """
#     if use_sampler:
#         sampler = BalancedMultiLabelSampler(labels, oversample_factor=oversample_factor)
#         loader = torch.utils.data.DataLoader(
#             dataset, batch_size=batch_size, 
#             sampler=sampler, num_workers=4, pin_memory=True
#         )
#     else:
#         loader = torch.utils.data.DataLoader(
#             dataset, batch_size=batch_size, 
#             shuffle=True, num_workers=4, pin_memory=True
#         )
    
#     return loader

# # === Example Usage ===
# """
# from training.dataset_loader import ECGDataset
# from training.balanced_sampler import create_balanced_dataloader

# # Create dataset
# train_dataset = ECGDataset(train_files, y_train, augment=True)

# # Create balanced dataloader
# train_loader = create_balanced_dataloader(
#     train_dataset, y_train, 
#     batch_size=32, 
#     use_sampler=True, 
#     oversample_factor=3.0  # Oversample rare classes 3x
# )
# """