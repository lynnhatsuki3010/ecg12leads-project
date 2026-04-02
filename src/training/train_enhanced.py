# training/train_enhanced.py
"""
TRAINING SCRIPT CẢI TIẾN
- Focal Loss cho class imbalance
- Class-specific augmentation
- Per-class threshold tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, classification_report

# ===== FOCAL LOSS =====
class FocalLoss(nn.Module):
    """
    Focal Loss cho multi-label classification
    Giúp model tập trung vào các class khó (rare classes)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: logits (batch, num_classes)
            targets: binary labels (batch, num_classes)
        """
        # Binary cross entropy
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Focal term
        pt = torch.exp(-BCE_loss)  # Probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        return focal_loss.mean()


class WeightedFocalLoss(nn.Module):
    """
    Focal Loss với class weights
    Cho các class cực kỳ hiếm (như LVH chỉ 52 samples)
    """
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        # Apply class weights
        if self.class_weights is not None:
            weights = self.class_weights.unsqueeze(0).to(inputs.device)
            focal_loss = focal_loss * weights
        
        return focal_loss.mean()


# ===== AUGMENTATION STRATEGIES =====
def lvh_preserving_augmentation(signal):
    """
    Augmentation cho LVH - GIỮ NGUYÊN biên độ
    Chỉ thay đổi: timing, baseline wander, noise
    """
    # 1. Baseline wander
    baseline = np.random.randn(signal.shape[1]) * 0.005
    baseline = np.cumsum(baseline) * 0.0005
    signal = signal + baseline
    
    # 2. Time shift (nhẹ)
    shift = np.random.randint(-50, 50)
    signal = np.roll(signal, shift, axis=1)
    
    # 3. Gaussian noise (nhẹ)
    noise = np.random.normal(0, 0.005, signal.shape)
    signal = signal + noise
    
    # 4. KHÔNG DÙNG scaling hoặc các phép biến đổi ảnh hưởng biên độ!
    
    return signal


def rhythm_focused_augmentation(signal):
    """
    Augmentation cho các bệnh về nhịp (AFIB, AF, ST, SB, SVT)
    Có thể thay đổi biên độ vì không ảnh hưởng chẩn đoán
    """
    # 1. Jittering
    noise = np.random.normal(0, 0.01, signal.shape)
    signal = signal + noise
    
    # 2. Scaling
    scale = np.random.uniform(0.9, 1.1)
    signal = signal * scale
    
    # 3. Time shift
    shift = np.random.randint(-100, 100)
    signal = np.roll(signal, shift, axis=1)
    
    # 4. Baseline wander
    baseline = np.random.randn(signal.shape[1]) * 0.01
    baseline = np.cumsum(baseline) * 0.001
    signal = signal + baseline
    
    return signal


# ===== THRESHOLD TUNING =====
def find_optimal_thresholds(model, val_loader, y_val, device='cuda', 
                           search_range=(0.1, 0.9), num_steps=50):
    """
    Tìm threshold tối ưu cho từng class riêng biệt
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        y_val: True labels (n_samples, n_classes)
        search_range: (min, max) threshold range
        num_steps: Number of thresholds to try
        
    Returns:
        best_thresholds: Array of shape (n_classes,)
    """
    model.eval()
    
    # Get predictions
    all_probs = []
    with torch.no_grad():
        for signals, _ in val_loader:
            signals = signals.to(device)
            outputs = model(signals)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.append(probs)
    
    y_pred_probs = np.vstack(all_probs)
    
    # Tune threshold for each class
    n_classes = y_val.shape[1]
    best_thresholds = np.zeros(n_classes)
    
    print("\n" + "="*60)
    print("🔍 THRESHOLD TUNING - Tìm ngưỡng tối ưu cho từng class")
    print("="*60)
    
    for i in range(n_classes):
        best_f1 = 0
        best_thresh = 0.5
        
        # Grid search
        thresholds = np.linspace(*search_range, num_steps)
        
        for thresh in thresholds:
            y_pred = (y_pred_probs[:, i] >= thresh).astype(int)
            f1 = f1_score(y_val[:, i], y_pred, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        best_thresholds[i] = best_thresh
        
        # Count positives at this threshold
        n_pos = (y_pred_probs[:, i] >= best_thresh).sum()
        n_true_pos = y_val[:, i].sum()
        
        print(f"Class {i}: thresh={best_thresh:.3f}, F1={best_f1:.3f}, "
              f"pred_pos={n_pos}, true_pos={n_true_pos}")
    
    print("="*60 + "\n")
    
    return best_thresholds


# ===== TRAINING LOOP WITH ENHANCEMENTS =====
def train_enhanced(model, train_loader, val_loader, y_val, 
                  num_epochs=50, lr=0.001, device='cuda',
                  use_focal_loss=True, class_weights=None,
                  patience=10):
    """
    Training loop với các cải tiến:
    - Focal Loss
    - Early stopping
    - Per-class threshold tuning
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Loss function
    if use_focal_loss:
        if class_weights is not None:
            criterion = WeightedFocalLoss(
                alpha=0.25, gamma=2.0, 
                class_weights=torch.tensor(class_weights, dtype=torch.float32)
            )
            print("✅ Sử dụng Weighted Focal Loss")
        else:
            criterion = FocalLoss(alpha=0.25, gamma=2.0)
            print("✅ Sử dụng Focal Loss")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("✅ Sử dụng BCE Loss")
    
    best_f1 = 0
    best_thresholds = None
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print(f"🚀 BẮT ĐẦU TRAINING")
    print(f"   Model: {model.__class__.__name__}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning rate: {lr}")
    print(f"   Device: {device}")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        # ===== TRAINING =====
        model.train()
        train_loss = 0
        
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(signals)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # ===== VALIDATION =====
        model.eval()
        val_probs = []
        
        with torch.no_grad():
            for signals, _ in val_loader:
                signals = signals.to(device)
                outputs = model(signals)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                val_probs.append(probs)
        
        val_probs = np.vstack(val_probs)
        
        # Evaluate with default threshold 0.5
        val_preds = (val_probs >= 0.5).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import f1_score
        f1_micro = f1_score(y_val, val_preds, average='micro', zero_division=0)
        f1_macro = f1_score(y_val, val_preds, average='macro', zero_division=0)
        
        # Learning rate scheduling
        scheduler.step(f1_micro)
        
        # Print progress
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
              f"Loss: {avg_train_loss:.4f} | "
              f"Val F1 (micro): {f1_micro:.4f} | "
              f"Val F1 (macro): {f1_macro:.4f}")
        
        # ===== EARLY STOPPING & CHECKPOINTING =====
        if f1_micro > best_f1:
            best_f1 = f1_micro
            patience_counter = 0
            
            # Tune thresholds on validation set
            best_thresholds = find_optimal_thresholds(
                model, val_loader, y_val, device=device
            )
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'thresholds': best_thresholds
            }, 'best_model_enhanced.pth')
            
            print(f"✅ Lưu model tốt nhất (F1 = {best_f1:.4f})\n")
        else:
            patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping tại epoch {epoch+1}")
                print(f"   Best F1: {best_f1:.4f}")
                break
    
    print(f"\n{'='*60}")
    print(f"✅ HOÀN TẤT TRAINING")
    print(f"   Best F1 (micro): {best_f1:.4f}")
    print(f"   Model saved: best_model_enhanced.pth")
    print(f"{'='*60}\n")
    
    return best_thresholds


# ===== CALCULATE CLASS WEIGHTS =====
def calculate_class_weights(y_train, method='inverse'):
    """
    Tính class weights cho WeightedFocalLoss
    
    Args:
        y_train: Binary labels (n_samples, n_classes)
        method: 'inverse' hoặc 'sqrt_inverse'
        
    Returns:
        weights: Array of shape (n_classes,)
    """
    class_counts = y_train.sum(axis=0)
    total_samples = len(y_train)
    
    if method == 'inverse':
        weights = total_samples / (class_counts + 1e-6)
    elif method == 'sqrt_inverse':
        weights = np.sqrt(total_samples / (class_counts + 1e-6))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize
    weights = weights / weights.sum() * len(weights)
    
    print("\n📊 Class Weights:")
    for i, (count, weight) in enumerate(zip(class_counts, weights)):
        print(f"   Class {i}: samples={count:.0f}, weight={weight:.3f}")
    
    return weights


# ===== EXAMPLE USAGE =====
if __name__ == "__main__":
    """
    Ví dụ sử dụng:
    
    1. Xử lý lại data với Constant Scaling
    2. Tính class weights
    3. Train với Focal Loss
    4. Tune thresholds
    5. Evaluate
    """
    
    print("""
    🎯 HƯỚNG DẪN SỬ DỤNG:
    
    # 1. Import
    from train_enhanced import (
        train_enhanced, 
        calculate_class_weights,
        find_optimal_thresholds
    )
    
    # 2. Tính class weights
    class_weights = calculate_class_weights(y_train, method='inverse')
    
    # 3. Training
    best_thresholds = train_enhanced(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        y_val=y_val,
        num_epochs=50,
        lr=0.001,
        use_focal_loss=True,
        class_weights=class_weights,
        patience=10
    )
    
    # 4. Evaluate với thresholds tối ưu
    test_probs = model.predict(test_loader)
    test_preds = (test_probs >= best_thresholds).astype(int)
    
    print(classification_report(y_test, test_preds))
    """)