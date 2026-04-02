#UPDATE 130126
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import copy
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# --- Mixup (giữ nguyên) ---
def mixup_data(x, y, alpha=0.4, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# --- Class Weights (giữ nguyên) ---
def compute_class_weights_cb(y_train, beta=0.9999, device="cuda"):
    num_samples_per_class = y_train.sum(axis=0)
    effective_num = 1.0 - np.power(beta, num_samples_per_class)
    weights = (1.0 - beta) / (effective_num + 1e-8)
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32).to(device)

# --- Focal Loss (giữ nguyên) ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        focal_factor = (1 - p_t) ** self.gamma
        loss = focal_factor * bce_loss
        if self.alpha is not None:
            loss = self.alpha * loss
        
        if self.reduction == "mean": return loss.mean()
        elif self.reduction == "sum": return loss.sum()
        return loss


# ========== COMPUTE METRICS FUNCTION ==========
def compute_metrics(model, dataloader, device, is_hybrid=False):
    """
    Compute F1, Precision, Recall, Accuracy for validation set.
    
    Returns:
        dict với keys: 'micro_f1', 'macro_f1', 'precision', 'recall', 
                       'accuracy', 'per_class_f1', 'per_class_precision', 'per_class_recall'
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in dataloader:
            if is_hybrid:
                signals, hand_features, labels = batch_data
                signals = signals.to(device)
                hand_features = hand_features.to(device)
                outputs = model(signals, hand_features)
            else:
                signals, labels = batch_data
                signals = signals.to(device)
                outputs = model(signals)
            
            # Predictions
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Concatenate
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)
    
    # Compute metrics
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    micro_recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    
    # Accuracy: exact match ratio (multilabel)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'precision': micro_precision,
        'recall': micro_recall,
        'accuracy': accuracy,
        'per_class_f1': per_class_f1,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall
    }


# ========== MAIN TRAINING WITH HISTORY TRACKING ==========
def train_model(model, train_loader, val_loader, y_train, 
                num_epochs=30, patience=7, lr=1e-3, device="cuda",
                loss_type="cb_focal", focal_gamma=2.0, 
                use_mixup=True, mixup_alpha=0.4, 
                use_swa=False, swa_start_epoch=15,
                is_hybrid=False,
                target_names=None):
    """
    Training với metrics tracking mỗi epoch + trả về history dict.
    
    Returns:
        model: Trained model
        history: Dict chứa {'train_loss': [], 'val_loss': [], 'train_f1': [], ...}
    """
    
    model = model.to(device)
    
    # Setup Loss
    if loss_type == "cb_focal":
        class_weights = compute_class_weights_cb(y_train, device=device)
        criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # SWA Setup
    if use_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=lr * 0.1)
    
    # ========== HISTORY TRACKING ==========
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_f1': [],
        'val_f1': [],
        'train_macro_f1': [],
        'val_macro_f1': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float("inf")
    best_micro_f1 = 0.0
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    model_type = "HybridECGModel" if is_hybrid else type(model).__name__
    print(f"🚀 Training Config: Mixup={use_mixup}, SWA={use_swa}, Model={model_type}")
    print("="*80)

    # ========== TRAINING LOOP ==========
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
        for batch_data in loop:
            # Unpack
            if is_hybrid:
                signals, hand_features, labels = batch_data
                signals = signals.to(device)
                hand_features = hand_features.to(device)
                labels = labels.to(device)
            else:
                signals, labels = batch_data
                signals = signals.to(device)
                labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixup
            if use_mixup:
                signals_mixed, targets_a, targets_b, lam = mixup_data(
                    signals, labels, alpha=mixup_alpha, device=device
                )
                if is_hybrid:
                    outputs = model(signals_mixed, hand_features)
                else:
                    outputs = model(signals_mixed)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                if is_hybrid:
                    outputs = model(signals, hand_features)
                else:
                    outputs = model(signals)
                loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=total_loss/len(train_loader))

        # Update scheduler
        if use_swa and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # ========== VALIDATION ==========
        val_model = swa_model if (use_swa and epoch >= swa_start_epoch) else model
        val_model.eval()
        total_val_loss = 0
        
        # Compute validation loss
        with torch.no_grad():
            for batch_data in val_loader:
                if is_hybrid:
                    signals, hand_features, labels = batch_data
                    signals = signals.to(device)
                    hand_features = hand_features.to(device)
                    labels = labels.to(device)
                    outputs = val_model(signals, hand_features)
                else:
                    signals, labels = batch_data
                    signals = signals.to(device)
                    labels = labels.to(device)
                    outputs = val_model(signals)
                
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        # ========== COMPUTE METRICS (Train + Val) ==========
        train_metrics = compute_metrics(val_model, train_loader, device, is_hybrid=is_hybrid)
        val_metrics = compute_metrics(val_model, val_loader, device, is_hybrid=is_hybrid)
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_f1'].append(train_metrics['micro_f1'])
        history['val_f1'].append(val_metrics['micro_f1'])
        history['train_macro_f1'].append(train_metrics['macro_f1'])
        history['val_macro_f1'].append(val_metrics['macro_f1'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  Train F1:   {train_metrics['micro_f1']:.4f} | Val F1:   {val_metrics['micro_f1']:.4f}")
        print(f"  Train Acc:  {train_metrics['accuracy']:.4f} | Val Acc:  {val_metrics['accuracy']:.4f}")
        
        # Print per-class F1 (abbreviated)
        if target_names is not None and len(target_names) == len(val_metrics['per_class_f1']):
            print(f"  Per-class Val F1:")
            for i, (name, f1) in enumerate(zip(target_names, val_metrics['per_class_f1'])):
                short_name = name.split('-')[-1].strip()
                print(f"    {short_name:8s}: {f1:.4f}", end="  ")
                if (i+1) % 3 == 0:
                    print()
            print()
        
        print("="*80)

        # Early Stopping
        improved = False
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            improved = True
        
        if val_metrics['micro_f1'] > best_micro_f1:
            best_micro_f1 = val_metrics['micro_f1']
            improved = True
        
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            if use_swa and epoch >= swa_start_epoch:
                torch.save(swa_model.state_dict(), "temp_best_swa.pth")
            patience_counter = 0
            print(f"✅ Best model updated! (Val Loss: {best_val_loss:.4f}, F1: {best_micro_f1:.4f})")
        else:
            patience_counter += 1
            print(f"⏳ No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print("🛑 Early stopping!")
                break
    
    # Finish Training
    if use_swa:
        print("\n🔥 Updating BatchNorm statistics for SWA model...")
        
        if is_hybrid:
            swa_model.train()
            with torch.no_grad():
                for batch_data in train_loader:
                    signals, hand_features, _ = batch_data
                    signals = signals.to(device)
                    hand_features = hand_features.to(device)
                    _ = swa_model(signals, hand_features)
            print("   ✅ BatchNorm updated for hybrid model")
        else:
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
            print("   ✅ BatchNorm updated for standard model")
        
        return swa_model, history
    
    model.load_state_dict(best_state)
    return model, history


if __name__ == "__main__":
    print("✅ Training module with history tracking ready!")


#UPDATE 100126
# training/train_improved.py (VERSION WITH METRICS)

# import torch
# import torch.nn as nn
# import numpy as np
# from tqdm import tqdm
# import copy
# from torch.optim.swa_utils import AveragedModel, SWALR
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from sklearn.metrics import f1_score, precision_score, recall_score

# # --- Mixup (giữ nguyên) ---
# def mixup_data(x, y, alpha=0.4, device='cuda'):
#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1
#     batch_size = x.size(0)
#     index = torch.randperm(batch_size).to(device)
#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     y_a, y_b = y, y[index]
#     return mixed_x, y_a, y_b, lam

# def mixup_criterion(criterion, pred, y_a, y_b, lam):
#     return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# # --- Class Weights (giữ nguyên) ---
# def compute_class_weights_cb(y_train, beta=0.9999, device="cuda"):
#     num_samples_per_class = y_train.sum(axis=0)
#     effective_num = 1.0 - np.power(beta, num_samples_per_class)
#     weights = (1.0 - beta) / (effective_num + 1e-8)
#     weights = weights / weights.sum() * len(weights)
#     return torch.tensor(weights, dtype=torch.float32).to(device)

# # --- Focal Loss (giữ nguyên) ---
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, logits, targets):
#         bce_loss = nn.functional.binary_cross_entropy_with_logits(
#             logits, targets, reduction="none"
#         )
#         prob = torch.sigmoid(logits)
#         p_t = prob * targets + (1 - prob) * (1 - targets)
#         focal_factor = (1 - p_t) ** self.gamma
#         loss = focal_factor * bce_loss
#         if self.alpha is not None:
#             loss = self.alpha * loss
        
#         if self.reduction == "mean": return loss.mean()
#         elif self.reduction == "sum": return loss.sum()
#         return loss


# # ========== NEW: COMPUTE METRICS FUNCTION ==========
# def compute_metrics(model, dataloader, device, is_hybrid=False):
#     """
#     Compute F1, Precision, Recall for validation set.
    
#     Returns:
#         dict với keys: 'micro_f1', 'macro_f1', 'precision', 'recall', 'per_class_f1'
#     """
#     model.eval()
#     all_preds = []
#     all_labels = []
    
#     with torch.no_grad():
#         for batch_data in dataloader:
#             if is_hybrid:
#                 signals, hand_features, labels = batch_data
#                 signals = signals.to(device)
#                 hand_features = hand_features.to(device)
#                 outputs = model(signals, hand_features)
#             else:
#                 signals, labels = batch_data
#                 signals = signals.to(device)
#                 outputs = model(signals)
            
#             # Predictions
#             probs = torch.sigmoid(outputs)
#             preds = (probs > 0.5).float()
            
#             all_preds.append(preds.cpu().numpy())
#             all_labels.append(labels.numpy())
    
#     # Concatenate
#     y_pred = np.vstack(all_preds)
#     y_true = np.vstack(all_labels)
    
#     # Compute metrics
#     micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
#     macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
#     precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
#     recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    
#     # Per-class F1
#     per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
#     return {
#         'micro_f1': micro_f1,
#         'macro_f1': macro_f1,
#         'precision': precision,
#         'recall': recall,
#         'per_class_f1': per_class_f1
#     }


# # ========== MAIN TRAINING WITH METRICS ==========
# def train_model(model, train_loader, val_loader, y_train, 
#                 num_epochs=30, patience=7, lr=1e-3, device="cuda",
#                 loss_type="cb_focal", focal_gamma=2.0, 
#                 use_mixup=True, mixup_alpha=0.4, 
#                 use_swa=False, swa_start_epoch=15,
#                 is_hybrid=False,
#                 target_names=None):  # ← THÊM target_names
#     """
#     Training với metrics tracking mỗi epoch.
#     """
    
#     model = model.to(device)
    
#     # Setup Loss
#     if loss_type == "cb_focal":
#         class_weights = compute_class_weights_cb(y_train, device=device)
#         criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
#     else:
#         criterion = nn.BCEWithLogitsLoss()

#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
#     scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
#     # SWA Setup
#     if use_swa:
#         swa_model = AveragedModel(model)
#         swa_scheduler = SWALR(optimizer, swa_lr=lr * 0.1)
    
#     best_val_loss = float("inf")
#     best_micro_f1 = 0.0  # ← Track best F1
#     best_state = copy.deepcopy(model.state_dict())
#     patience_counter = 0

#     model_type = "HybridECGModel" if is_hybrid else type(model).__name__
#     print(f"🚀 Training Config: Mixup={use_mixup}, SWA={use_swa}, Model={model_type}")
#     print("="*80)

#     # ========== TRAINING LOOP ==========
#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
#         for batch_data in loop:
#             # Unpack
#             if is_hybrid:
#                 signals, hand_features, labels = batch_data
#                 signals = signals.to(device)
#                 hand_features = hand_features.to(device)
#                 labels = labels.to(device)
#             else:
#                 signals, labels = batch_data
#                 signals = signals.to(device)
#                 labels = labels.to(device)
            
#             optimizer.zero_grad()
            
#             # Mixup
#             if use_mixup:
#                 signals_mixed, targets_a, targets_b, lam = mixup_data(
#                     signals, labels, alpha=mixup_alpha, device=device
#                 )
#                 if is_hybrid:
#                     outputs = model(signals_mixed, hand_features)
#                 else:
#                     outputs = model(signals_mixed)
#                 loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
#             else:
#                 if is_hybrid:
#                     outputs = model(signals, hand_features)
#                 else:
#                     outputs = model(signals)
#                 loss = criterion(outputs, labels)
            
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             total_loss += loss.item()
#             loop.set_postfix(loss=total_loss/len(train_loader))

#         # Update scheduler
#         if use_swa and epoch >= swa_start_epoch:
#             swa_model.update_parameters(model)
#             swa_scheduler.step()
#         else:
#             scheduler.step()

#         # ========== VALIDATION WITH METRICS ==========
#         val_model = swa_model if (use_swa and epoch >= swa_start_epoch) else model
#         val_model.eval()
#         total_val_loss = 0
        
#         # Compute validation loss
#         with torch.no_grad():
#             for batch_data in val_loader:
#                 if is_hybrid:
#                     signals, hand_features, labels = batch_data
#                     signals = signals.to(device)
#                     hand_features = hand_features.to(device)
#                     labels = labels.to(device)
#                     outputs = val_model(signals, hand_features)
#                 else:
#                     signals, labels = batch_data
#                     signals = signals.to(device)
#                     labels = labels.to(device)
#                     outputs = val_model(signals)
                
#                 loss = criterion(outputs, labels)
#                 total_val_loss += loss.item()

#         avg_train_loss = total_loss / len(train_loader)
#         avg_val_loss = total_val_loss / len(val_loader)
        
#         # ========== COMPUTE METRICS ==========
#         metrics = compute_metrics(val_model, val_loader, device, is_hybrid=is_hybrid)
        
#         # Print epoch results
#         print(f"\nEpoch {epoch+1}/{num_epochs}:")
#         print(f"  Train Loss: {avg_train_loss:.4f}")
#         print(f"  Val Loss:   {avg_val_loss:.4f}")
#         print(f"  Micro F1:   {metrics['micro_f1']:.4f}")
#         print(f"  Macro F1:   {metrics['macro_f1']:.4f}")
#         print(f"  Precision:  {metrics['precision']:.4f}")
#         print(f"  Recall:     {metrics['recall']:.4f}")
        
#         # Print per-class F1 (abbreviated)
#         if target_names is not None and len(target_names) == len(metrics['per_class_f1']):
#             print(f"  Per-class F1:")
#             for i, (name, f1) in enumerate(zip(target_names, metrics['per_class_f1'])):
#                 short_name = name.split('-')[-1].strip()  # Get short name
#                 print(f"    {short_name:8s}: {f1:.4f}", end="  ")
#                 if (i+1) % 3 == 0:  # Newline every 3 classes
#                     print()
#             print()  # Final newline
        
#         print("="*80)

#         # Early Stopping (based on val loss OR F1)
#         improved = False
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             improved = True
        
#         if metrics['micro_f1'] > best_micro_f1:
#             best_micro_f1 = metrics['micro_f1']
#             improved = True
        
#         if improved:
#             best_state = copy.deepcopy(model.state_dict())
#             if use_swa and epoch >= swa_start_epoch:
#                 torch.save(swa_model.state_dict(), "temp_best_swa.pth")
#             patience_counter = 0
#             print(f"✅ Best model updated! (Val Loss: {best_val_loss:.4f}, Micro F1: {best_micro_f1:.4f})")
#         else:
#             patience_counter += 1
#             print(f"⏳ No improvement ({patience_counter}/{patience})")
#             if patience_counter >= patience:
#                 print("🛑 Early stopping!")
#                 break
    
#     # Finish Training
#     if use_swa:
#         print("\n📥 Updating BatchNorm statistics for SWA model...")
        
#         if is_hybrid:
#             swa_model.train()
#             with torch.no_grad():
#                 for batch_data in train_loader:
#                     signals, hand_features, _ = batch_data
#                     signals = signals.to(device)
#                     hand_features = hand_features.to(device)
#                     _ = swa_model(signals, hand_features)
#             print("   ✅ BatchNorm updated for hybrid model")
#         else:
#             torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
#             print("   ✅ BatchNorm updated for standard model")
        
#         return swa_model
    
#     model.load_state_dict(best_state)
#     return model


# if __name__ == "__main__":
#     print("✅ Training module with metrics ready!")

#UPDATE 070126
# training/train_improved.py (CẬP NHẬT)
# import torch
# import torch.nn as nn
# import numpy as np
# from tqdm import tqdm
# import copy
# from torch.optim.swa_utils import AveragedModel, SWALR
# from torch.optim.lr_scheduler import CosineAnnealingLR

# # --- Mixup Utilities (GIỮ NGUYÊN) ---
# def mixup_data(x, y, alpha=0.4, device='cuda'):
#     '''Returns mixed inputs, pairs of targets, and lambda'''
#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1

#     batch_size = x.size(0)
#     index = torch.randperm(batch_size).to(device)

#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     y_a, y_b = y, y[index]
#     return mixed_x, y_a, y_b, lam

# def mixup_criterion(criterion, pred, y_a, y_b, lam):
#     return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# # --- Class Weights (GIỮ NGUYÊN) ---
# def compute_class_weights_cb(y_train, beta=0.9999, device="cuda"):
#     num_samples_per_class = y_train.sum(axis=0)
#     effective_num = 1.0 - np.power(beta, num_samples_per_class)
#     weights = (1.0 - beta) / (effective_num + 1e-8)
#     weights = weights / weights.sum() * len(weights)
#     return torch.tensor(weights, dtype=torch.float32).to(device)

# # --- Loss Functions (GIỮ NGUYÊN) ---
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, logits, targets):
#         bce_loss = nn.functional.binary_cross_entropy_with_logits(
#             logits, targets, reduction="none"
#         )
#         prob = torch.sigmoid(logits)
#         p_t = prob * targets + (1 - prob) * (1 - targets)
#         focal_factor = (1 - p_t) ** self.gamma
#         loss = focal_factor * bce_loss
#         if self.alpha is not None:
#             loss = self.alpha * loss
        
#         if self.reduction == "mean": return loss.mean()
#         elif self.reduction == "sum": return loss.sum()
#         return loss


# # --- Main Training Function (CẬP NHẬT - HYBRID SUPPORT) ---
# def train_model(model, train_loader, val_loader, y_train, 
#                 num_epochs=30, patience=7, lr=1e-3, device="cuda",
#                 loss_type="cb_focal", focal_gamma=2.0, 
#                 use_mixup=True, mixup_alpha=0.4, 
#                 use_swa=False, swa_start_epoch=15,
#                 is_hybrid=False):  # ← THÊM PARAMETER NÀY
#     """
#     Training function với support cho cả standard và hybrid models.
    
#     Args:
#         is_hybrid: True nếu model là HybridECGModel (cần 2 inputs)
#                    False nếu model là ResNet18_LSTM_Attn (1 input)
#     """
    
#     model = model.to(device)
    
#     # Setup Loss (GIỮ NGUYÊN)
#     if loss_type == "cb_focal":
#         class_weights = compute_class_weights_cb(y_train, device=device)
#         criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
#     else:
#         criterion = nn.BCEWithLogitsLoss()

#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
#     scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
#     # SWA Setup (GIỮ NGUYÊN)
#     if use_swa:
#         swa_model = AveragedModel(model)
#         swa_scheduler = SWALR(optimizer, swa_lr=lr * 0.1)
    
#     best_val_loss = float("inf")
#     best_state = copy.deepcopy(model.state_dict())
#     patience_counter = 0

#     model_type = "HybridECGModel" if is_hybrid else type(model).__name__
#     print(f"🚀 Training Config: Mixup={use_mixup}, SWA={use_swa}, Model={model_type}")

#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
#         # ========== TRAINING LOOP - HYBRID SUPPORT ==========
#         for batch_data in loop:  # ← THAY ĐỔI: batch_data thay vì signals, labels
            
#             # Unpack data based on model type
#             if is_hybrid:
#                 signals, hand_features, labels = batch_data
#                 signals = signals.to(device)
#                 hand_features = hand_features.to(device)
#                 labels = labels.to(device)
#             else:
#                 signals, labels = batch_data
#                 signals = signals.to(device)
#                 labels = labels.to(device)
            
#             optimizer.zero_grad()
            
#             # === MIXUP LOGIC ===
#             if use_mixup:
#                 # Mixup chỉ áp dụng cho signals, KHÔNG cho hand_features
#                 signals_mixed, targets_a, targets_b, lam = mixup_data(
#                     signals, labels, alpha=mixup_alpha, device=device
#                 )
                
#                 if is_hybrid:
#                     # Hand features KHÔNG được mixup (giữ nguyên)
#                     outputs = model(signals_mixed, hand_features)
#                 else:
#                     outputs = model(signals_mixed)
                
#                 loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
#             else:
#                 # No mixup
#                 if is_hybrid:
#                     outputs = model(signals, hand_features)
#                 else:
#                     outputs = model(signals)
                
#                 loss = criterion(outputs, labels)
            
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             total_loss += loss.item()
#             loop.set_postfix(loss=total_loss/len(train_loader))

#         # Update SWA (GIỮ NGUYÊN)
#         if use_swa and epoch >= swa_start_epoch:
#             swa_model.update_parameters(model)
#             swa_scheduler.step()
#         else:
#             scheduler.step()

#         # ========== VALIDATION - HYBRID SUPPORT ==========
#         val_model = swa_model if (use_swa and epoch >= swa_start_epoch) else model
#         val_model.eval()
#         total_val_loss = 0
        
#         with torch.no_grad():
#             for batch_data in val_loader:
                
#                 if is_hybrid:
#                     signals, hand_features, labels = batch_data
#                     signals = signals.to(device)
#                     hand_features = hand_features.to(device)
#                     labels = labels.to(device)
#                     outputs = val_model(signals, hand_features)
#                 else:
#                     signals, labels = batch_data
#                     signals = signals.to(device)
#                     labels = labels.to(device)
#                     outputs = val_model(signals)
                
#                 loss = criterion(outputs, labels)
#                 total_val_loss += loss.item()

#         avg_train_loss = total_loss / len(train_loader)
#         avg_val_loss = total_val_loss / len(val_loader)
        
#         print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

#         # Early Stopping (GIỮ NGUYÊN)
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             best_state = copy.deepcopy(model.state_dict())
#             if use_swa and epoch >= swa_start_epoch:
#                 torch.save(swa_model.state_dict(), "temp_best_swa.pth")
#             patience_counter = 0
#         else:
#             patience_counter += 1
#             if patience_counter >= patience:
#                 print("Early stopping!")
#                 break
    
#     # Finish Training (GIỮ NGUYÊN)
#     if use_swa:
#         print("📥 Updating BatchNorm statistics for SWA model...")
        
#         # Custom update_bn for hybrid models
#         if is_hybrid:
#             # Manual update for hybrid model
#             swa_model.train()
#             with torch.no_grad():
#                 for batch_data in train_loader:
#                     signals, hand_features, _ = batch_data  # Unpack 3 items
#                     signals = signals.to(device)
#                     hand_features = hand_features.to(device)
#                     _ = swa_model(signals, hand_features)  # Forward with 2 inputs
#             print("   BatchNorm updated for hybrid model")
#         else:
#             # Standard update for non-hybrid models
#             torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
#             print("   BatchNorm updated for standard model")
        
#         return swa_model
    
#     model.load_state_dict(best_state)
#     return model


# training/train_improved.py
# import torch
# import torch.nn as nn
# import numpy as np
# from tqdm import tqdm
# import copy
# from torch.optim.swa_utils import AveragedModel, SWALR
# from torch.optim.lr_scheduler import CosineAnnealingLR

# # --- Mixup Utilities ---
# def mixup_data(x, y, alpha=0.4, device='cuda'):
#     '''Returns mixed inputs, pairs of targets, and lambda'''
#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1

#     batch_size = x.size(0)
#     index = torch.randperm(batch_size).to(device)

#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     y_a, y_b = y, y[index]
#     return mixed_x, y_a, y_b, lam

# def mixup_criterion(criterion, pred, y_a, y_b, lam):
#     return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# # --- Class Weights ---
# def compute_class_weights_cb(y_train, beta=0.9999, device="cuda"):
#     num_samples_per_class = y_train.sum(axis=0)
#     effective_num = 1.0 - np.power(beta, num_samples_per_class)
#     weights = (1.0 - beta) / (effective_num + 1e-8)
#     weights = weights / weights.sum() * len(weights)
#     return torch.tensor(weights, dtype=torch.float32).to(device)

# # --- Loss Functions ---
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, logits, targets):
#         bce_loss = nn.functional.binary_cross_entropy_with_logits(
#             logits, targets, reduction="none"
#         )
#         prob = torch.sigmoid(logits)
#         p_t = prob * targets + (1 - prob) * (1 - targets)
#         focal_factor = (1 - p_t) ** self.gamma
#         loss = focal_factor * bce_loss
#         if self.alpha is not None:
#             loss = self.alpha * loss
        
#         if self.reduction == "mean": return loss.mean()
#         elif self.reduction == "sum": return loss.sum()
#         return loss

# # --- Main Training Function ---
# def train_model(model, train_loader, val_loader, y_train, 
#                 num_epochs=30, patience=7, lr=1e-3, device="cuda",
#                 loss_type="cb_focal", focal_gamma=2.0, 
#                 use_mixup=True, mixup_alpha=0.4, 
#                 use_swa=False, swa_start_epoch=15):
    
#     model = model.to(device)
    
#     # Setup Loss
#     if loss_type == "cb_focal":
#         class_weights = compute_class_weights_cb(y_train, device=device)
#         criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
#     else:
#         # Fallback to BCE
#         criterion = nn.BCEWithLogitsLoss()

#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
#     scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
#     # SWA Setup
#     if use_swa:
#         swa_model = AveragedModel(model)
#         swa_scheduler = SWALR(optimizer, swa_lr=lr * 0.1)
    
#     best_val_loss = float("inf")
#     best_state = copy.deepcopy(model.state_dict())
#     patience_counter = 0

#     print(f"🚀 Training Config: Mixup={use_mixup}, SWA={use_swa}, Model={type(model).__name__}")

#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
#         for signals, labels in loop:
#             signals, labels = signals.to(device), labels.to(device)
#             optimizer.zero_grad()
            
#             # === MIXUP LOGIC ===
#             if use_mixup:
#                 signals, targets_a, targets_b, lam = mixup_data(signals, labels, alpha=mixup_alpha, device=device)
#                 outputs = model(signals)
#                 loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
#             else:
#                 outputs = model(signals)
#                 loss = criterion(outputs, labels)
            
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             total_loss += loss.item()
#             loop.set_postfix(loss=total_loss/len(train_loader))

#         # Update SWA
#         if use_swa and epoch >= swa_start_epoch:
#             swa_model.update_parameters(model)
#             swa_scheduler.step()
#         else:
#             scheduler.step()

#         # Validation
#         val_model = swa_model if (use_swa and epoch >= swa_start_epoch) else model
#         val_model.eval()
#         total_val_loss = 0
        
#         with torch.no_grad():
#             for signals, labels in val_loader:
#                 signals, labels = signals.to(device), labels.to(device)
#                 outputs = val_model(signals)
#                 loss = criterion(outputs, labels)
#                 total_val_loss += loss.item()

#         avg_train_loss = total_loss / len(train_loader)
#         avg_val_loss = total_val_loss / len(val_loader)
        
#         print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

#         # Early Stopping check (on standard model to be safe)
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             best_state = copy.deepcopy(model.state_dict()) # Save standard weights
#             if use_swa and epoch >= swa_start_epoch:
#                  # Also save SWA weights if it's currently better
#                  torch.save(swa_model.state_dict(), "temp_best_swa.pth")
#             patience_counter = 0
#         else:
#             patience_counter += 1
#             if patience_counter >= patience:
#                 print("Early stopping!")
#                 break
    
#     # Finish Training
#     if use_swa:
#         print("📥 Updating BatchNorm statistics for SWA model...")
#         torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
#         return swa_model # Return the SWA model
    
#     model.load_state_dict(best_state)
#     return model