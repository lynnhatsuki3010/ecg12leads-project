import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

def find_best_thresholds(model, val_loader, device, num_classes, 
                          search_grid=None, metric="f1", 
                          min_precision=None, min_recall=None,
                          verbose=True):
    model.eval()
    preds = []
    trues = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            out = model(x)
            if isinstance(out, tuple): out = out[0]
            probs = torch.sigmoid(out).cpu().numpy()
            preds.append(probs)
            trues.append(y.numpy())
    
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    
    if search_grid is None:
        search_grid = np.arange(0.05, 0.96, 0.05)
    
    best_thresholds = np.zeros(num_classes)
    
    if verbose:
        print("\n🎯 Finding optimal thresholds per class:")
        print("-" * 95)
        print(f"{'Class':<8} {'Samples':<10} {'Pos%':<8} {'Threshold':<12} "
              f"{'Precision':<12} {'Recall':<12} {'F1':<10}")
        print("-" * 95)
    
    for c in range(num_classes):
        y_true = trues[:, c]
        y_prob = preds[:, c]
        
        n_pos = y_true.sum()
        pos_rate = 100 * n_pos / len(y_true)
        
        best_score = -1
        best_t = 0.5
        best_stats = {"precision": 0, "recall": 0, "f1": 0}
        
        for t in search_grid:
            y_pred = (y_prob >= t).astype(int)
            
            # Chỉ tính metrics nếu có dự đoán dương (tránh warning)
            if y_pred.sum() == 0 and metric != "specificity":
                continue

            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Constraints check
            if min_precision and p < min_precision: continue
            if min_recall and r < min_recall: continue
            
            score = f1 # Default metric
            if metric == "precision": score = p
            elif metric == "recall": score = r
            
            if score > best_score:
                best_score = score
                best_t = t
                best_stats = {"precision": p, "recall": r, "f1": f1}
        
        # Fallback: Nếu không tìm được threshold nào thỏa mãn constraint, giữ 0.5 hoặc ngưỡng thấp nhất
        if best_score == -1:
             best_t = 0.1 # Fallback an toàn cho lớp khó
             
        best_thresholds[c] = best_t
        
        if verbose:
            print(f"{c:<8} {int(n_pos):<10} {pos_rate:<8.1f} {best_t:<12.3f} "
                  f"{best_stats['precision']:<12.3f} {best_stats['recall']:<12.3f} "
                  f"{best_stats['f1']:<10.3f}")
    
    return best_thresholds

def adaptive_threshold_search(model, val_loader, device, num_classes, verbose=True):
    """
    Tìm kiếm ngưỡng thích nghi: 
    - Lớp hiếm (<5%): Quét ngưỡng thấp (0.05 - 0.6)
    - Lớp vừa (5-15%): Quét ngưỡng trung bình (0.1 - 0.8)
    - Lớp phổ biến (>15%): Quét ngưỡng cao (0.3 - 0.9)
    """
    model.eval()
    preds = []
    trues = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            out = model(x)
            if isinstance(out, tuple): out = out[0]
            probs = torch.sigmoid(out).cpu().numpy()
            preds.append(probs)
            trues.append(y.numpy())
            
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    
    class_freqs = trues.sum(axis=0) / len(trues)
    best_thresholds = np.zeros(num_classes)
    
    if verbose:
        print("\n🎯 Adaptive Threshold Search (Chiến lược cho lớp hiếm):")
        print("-" * 105)
        print(f"{'Class':<8} {'Freq(%)':<10} {'Strategy':<15} {'Range':<15} "
              f"{'Best T':<10} {'Prec':<10} {'Recall':<10} {'F1':<10}")
        print("-" * 105)
        
    for c in range(num_classes):
        freq = class_freqs[c]
        y_true = trues[:, c]
        y_prob = preds[:, c]
        
        # Chiến lược chọn dải search
        if freq < 0.05:
            strategy = "Very Rare"
            grid = np.arange(0.05, 0.6, 0.02) # Quét rất mịn ở vùng thấp
        elif freq < 0.15:
            strategy = "Rare"
            grid = np.arange(0.1, 0.8, 0.05)
        else:
            strategy = "Common"
            grid = np.arange(0.2, 0.95, 0.05)
            
        best_f1 = -1
        best_t = 0.5
        best_stats = (0,0,0)
        
        for t in grid:
            y_pred = (y_prob >= t).astype(int)
            if y_pred.sum() == 0: continue
            
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Ưu tiên F1, nhưng với lớp rất hiếm, ưu tiên Recall một chút
            score = f1
            if strategy == "Very Rare":
                score = f1 * 0.4 + r * 0.6 # Bias nhẹ về Recall
            
            if score > best_f1:
                best_f1 = score
                best_t = t
                best_stats = (p, r, f1)
                
        best_thresholds[c] = best_t
        p, r, f1 = best_stats
        
        if verbose:
            print(f"{c:<8} {freq*100:<10.1f} {strategy:<15} {f'{grid[0]}-{grid[-1]}':<15} "
                  f"{best_t:<10.2f} {p:<10.3f} {r:<10.3f} {f1:<10.3f}")
            
    return best_thresholds

def find_balanced_thresholds(model, val_loader, device, num_classes):
    # Wrapper giữ lại để tương thích ngược
    return adaptive_threshold_search(model, val_loader, device, num_classes)

def find_thresholds_with_constraints(model, val_loader, device, num_classes, **kwargs):
    # Wrapper giữ lại
    return find_best_thresholds(model, val_loader, device, num_classes, **kwargs)

def find_best_thresholds_fast_with_constraints(probs, labels, **kwargs):
    # Wrapper giữ lại
    return np.array([0.5]*probs.shape[1]) # Placeholder nếu cần



# import numpy as np
# import torch
# from sklearn.metrics import f1_score, precision_score, recall_score

# def find_best_thresholds(model, val_loader, device, num_classes, 
#                           search_grid=None, metric="f1", 
#                           min_precision=None, min_recall=None,
#                           verbose=True):
#     """
#     Enhanced threshold tuning with additional options
    
#     Args:
#         model: Trained model
#         val_loader: Validation dataloader
#         device: cuda/cpu
#         num_classes: Number of classes
#         search_grid: Array of thresholds to try (default: 0.05-0.95 in steps of 0.05)
#         metric: "f1" | "precision" | "recall" | "balanced"
#         min_precision: If set, only consider thresholds with precision >= this
#         min_recall: If set, only consider thresholds with recall >= this
#         verbose: Print per-class stats
    
#     Returns:
#         best_thresholds: (num_classes,) array
#     """
#     model.eval()
#     preds = []
#     trues = []
    
#     with torch.no_grad():
#         for x, y in val_loader:
#             x = x.to(device)
#             out = model(x)
            
#             # Handle tuple output (e.g., logits + attention)
#             if isinstance(out, tuple):
#                 out = out[0]
            
#             probs = torch.sigmoid(out).cpu().numpy()
#             preds.append(probs)
#             trues.append(y.numpy())
    
#     preds = np.vstack(preds)
#     trues = np.vstack(trues)
    
#     # Use finer grid for better threshold search
#     if search_grid is None:
#         search_grid = np.arange(0.05, 0.96, 0.05)  # 0.05, 0.10, ..., 0.95
    
#     best_thresholds = np.zeros(num_classes)
    
#     if verbose:
#         print("\n🎯 Finding optimal thresholds per class:")
#         print("-" * 85)
#         print(f"{'Class':<8} {'Samples':<10} {'Pos%':<8} {'Threshold':<12} "
#               f"{'Precision':<12} {'Recall':<12} {'F1':<10}")
#         print("-" * 85)
    
#     for c in range(num_classes):
#         y_true = trues[:, c]
#         y_prob = preds[:, c]
        
#         n_pos = y_true.sum()
#         n_total = len(y_true)
#         pos_rate = 100 * n_pos / n_total
        
#         best_score = -1
#         best_t = 0.5
#         best_stats = {"precision": 0, "recall": 0, "f1": 0}
        
#         for t in search_grid:
#             y_pred = (y_prob >= t).astype(int)
            
#             # Calculate metrics
#             p = precision_score(y_true, y_pred, zero_division=0)
#             r = recall_score(y_true, y_pred, zero_division=0)
#             f1 = f1_score(y_true, y_pred, zero_division=0)
            
#             # Apply constraints if specified
#             if min_precision is not None and p < min_precision:
#                 continue
#             if min_recall is not None and r < min_recall:
#                 continue
            
#             # Calculate score based on metric
#             if metric == "f1":
#                 score = f1
#             elif metric == "precision":
#                 score = p
#             elif metric == "recall":
#                 score = r
#             elif metric == "balanced":
#                 # Harmonic mean of precision and recall (same as F1)
#                 score = f1
#             else:
#                 score = f1
            
#             if score > best_score:
#                 best_score = score
#                 best_t = t
#                 best_stats = {"precision": p, "recall": r, "f1": f1}
        
#         best_thresholds[c] = best_t
        
#         if verbose:
#             print(f"{c:<8} {int(n_pos):<10} {pos_rate:<8.1f} {best_t:<12.3f} "
#                   f"{best_stats['precision']:<12.3f} {best_stats['recall']:<12.3f} "
#                   f"{best_stats['f1']:<10.3f}")
    
#     if verbose:
#         print("-" * 85)
#         print(f"\n✅ Best thresholds: {best_thresholds}")
    
#     return best_thresholds


# def find_thresholds_with_constraints(model, val_loader, device, num_classes,
#                                       target_precision=0.7, target_recall=0.7,
#                                       search_grid=None, verbose=True):
#     """
#     Find thresholds that satisfy both precision AND recall constraints
#     Useful when you want to balance precision/recall for imbalanced classes
    
#     Args:
#         target_precision: Minimum acceptable precision (e.g., 0.7)
#         target_recall: Minimum acceptable recall (e.g., 0.7)
#     """
#     return find_best_thresholds(
#         model, val_loader, device, num_classes,
#         search_grid=search_grid,
#         metric="f1",  # Among valid thresholds, pick the one with best F1
#         min_precision=target_precision,
#         min_recall=target_recall,
#         verbose=verbose
#     )


# def find_best_thresholds_fast_with_constraints(probs, labels, 
#                                                  min_precision=0.7, min_recall=0.7,
#                                                  verbose=True):
#     """
#     Fast threshold search with P and R constraints
#     Use this for final threshold tuning to achieve target P&R
#     """
#     from sklearn.metrics import precision_score, recall_score, f1_score
    
#     num_classes = probs.shape[1]
#     best_thresholds = []
    
#     if verbose:
#         print(f"\nSearching thresholds (target: P≥{min_precision}, R≥{min_recall}):")
#         print("-" * 85)
#         print(f"{'Class':<8} {'Samples':<10} {'Threshold':<12} {'P':<8} {'R':<8} {'F1':<8} {'OK':<5}")
#         print("-" * 85)
    
#     for c in range(num_classes):
#         y_true = labels[:, c]
#         y_prob = probs[:, c]
        
#         best_f1 = 0
#         best_t = 0.5
#         best_p, best_r = 0, 0
#         found_valid = False
        
#         # Two-phase search: coarse then fine
#         # Phase 1: Coarse search
#         for t in np.arange(0.1, 0.9, 0.05):
#             y_pred = (y_prob >= t).astype(int)
#             if y_pred.sum() == 0:
#                 continue
            
#             p = precision_score(y_true, y_pred, zero_division=0)
#             r = recall_score(y_true, y_pred, zero_division=0)
#             f1 = f1_score(y_true, y_pred, zero_division=0)
            
#             # Check if constraints satisfied
#             if p >= min_precision and r >= min_recall:
#                 if f1 > best_f1:
#                     best_f1 = f1
#                     best_t = t
#                     best_p, best_r = p, r
#                     found_valid = True
        
#         # Phase 2: Fine search around best threshold
#         if found_valid:
#             for t in np.arange(max(0.1, best_t-0.1), min(0.9, best_t+0.1), 0.01):
#                 y_pred = (y_prob >= t).astype(int)
#                 if y_pred.sum() == 0:
#                     continue
                
#                 p = precision_score(y_true, y_pred, zero_division=0)
#                 r = recall_score(y_true, y_pred, zero_division=0)
#                 f1 = f1_score(y_true, y_pred, zero_division=0)
                
#                 if p >= min_precision and r >= min_recall and f1 > best_f1:
#                     best_f1 = f1
#                     best_t = t
#                     best_p, best_r = p, r
        
#         # If no valid threshold, find best F1
#         if not found_valid:
#             for t in np.arange(0.05, 0.95, 0.02):
#                 y_pred = (y_prob >= t).astype(int)
#                 if y_pred.sum() == 0:
#                     continue
                
#                 f1 = f1_score(y_true, y_pred, zero_division=0)
#                 if f1 > best_f1:
#                     best_f1 = f1
#                     best_t = t
#                     best_p = precision_score(y_true, y_pred, zero_division=0)
#                     best_r = recall_score(y_true, y_pred, zero_division=0)
        
#         best_thresholds.append(best_t)
        
#         ok = "✅" if (best_p >= min_precision and best_r >= min_recall) else "⚠️"
#         n_pos = y_true.sum()
        
#         if verbose:
#             print(f"{c:<8} {int(n_pos):<10} {best_t:<12.3f} {best_p:<8.3f} "
#                   f"{best_r:<8.3f} {best_f1:<8.3f} {ok:<5}")
    
#     if verbose:
#         print("-" * 85)
    
#     return np.array(best_thresholds)


# def adaptive_threshold_search(model, val_loader, device, num_classes, 
#                                class_weights=None, verbose=True):
#     """
#     Adaptive threshold search based on class frequency
#     - Rare classes get wider search range and lower default threshold
#     - Common classes get narrower search range
    
#     Args:
#         class_weights: (num_classes,) array of class frequencies
#                        If None, will be computed from val_loader
#     """
#     model.eval()
#     preds = []
#     trues = []
    
#     with torch.no_grad():
#         for x, y in val_loader:
#             x = x.to(device)
#             out = model(x)
#             if isinstance(out, tuple):
#                 out = out[0]
#             probs = torch.sigmoid(out).cpu().numpy()
#             preds.append(probs)
#             trues.append(y.numpy())
    
#     preds = np.vstack(preds)
#     trues = np.vstack(trues)
    
#     # Compute class frequencies if not provided
#     if class_weights is None:
#         class_weights = trues.sum(axis=0) / len(trues)
    
#     best_thresholds = np.zeros(num_classes)
    
#     if verbose:
#         print("\n🎯 Adaptive threshold search:")
#         print("-" * 85)
#         print(f"{'Class':<8} {'Frequency':<12} {'Search Range':<20} "
#               f"{'Best T':<12} {'F1':<10}")
#         print("-" * 85)
    
#     for c in range(num_classes):
#         y_true = trues[:, c]
#         y_prob = preds[:, c]
#         freq = class_weights[c]
        
#         # Adaptive search grid based on class frequency
#         if freq < 0.05:  # Very rare class (< 5%)
#             search_grid = np.arange(0.1, 0.7, 0.05)  # Lower thresholds
#         elif freq < 0.15:  # Rare class (5-15%)
#             search_grid = np.arange(0.2, 0.8, 0.05)
#         else:  # Common class (> 15%)
#             search_grid = np.arange(0.3, 0.9, 0.05)  # Higher thresholds
        
#         best_f1 = -1
#         best_t = 0.5
        
#         for t in search_grid:
#             y_pred = (y_prob >= t).astype(int)
#             f1 = f1_score(y_true, y_pred, zero_division=0)
            
#             if f1 > best_f1:
#                 best_f1 = f1
#                 best_t = t
        
#         best_thresholds[c] = best_t
        
#         if verbose:
#             print(f"{c:<8} {freq:<12.3f} {f'[{search_grid[0]:.2f}-{search_grid[-1]:.2f}]':<20} "
#                   f"{best_t:<12.3f} {best_f1:<10.3f}")
    
#     if verbose:
#         print("-" * 85)
    
#     return best_thresholds

# def find_balanced_thresholds(model, val_loader, device, num_classes,
#                             precision_weight=1.0, recall_weight=1.0):
#     """
#     Find thresholds that balance precision and recall based on custom weights
    
#     Args:
#         precision_weight: Weight for precision (e.g., 1.5 means prefer precision)
#         recall_weight: Weight for recall (e.g., 2.0 means prefer recall)
    
#     F_beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
#     where beta = sqrt(recall_weight / precision_weight)
#     """
#     from sklearn.metrics import precision_score, recall_score, fbeta_score
    
#     model.eval()
#     all_probs = []
#     all_labels = []
    
#     with torch.no_grad():
#         for signals, labels in val_loader:
#             signals = signals.to(device)
#             outputs = model(signals)
#             if isinstance(outputs, tuple):
#                 outputs = outputs[0]
#             probs = torch.sigmoid(outputs).cpu().numpy()
#             all_probs.append(probs)
#             all_labels.append(labels.numpy())
    
#     all_probs = np.vstack(all_probs)
#     all_labels = np.vstack(all_labels)
    
#     # Calculate beta for F-beta score
#     beta = np.sqrt(recall_weight / precision_weight)
    
#     best_thresholds = []
    
#     print(f"\n🎯 Finding balanced thresholds (P_weight={precision_weight}, R_weight={recall_weight}):")
#     print("-" * 90)
#     print(f"{'Class':<8} {'Samples':<10} {'Threshold':<12} {'Precision':<12} {'Recall':<12} "
#           f"{'F{beta:.1f}':<10}")
#     print("-" * 90)
    
#     for c in range(num_classes):
#         y_true = all_labels[:, c]
#         y_prob = all_probs[:, c]
        
#         best_score = -1
#         best_t = 0.5
#         best_p, best_r = 0, 0
        
#         # Search thresholds
#         for t in np.arange(0.1, 0.9, 0.02):
#             y_pred = (y_prob >= t).astype(int)
            
#             if y_pred.sum() == 0:
#                 continue
            
#             p = precision_score(y_true, y_pred, zero_division=0)
#             r = recall_score(y_true, y_pred, zero_division=0)
            
#             # F-beta score (balanced objective)
#             score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
            
#             if score > best_score:
#                 best_score = score
#                 best_t = t
#                 best_p, best_r = p, r
        
#         best_thresholds.append(best_t)
        
#         n_pos = y_true.sum()
#         print(f"{c:<8} {int(n_pos):<10} {best_t:<12.3f} {best_p:<12.3f} {best_r:<12.3f} "
#               f"{best_score:<10.3f}")
    
#     print("-" * 90)
    
#     return np.array(best_thresholds)




# # ==================== Example Usage ====================
# """
# # Option 1: Your original method (simple and effective)
# best_thresholds = find_best_thresholds(
#     model, val_loader, device="cuda", num_classes=7
# )

# # Option 2: With precision/recall constraints (for imbalanced data)
# best_thresholds = find_thresholds_with_constraints(
#     model, val_loader, device="cuda", num_classes=7,
#     target_precision=0.7,  # At least 70% precision
#     target_recall=0.6      # At least 60% recall
# )

# # Option 3: Adaptive search (automatically adjusts for rare classes)
# best_thresholds = adaptive_threshold_search(
#     model, val_loader, device="cuda", num_classes=7
# )

# # Then evaluate
# evaluate_model(model, val_loader, y_val, device="cuda", 
#                best_thresholds=best_thresholds, target_names=target_names)
# """


# import numpy as np
# import torch
# from sklearn.metrics import f1_score, precision_score, recall_score


# def find_balanced_thresholds(model, val_loader, device, num_classes,
#                             precision_weight=1.0, recall_weight=1.0):
#     """
#     Find thresholds that balance precision and recall based on custom weights
    
#     Args:
#         precision_weight: Weight for precision (e.g., 1.5 means prefer precision)
#         recall_weight: Weight for recall (e.g., 2.0 means prefer recall)
    
#     F_beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
#     where beta = sqrt(recall_weight / precision_weight)
#     """
#     from sklearn.metrics import precision_score, recall_score, fbeta_score
    
#     model.eval()
#     all_probs = []
#     all_labels = []
    
#     with torch.no_grad():
#         for signals, labels in val_loader:
#             signals = signals.to(device)
#             outputs = model(signals)
#             if isinstance(outputs, tuple):
#                 outputs = outputs[0]
#             probs = torch.sigmoid(outputs).cpu().numpy()
#             all_probs.append(probs)
#             all_labels.append(labels.numpy())
    
#     all_probs = np.vstack(all_probs)
#     all_labels = np.vstack(all_labels)
    
#     # Calculate beta for F-beta score
#     beta = np.sqrt(recall_weight / precision_weight)
    
#     best_thresholds = []
    
#     print(f"\n🎯 Finding balanced thresholds (P_weight={precision_weight}, R_weight={recall_weight}):")
#     print("-" * 90)
#     print(f"{'Class':<8} {'Samples':<10} {'Threshold':<12} {'Precision':<12} {'Recall':<12} "
#           f"{'F{beta:.1f}':<10}")
#     print("-" * 90)
    
#     for c in range(num_classes):
#         y_true = all_labels[:, c]
#         y_prob = all_probs[:, c]
        
#         best_score = -1
#         best_t = 0.5
#         best_p, best_r = 0, 0
        
#         # Search thresholds
#         for t in np.arange(0.1, 0.9, 0.02):
#             y_pred = (y_prob >= t).astype(int)
            
#             if y_pred.sum() == 0:
#                 continue
            
#             p = precision_score(y_true, y_pred, zero_division=0)
#             r = recall_score(y_true, y_pred, zero_division=0)
            
#             # F-beta score (balanced objective)
#             score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
            
#             if score > best_score:
#                 best_score = score
#                 best_t = t
#                 best_p, best_r = p, r
        
#         best_thresholds.append(best_t)
        
#         n_pos = y_true.sum()
#         print(f"{c:<8} {int(n_pos):<10} {best_t:<12.3f} {best_p:<12.3f} {best_r:<12.3f} "
#               f"{best_score:<10.3f}")
    
#     print("-" * 90)
    
#     return np.array(best_thresholds)



# def find_best_thresholds(model, val_loader, device, num_classes, 
#                           search_grid=None, metric="f1", 
#                           min_precision=None, min_recall=None,
#                           verbose=True):
#     """
#     Enhanced threshold tuning with additional options
    
#     Args:
#         model: Trained model
#         val_loader: Validation dataloader
#         device: cuda/cpu
#         num_classes: Number of classes
#         search_grid: Array of thresholds to try (default: 0.05-0.95 in steps of 0.05)
#         metric: "f1" | "precision" | "recall" | "balanced"
#         min_precision: If set, only consider thresholds with precision >= this
#         min_recall: If set, only consider thresholds with recall >= this
#         verbose: Print per-class stats
    
#     Returns:
#         best_thresholds: (num_classes,) array
#     """
#     model.eval()
#     preds = []
#     trues = []
    
#     with torch.no_grad():
#         for x, y in val_loader:
#             x = x.to(device)
#             out = model(x)
            
#             # Handle tuple output (e.g., logits + attention)
#             if isinstance(out, tuple):
#                 out = out[0]
            
#             probs = torch.sigmoid(out).cpu().numpy()
#             preds.append(probs)
#             trues.append(y.numpy())
    
#     preds = np.vstack(preds)
#     trues = np.vstack(trues)
    
#     # Use finer grid for better threshold search
#     if search_grid is None:
#         search_grid = np.arange(0.05, 0.96, 0.05)  # 0.05, 0.10, ..., 0.95
    
#     best_thresholds = np.zeros(num_classes)
    
#     if verbose:
#         print("\n Finding optimal thresholds per class:")
#         print("-" * 85)
#         print(f"{'Class':<8} {'Samples':<10} {'Pos%':<8} {'Threshold':<12} "
#               f"{'Precision':<12} {'Recall':<12} {'F1':<10}")
#         print("-" * 85)
    
#     for c in range(num_classes):
#         y_true = trues[:, c]
#         y_prob = preds[:, c]
        
#         n_pos = y_true.sum()
#         n_total = len(y_true)
#         pos_rate = 100 * n_pos / n_total
        
#         best_score = -1
#         best_t = 0.5
#         best_stats = {"precision": 0, "recall": 0, "f1": 0}
        
#         for t in search_grid:
#             y_pred = (y_prob >= t).astype(int)
            
#             # Calculate metrics
#             p = precision_score(y_true, y_pred, zero_division=0)
#             r = recall_score(y_true, y_pred, zero_division=0)
#             f1 = f1_score(y_true, y_pred, zero_division=0)
            
#             # Apply constraints if specified
#             if min_precision is not None and p < min_precision:
#                 continue
#             if min_recall is not None and r < min_recall:
#                 continue
            
#             # Calculate score based on metric
#             if metric == "f1":
#                 score = f1
#             elif metric == "precision":
#                 score = p
#             elif metric == "recall":
#                 score = r
#             elif metric == "balanced":
#                 # Harmonic mean of precision and recall (same as F1)
#                 score = f1
#             else:
#                 score = f1
            
#             if score > best_score:
#                 best_score = score
#                 best_t = t
#                 best_stats = {"precision": p, "recall": r, "f1": f1}
        
#         best_thresholds[c] = best_t
        
#         if verbose:
#             print(f"{c:<8} {int(n_pos):<10} {pos_rate:<8.1f} {best_t:<12.3f} "
#                   f"{best_stats['precision']:<12.3f} {best_stats['recall']:<12.3f} "
#                   f"{best_stats['f1']:<10.3f}")
    
#     if verbose:
#         print("-" * 85)
#         print(f"\n Best thresholds: {best_thresholds}")
    
#     return best_thresholds


# def find_thresholds_with_constraints(model, val_loader, device, num_classes,
#                                       target_precision=0.7, target_recall=0.7,
#                                       search_grid=None, verbose=True):
#     """
#     Find thresholds that satisfy both precision AND recall constraints
#     Useful when you want to balance precision/recall for imbalanced classes
    
#     Args:
#         target_precision: Minimum acceptable precision (e.g., 0.7)
#         target_recall: Minimum acceptable recall (e.g., 0.7)
#     """
#     return find_best_thresholds(
#         model, val_loader, device, num_classes,
#         search_grid=search_grid,
#         metric="f1",  # Among valid thresholds, pick the one with best F1
#         min_precision=target_precision,
#         min_recall=target_recall,
#         verbose=verbose
#     )


# def adaptive_threshold_search(model, val_loader, device, num_classes, 
#                                class_weights=None, verbose=True):
#     """
#     Adaptive threshold search based on class frequency
#     - Rare classes get wider search range and lower default threshold
#     - Common classes get narrower search range
    
#     Args:
#         class_weights: (num_classes,) array of class frequencies
#                        If None, will be computed from val_loader
#     """
#     model.eval()
#     preds = []
#     trues = []
    
#     with torch.no_grad():
#         for x, y in val_loader:
#             x = x.to(device)
#             out = model(x)
#             if isinstance(out, tuple):
#                 out = out[0]
#             probs = torch.sigmoid(out).cpu().numpy()
#             preds.append(probs)
#             trues.append(y.numpy())
    
#     preds = np.vstack(preds)
#     trues = np.vstack(trues)
    
#     # Compute class frequencies if not provided
#     if class_weights is None:
#         class_weights = trues.sum(axis=0) / len(trues)
    
#     best_thresholds = np.zeros(num_classes)
    
#     if verbose:
#         print("\n Adaptive threshold search:")
#         print("-" * 85)
#         print(f"{'Class':<8} {'Frequency':<12} {'Search Range':<20} "
#               f"{'Best T':<12} {'F1':<10}")
#         print("-" * 85)
    
#     for c in range(num_classes):
#         y_true = trues[:, c]
#         y_prob = preds[:, c]
#         freq = class_weights[c]
        
#         # Adaptive search grid based on class frequency
#         if freq < 0.05:  # Very rare class (< 5%)
#             search_grid = np.arange(0.1, 0.7, 0.05)  # Lower thresholds
#         elif freq < 0.15:  # Rare class (5-15%)
#             search_grid = np.arange(0.2, 0.8, 0.05)
#         else:  # Common class (> 15%)
#             search_grid = np.arange(0.3, 0.9, 0.05)  # Higher thresholds
        
#         best_f1 = -1
#         best_t = 0.5
        
#         for t in search_grid:
#             y_pred = (y_prob >= t).astype(int)
#             f1 = f1_score(y_true, y_pred, zero_division=0)
            
#             if f1 > best_f1:
#                 best_f1 = f1
#                 best_t = t
        
#         best_thresholds[c] = best_t
        
#         if verbose:
#             print(f"{c:<8} {freq:<12.3f} {f'[{search_grid[0]:.2f}-{search_grid[-1]:.2f}]':<20} "
#                   f"{best_t:<12.3f} {best_f1:<10.3f}")
    
#     if verbose:
#         print("-" * 85)
    
#     return best_thresholds


# # ==================== Example Usage ====================
# """
# # Option 1: Your original method (simple and effective)
# best_thresholds = find_best_thresholds(
#     model, val_loader, device="cuda", num_classes=7
# )

# # Option 2: With precision/recall constraints (for imbalanced data)
# best_thresholds = find_thresholds_with_constraints(
#     model, val_loader, device="cuda", num_classes=7,
#     target_precision=0.7,  # At least 70% precision
#     target_recall=0.6      # At least 60% recall
# )

# # Option 3: Adaptive search (automatically adjusts for rare classes)
# best_thresholds = adaptive_threshold_search(
#     model, val_loader, device="cuda", num_classes=7
# )

# # Then evaluate
# evaluate_model(model, val_loader, y_val, device="cuda", 
#                best_thresholds=best_thresholds, target_names=target_names)
# """


# import numpy as np
# import torch
# from sklearn.metrics import f1_score, precision_recall_curve
# from tqdm import tqdm

# def find_optimal_thresholds(model, val_loader, device="cuda", metric="f1"):
#     """
#     Find optimal threshold for each class independently
    
#     Args:
#         model: Trained model
#         val_loader: Validation dataloader
#         metric: "f1" | "precision" | "recall" | "balanced"
    
#     Returns:
#         best_thresholds: (num_classes,) array of optimal thresholds
#     """
#     model.eval()
#     all_probs = []
#     all_labels = []
    
#     print("🔍 Collecting predictions on validation set...")
#     with torch.no_grad():
#         for signals, labels in tqdm(val_loader, desc="Inference"):
#             signals = signals.to(device)
#             outputs = model(signals)
#             probs = torch.sigmoid(outputs).cpu().numpy()
#             all_probs.append(probs)
#             all_labels.append(labels.numpy())
    
#     all_probs = np.vstack(all_probs)
#     all_labels = np.vstack(all_labels)
#     num_classes = all_labels.shape[1]
    
#     best_thresholds = []
    
#     print(f"\nFinding optimal thresholds per class (metric={metric}):\n")
    
#     for i in range(num_classes):
#         y_true = all_labels[:, i]
#         y_prob = all_probs[:, i]
        
#         # Try different thresholds
#         thresholds = np.arange(0.1, 0.9, 0.01)
#         best_score = 0
#         best_thresh = 0.5
        
#         for thresh in thresholds:
#             y_pred = (y_prob >= thresh).astype(int)
            
#             if metric == "f1":
#                 score = f1_score(y_true, y_pred, zero_division=0)
#             elif metric == "precision":
#                 from sklearn.metrics import precision_score
#                 score = precision_score(y_true, y_pred, zero_division=0)
#             elif metric == "recall":
#                 from sklearn.metrics import recall_score
#                 score = recall_score(y_true, y_pred, zero_division=0)
#             elif metric == "balanced":
#                 # Balance precision and recall
#                 from sklearn.metrics import precision_score, recall_score
#                 p = precision_score(y_true, y_pred, zero_division=0)
#                 r = recall_score(y_true, y_pred, zero_division=0)
#                 score = 2 * (p * r) / (p + r + 1e-8)  # F1
            
#             if score > best_score:
#                 best_score = score
#                 best_thresh = thresh
        
#         best_thresholds.append(best_thresh)
        
#         # Count samples
#         n_positive = y_true.sum()
#         n_total = len(y_true)
        
#         print(f"   Class {i}: threshold={best_thresh:.3f}, {metric}={best_score:.3f} "
#               f"(pos={n_positive}/{n_total} = {100*n_positive/n_total:.1f}%)")
    
#     return np.array(best_thresholds)

# def find_thresholds_by_precision_recall_curve(model, val_loader, device="cuda", 
#                                                target_precision=0.7, target_recall=0.7):
#     """
#     Find thresholds using precision-recall curve
#     Useful for balancing precision and recall
#     """
#     model.eval()
#     all_probs = []
#     all_labels = []
    
#     with torch.no_grad():
#         for signals, labels in tqdm(val_loader, desc="Collecting predictions"):
#             signals = signals.to(device)
#             outputs = model(signals)
#             probs = torch.sigmoid(outputs).cpu().numpy()
#             all_probs.append(probs)
#             all_labels.append(labels.numpy())
    
#     all_probs = np.vstack(all_probs)
#     all_labels = np.vstack(all_labels)
#     num_classes = all_labels.shape[1]
    
#     best_thresholds = []
    
#     print(f"\nFinding thresholds via PR curve (target: P≥{target_precision}, R≥{target_recall}):\n")
    
#     for i in range(num_classes):
#         y_true = all_labels[:, i]
#         y_prob = all_probs[:, i]
        
#         # Get precision-recall curve
#         precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        
#         # Find threshold where precision >= target AND recall >= target
#         valid_mask = (precisions[:-1] >= target_precision) & (recalls[:-1] >= target_recall)
        
#         if valid_mask.any():
#             # Pick threshold with highest F1
#             f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
#             best_idx = np.argmax(f1_scores * valid_mask)
#             best_thresh = thresholds[best_idx]
#             best_p = precisions[best_idx]
#             best_r = recalls[best_idx]
#         else:
#             # Fallback: maximize F1
#             f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
#             best_idx = np.argmax(f1_scores)
#             best_thresh = thresholds[best_idx]
#             best_p = precisions[best_idx]
#             best_r = recalls[best_idx]
        
#         best_thresholds.append(best_thresh)
        
#         n_positive = y_true.sum()
#         print(f"   Class {i}: threshold={best_thresh:.3f}, P={best_p:.3f}, R={best_r:.3f} "
#               f"(pos={n_positive})")
    
#     return np.array(best_thresholds)

# def evaluate_with_thresholds(model, val_loader, thresholds, device="cuda", target_names=None):
#     """
#     Evaluate model with custom thresholds
#     """
#     from sklearn.metrics import classification_report
    
#     model.eval()
#     all_probs = []
#     all_labels = []
    
#     with torch.no_grad():
#         for signals, labels in tqdm(val_loader, desc="Evaluating"):
#             signals = signals.to(device)
#             outputs = model(signals)
#             probs = torch.sigmoid(outputs).cpu().numpy()
#             all_probs.append(probs)
#             all_labels.append(labels.numpy())
    
#     all_probs = np.vstack(all_probs)
#     all_labels = np.vstack(all_labels)
    
#     # Apply per-class thresholds
#     all_preds = np.zeros_like(all_probs)
#     for i in range(all_probs.shape[1]):
#         all_preds[:, i] = (all_probs[:, i] >= thresholds[i]).astype(int)
    
#     # Print classification report
#     print("\n" + "="*60)
#     print("Classification Report with Optimized Thresholds")
#     print("="*60)
#     print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
    
#     return all_preds, all_probs

# # === Example Usage ===
# """
# # 1. Find optimal thresholds
# best_thresholds = find_optimal_thresholds(
#     model, val_loader, device="cuda", metric="f1"
# )

# # 2. OR use precision-recall curve
# best_thresholds = find_thresholds_by_precision_recall_curve(
#     model, val_loader, device="cuda", 
#     target_precision=0.7, target_recall=0.7
# )

# # 3. Evaluate with new thresholds
# target_names = ["SB", "SR", "AF", "AFIB", "SVT", "ST", "AT"]
# evaluate_with_thresholds(
#     model, val_loader, best_thresholds, 
#     device="cuda", target_names=target_names
# )
# """



# import numpy as np
# import torch
# import torch.nn as nn
# from sklearn.metrics import f1_score

# def find_best_thresholds(model, val_loader, device, num_classes, search_grid=None):
#     model.eval()
#     preds = []
#     trues = []
#     with torch.no_grad():
#         for x, y in val_loader:
#             x = x.to(device)
#             out = model(x)  # expect logits or (logits, att) if your model returns tuple => handle that
#             if isinstance(out, tuple):
#                 out = out[0]
#             probs = torch.sigmoid(out).cpu().numpy()
#             preds.append(probs)
#             trues.append(y.numpy())
#     preds = np.vstack(preds)
#     trues = np.vstack(trues)
#     if search_grid is None:
#         search_grid = np.linspace(0.1, 0.9, 41)
#     best_thresholds = np.zeros(num_classes)
#     for c in range(num_classes):
#         best_f1 = -1
#         best_t = 0.5
#         for t in search_grid:
#             pbin = (preds[:,c] >= t).astype(int)
#             f1 = f1_score(trues[:,c], pbin, zero_division=0)
#             if f1 > best_f1:
#                 best_f1 = f1
#                 best_t = t
#         best_thresholds[c] = best_t
#     return best_thresholds
