import numpy as np
import torch
from sklearn.metrics import classification_report, multilabel_confusion_matrix, precision_recall_fscore_support

def evaluate_model(model, test_loader, y_test, device="cuda", best_thresholds=None, 
                   target_names=None, return_probs=True):
    """
    Hàm đánh giá tiêu chuẩn với support cho visualization.
    
    Returns:
        preds_bin: Binary predictions (N, num_classes)
        preds_prob: Probabilities (N, num_classes)
        cms: Confusion matrices
        metrics_dict: Dict chứa precision, recall, f1 per class (để vẽ bar chart)
    """
    model = model.to(device)
    model.eval()
    preds_prob = []
    preds_bin = []

    print("⏳ Đang thực hiện đánh giá (Standard Evaluation)...")
    
    with torch.no_grad():
        for signals, _ in test_loader:
            signals = signals.to(device)
            outputs = model(signals)

            if isinstance(outputs, tuple):  
                outputs = outputs[0]

            probs = torch.sigmoid(outputs).cpu().numpy()
            preds_prob.append(probs)

    preds_prob = np.vstack(preds_prob)

    # === Apply threshold ===
    if best_thresholds is None:
        preds_bin = (preds_prob >= 0.5).astype(int)
    else:
        preds_bin = (preds_prob >= best_thresholds).astype(int)

    if isinstance(y_test, torch.Tensor):
        y_test = y_test.cpu().numpy()

    # === Metrics ===
    print("\n" + "="*30 + " CLASSIFICATION REPORT " + "="*30)
    print(classification_report(
        y_test, preds_bin,
        zero_division=0,
        target_names=target_names
    ))

    print("\n" + "="*30 + " CONFUSION MATRICES " + "="*30)
    cms = multilabel_confusion_matrix(y_test, preds_bin)
    
    if target_names:
        for i, cm in enumerate(cms):
            print(f"🔹 {target_names[i]}:\n{cm}\n")
    else:
        for i, cm in enumerate(cms):
            print(f"🔹 Label {i}:\n{cm}\n")

    # === Per-class metrics for visualization ===
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds_bin, average=None, zero_division=0
    )
    
    metrics_dict = {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    if return_probs:
        return preds_bin, preds_prob, cms, metrics_dict
    else:
        return preds_bin, preds_prob, cms


def evaluate_model_sliding_window(model, test_loader, y_test, device="cuda", 
                                   best_thresholds=None, target_names=None, return_probs=True):
    """
    Hàm đánh giá nâng cao với TTA support.
    """
    model = model.to(device)
    model.eval()
    
    all_probs = []
    
    print("⏳ Đang thực hiện đánh giá (Inference)...")
    with torch.no_grad():
        for signals, _ in test_loader:
            signals = signals.to(device)
            
            outputs = model(signals)
            if isinstance(outputs, tuple): 
                outputs = outputs[0]
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.append(probs)
            
    y_pred_probs = np.vstack(all_probs)
    
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.cpu().numpy()
        
    if best_thresholds is None:
        best_thresholds = 0.5
        
    y_pred_bin = (y_pred_probs >= best_thresholds).astype(int)
    
    # === Báo cáo ===
    print("\n" + "="*30 + " CLASSIFICATION REPORT (Sliding Window / Robust) " + "="*30)
    print(classification_report(
        y_test, y_pred_bin, 
        target_names=target_names, 
        zero_division=0
    ))
    
    print("\n" + "="*30 + " CONFUSION MATRICES " + "="*30)
    cms = multilabel_confusion_matrix(y_test, y_pred_bin)
    
    if target_names:
        for i, cm in enumerate(cms):
            print(f"🔹 {target_names[i]}:\n{cm}\n")
    
    # Per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_bin, average=None, zero_division=0
    )
    
    metrics_dict = {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    if return_probs:
        return y_pred_bin, y_pred_probs, cms, metrics_dict
    else:
        return y_pred_bin, y_pred_probs, cms


if __name__ == "__main__":
    print("✅ Evaluation module with visualization support ready!")


# import numpy as np
# import torch
# from sklearn.metrics import classification_report, multilabel_confusion_matrix

# def evaluate_model(model, test_loader, y_test, device="cuda", best_thresholds=None, target_names=None):
#     model = model.to(device)
#     model.eval()
#     preds_prob = []   # lưu xác suất
#     preds_bin = []    # lưu nhị phân theo threshold

#     with torch.no_grad():
#         for signals, _ in test_loader:
#             signals = signals.to(device)

#             outputs = model(signals)

#             if isinstance(outputs, tuple):  
#                 # nếu model trả (logits, attention)
#                 outputs = outputs[0]

#             probs = torch.sigmoid(outputs).cpu().numpy()
#             preds_prob.append(probs)

#     preds_prob = np.vstack(preds_prob)

#     # === Apply threshold ===
#     if best_thresholds is None:
#         # dùng threshold 0.5 mặc định
#         preds_bin = (preds_prob >= 0.5).astype(int)
#     else:
#         # per-class threshold vector: shape (num_classes,)
#         preds_bin = (preds_prob >= best_thresholds).astype(int)

#     # y_test có thể là tensor
#     if isinstance(y_test, torch.Tensor):
#         y_test = y_test.cpu().numpy()

#     print("\n=== Classification Report ===")
#     print(classification_report(
#         y_test, preds_bin,
#         zero_division=0,
#         target_names=target_names
#     ))

#     print("\n=== Confusion Matrices ===")
#     cms = multilabel_confusion_matrix(y_test, preds_bin)
#     for i, cm in enumerate(cms):
#         print(f"{target_names[i] if target_names else f'Label {i}'}:\n{cm}\n")

#     return preds_bin, preds_prob, cms
