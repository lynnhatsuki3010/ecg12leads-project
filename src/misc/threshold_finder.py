# misc/threshold_finder.py
import numpy as np
from sklearn.metrics import f1_score

def find_best_threshold(y_true, y_pred_proba):
    thresholds = np.linspace(0.1, 0.9, 17)
    best_thresh = []
    for i in range(y_true.shape[1]):
        best_t, best_f1 = 0.5, 0
        for t in thresholds:
            preds = (y_pred_proba[:, i] > t).astype(int)
            f1 = f1_score(y_true[:, i], preds)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresh.append(best_t)
    return np.array(best_thresh)
