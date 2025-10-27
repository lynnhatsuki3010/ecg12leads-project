import numpy as np
import torch
from sklearn.metrics import classification_report, multilabel_confusion_matrix

def evaluate_model(model, test_loader, y_test, device="cuda"):
    model = model.to(device)
    model.eval()
    preds = []

    with torch.no_grad():
        for signals, _ in test_loader:
            signals = signals.to(device)
            outputs = model(signals)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds.append(probs)

    preds = np.vstack(preds)
    preds_bin = (preds > 0.5).astype(int)

    # ⚠️ y_test có thể là tensor → convert sang numpy
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.cpu().numpy()

    print("\nClassification Report:")
    print(classification_report(y_test, preds_bin, target_names=[str(c) for c in range(y_test.shape[1])]))

    print("\nConfusion Matrices:")
    cms = multilabel_confusion_matrix(y_test, preds_bin)
    for i, cm in enumerate(cms):
        print(f"Label {i}:\n{cm}\n")

    return preds_bin, preds, cms
