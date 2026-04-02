import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from itertools import cycle

def plot_confusion_matrix(y_true, y_pred, target_names, save_path=None):
    """
    Vẽ confusion matrix đẹp cho từng class (multilabel)
    
    Args:
        y_true: Ground truth labels (N, num_classes)
        y_pred: Predicted labels (N, num_classes) 
        target_names: List tên các class
        save_path: Đường dẫn lưu ảnh (optional)
    """
    num_classes = y_true.shape[1]
    
    # Tính số hàng/cột cho subplot grid
    ncols = 3
    nrows = (num_classes + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten() if num_classes > 1 else [axes]
    
    for i in range(num_classes):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        
        # Vẽ heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    cbar=True, square=True,
                    xticklabels=['Neg', 'Pos'],
                    yticklabels=['Neg', 'Pos'],
                    ax=axes[i])
        
        axes[i].set_title(f'{target_names[i]}', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Actual', fontsize=10)
        axes[i].set_xlabel('Prediction', fontsize=10)
    
    # Ẩn các subplot thừa
    for j in range(num_classes, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrices saved to {save_path}")
    
    plt.show()


def plot_training_history(history, save_path=None):
    """
    Vẽ biểu đồ Train/Val Loss và Metrics
    
    Args:
        history: Dict chứa {'train_loss': [], 'val_loss': [], 
                           'train_acc': [], 'val_acc': [], 
                           'train_f1': [], 'val_f1': []}
        save_path: Đường dẫn lưu ảnh
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'orange', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epochs', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Accuracy (nếu có)
    if 'train_acc' in history and 'val_acc' in history:
        axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, history['val_acc'], 'orange', label='Val Accuracy', linewidth=2)
        axes[0, 1].set_title('Accuracy Over Epochs', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epochs', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].legend(loc='lower right')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'Accuracy not tracked', 
                       ha='center', va='center', fontsize=12)
        axes[0, 1].axis('off')
    
    # 3. Micro F1
    if 'train_f1' in history and 'val_f1' in history:
        axes[1, 0].plot(epochs, history['train_f1'], 'b-', label='Train Micro F1', linewidth=2)
        axes[1, 0].plot(epochs, history['val_f1'], 'orange', label='Val Micro F1', linewidth=2)
        axes[1, 0].set_title('Micro F1 Score Over Epochs', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epochs', fontsize=12)
        axes[1, 0].set_ylabel('Micro F1', fontsize=12)
        axes[1, 0].legend(loc='lower right')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Macro F1
    if 'train_macro_f1' in history and 'val_macro_f1' in history:
        axes[1, 1].plot(epochs, history['train_macro_f1'], 'b-', label='Train Macro F1', linewidth=2)
        axes[1, 1].plot(epochs, history['val_macro_f1'], 'orange', label='Val Macro F1', linewidth=2)
        axes[1, 1].set_title('Macro F1 Score Over Epochs', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epochs', fontsize=12)
        axes[1, 1].set_ylabel('Macro F1', fontsize=12)
        axes[1, 1].legend(loc='lower right')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training history saved to {save_path}")
    
    plt.show()


def plot_roc_curves(y_true, y_pred_probs, target_names, save_path=None):
    """
    Vẽ ROC curves cho multilabel classification
    
    Args:
        y_true: Ground truth (N, num_classes)
        y_pred_probs: Predicted probabilities (N, num_classes)
        target_names: List tên các class
        save_path: Đường dẫn lưu ảnh
    """
    num_classes = y_true.shape[1]
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Plot micro-average
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
             color='deeppink', linestyle=':', linewidth=3)
    
    # Plot each class
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{target_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    # Plot random classifier
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Multilabel Classification', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC curves saved to {save_path}")
    
    plt.show()
    
    return roc_auc


def plot_per_class_metrics(metrics_dict, target_names, save_path=None):
    """
    Vẽ bar chart cho metrics của từng class
    
    Args:
        metrics_dict: Dict {'precision': [], 'recall': [], 'f1': []}
        target_names: List tên các class
        save_path: Đường dẫn lưu ảnh
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    x = np.arange(len(target_names))
    width = 0.6
    
    # Precision
    axes[0].bar(x, metrics_dict['precision'], width, color='skyblue', edgecolor='black')
    axes[0].set_title('Precision per Class', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Precision', fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(target_names, rotation=45, ha='right')
    axes[0].set_ylim([0, 1.0])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Recall
    axes[1].bar(x, metrics_dict['recall'], width, color='lightcoral', edgecolor='black')
    axes[1].set_title('Recall per Class', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Recall', fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(target_names, rotation=45, ha='right')
    axes[1].set_ylim([0, 1.0])
    axes[1].grid(axis='y', alpha=0.3)
    
    # F1-Score
    axes[2].bar(x, metrics_dict['f1'], width, color='lightgreen', edgecolor='black')
    axes[2].set_title('F1-Score per Class', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('F1-Score', fontsize=12)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(target_names, rotation=45, ha='right')
    axes[2].set_ylim([0, 1.0])
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Per-class metrics saved to {save_path}")
    
    plt.show()


# Example usage
if __name__ == "__main__":
    print("✅ Visualization module ready!")
    print("\nAvailable functions:")
    print("  - plot_confusion_matrix()")
    print("  - plot_training_history()")
    print("  - plot_roc_curves()")
    print("  - plot_per_class_metrics()")