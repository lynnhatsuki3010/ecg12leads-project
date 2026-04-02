import torch
import torch.nn as nn
from tqdm import tqdm  # <- thêm tqdm
from training.losses import MultiLabelFocalLoss, effective_num_weights

def compute_pos_weight(y_train, device="cuda"):
    num_samples, num_classes = y_train.shape
    positive_counts = y_train.sum(axis=0)
    negative_counts = num_samples - positive_counts
    pos_weight = negative_counts / (positive_counts + 1e-6)
    return torch.tensor(pos_weight, dtype=torch.float32).to(device)

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, logits, targets):
#         """
#         logits: (B, C)
#         targets: (B, C)
#         """
#         bce_loss = nn.functional.binary_cross_entropy_with_logits(
#             logits, targets, reduction="none"
#         )

#         prob = torch.sigmoid(logits)
#         p_t = prob * targets + (1 - prob) * (1 - targets)

#         focal_factor = (1 - p_t) ** self.gamma

#         loss = self.alpha * focal_factor * bce_loss

#         if self.reduction == "mean":
#             return loss.mean()
#         elif self.reduction == "sum":
#             return loss.sum()
#         else:
#             return loss


def train_model(model, train_loader, val_loader, y_train, num_epochs=20, patience=5, lr=1e-3, device="cuda"): #thêm a_norm
    import copy
    pos_weight = compute_pos_weight(y_train, device)
    model = model.to(device)
    
    # If you're using a fixed adjacency matrix, move it to device (gnn_cnn)
    # if A_norm is not None:
    #     A_norm = A_norm.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    
    # import copy

    # model = model.to(device)

    # # === Thay loss bằng Focal Loss ===
    # #alpha = effective_num_weights(y_train)   # tính alpha theo phân bố dữ liệu
    # criterion = FocalLoss(alpha=1.0, gamma=2.0) # không dùng MultiFocalLoss(alpha=None, gamma=2.0)

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # best_val_loss = float("inf")
    # best_state = copy.deepcopy(model.state_dict())
    # patience_counter = 0

    for epoch in range(num_epochs):
        # === Train ===
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for signals, labels in loop:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(signals) #RESNET501D
            
            #outputs = model(signals, A_norm=None) #gnn_cnn
            
            # NEW (CORRECT) GNN_TRANSFORMER
            # try:
            #     outputs = model(signals, A_norm)  # Passes actual A_norm
            # except TypeError:
            #     outputs = model(signals)  # For models without A_norm
            
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(train_loss=total_loss/len(train_loader))  # hiển thị loss trung bình

        avg_train_loss = total_loss / len(train_loader)

        # === Validate ===
        model.eval()
        total_val_loss = 0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for signals, labels in val_loop:
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                
                #outputs = model(signals, A_norm=None) #gnn_cnn
                
                # NEW (CORRECT) GNN_TRANSFORMER
                # try:
                #     outputs = model(signals, A_norm)  # Passes actual A_norm
                # except TypeError:
                #     outputs = model(signals)  # For models without A_norm
                    
                    
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                val_loop.set_postfix(val_loss=total_val_loss/len(val_loader))

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        # === Early stopping ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping!")
                break

    model.load_state_dict(best_state)
    return model
