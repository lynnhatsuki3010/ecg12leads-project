import torch
import torch.nn as nn
from tqdm import tqdm  # <- thêm tqdm

def compute_pos_weight(y_train, device="cuda"):
    num_samples, num_classes = y_train.shape
    positive_counts = y_train.sum(axis=0)
    negative_counts = num_samples - positive_counts
    pos_weight = negative_counts / (positive_counts + 1e-6)
    return torch.tensor(pos_weight, dtype=torch.float32).to(device)


def train_model(model, train_loader, val_loader, y_train, num_epochs=20, patience=5, lr=1e-3, device="cuda"):
    import copy
    pos_weight = compute_pos_weight(y_train, device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(num_epochs):
        # === Train ===
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for signals, labels in loop:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(signals)
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
