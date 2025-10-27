# main.py
"""
MAIN PIPELINE — ECG 12-Lead Classification
-------------------------------------------
Quy trình chính gồm:
1️⃣ Tiền xử lý WFDB và tạo metadata sạch
2️⃣ Phân tích và trực quan hóa top bệnh
3️⃣ Huấn luyện mô hình ResNet1D + Attention
4️⃣ Đánh giá và tìm ngưỡng tối ưu
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# === Import nội bộ ===
from processing.preprocess import run_preprocessing
from visualize.plot_statistics import analyze_top_diseases
from misc.gpu_check import check_gpu
from misc.threshold_finder import find_best_threshold
from training.model import ResNet1DAttention
from training.train import train_model
from training.evaluate import evaluate_model
from training.dataset_loader import ECGDataset


# === Cấu hình đường dẫn ===
ROOT_DIR = r"E:\NCKH - 2026\ECG Project"
WFDB_DIR = os.path.join(ROOT_DIR, "data\raw\WFDBRecords")
SNOMED_PATH = os.path.join(ROOT_DIR, "data\raw\ConditionNames_SNOMED-CT.csv")
METADATA_CLEAN = os.path.join(ROOT_DIR, "data\processed\patient_metadata_clean.xlsx")
SPLIT_DIR = os.path.join(ROOT_DIR, "data")


# ============================================================
def main(mode="train"):
    """
    mode: "preprocess", "analyze", hoặc "train"
    """
    print("🚀 BẮT ĐẦU PIPELINE ECG 12-LEAD\n")

    # === (1) Tiền xử lý dữ liệu ===
    if mode in ["preprocess", "train"]:
        print("🔹 Bước 1: Tạo và làm sạch metadata từ WFDB...")
        run_preprocessing()

    # === (2) Phân tích thống kê ===
    if mode in ["analyze", "train"]:
        print("\n🔹 Bước 2: Phân tích top bệnh phổ biến...")
        analyze_top_diseases(METADATA_CLEAN, SNOMED_PATH, top_n=10)

    if mode == "preprocess" or mode == "analyze":
        print("\n✅ Hoàn tất chế độ rút gọn:", mode)
        return

    # === (3) Kiểm tra GPU ===
    print("\n🔹 Bước 3: Kiểm tra GPU...")
    device = check_gpu()

    # === (4) Load split files ===
    print("\n🔹 Bước 4: Chuẩn bị dataset & DataLoader...")

    def load_split_file(filename):
        path = os.path.join(SPLIT_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Không tìm thấy file split: {path}")
        return np.load(path, allow_pickle=True)

    train_files = load_split_file("train_files.npy")
    val_files   = load_split_file("val_files.npy")
    test_files  = load_split_file("test_files.npy")
    y_train     = load_split_file("y_train.npy")
    y_val       = load_split_file("y_val.npy")
    y_test      = load_split_file("y_test.npy")

    # Tạo dataset
    train_dataset = ECGDataset(train_files, y_train, augment=True)
    val_dataset   = ECGDataset(val_files, y_val, augment=False)
    test_dataset  = ECGDataset(test_files, y_test, augment=False)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    print(f"✅ Dataset: {len(train_loader)} train batches, {len(val_loader)} val, {len(test_loader)} test")

    # === (5) Khởi tạo & huấn luyện mô hình ===
    print("\n🔹 Bước 5: Huấn luyện mô hình ResNet1D + Attention...")
    model = ResNet1DAttention(num_classes=y_train.shape[1])
    model = train_model(model, train_loader, val_loader, device=device)

    # === (6) Đánh giá sơ bộ ===
    print("\n🔹 Bước 6: Đánh giá sơ bộ trên test set...")
    evaluate_model(model, test_loader, y_test, device=device)

    # === (7) Tính xác suất & tìm ngưỡng tối ưu ===
    print("\n🔹 Bước 7: Tìm ngưỡng tối ưu theo F1-score...")
    model.eval()
    y_true, y_pred_proba = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = torch.sigmoid(model(X_batch))
            y_true.append(y_batch.numpy())
            y_pred_proba.append(outputs.cpu().numpy())

    y_true = np.vstack(y_true)
    y_pred_proba = np.vstack(y_pred_proba)
    best_thresholds = find_best_threshold(y_true, y_pred_proba)
    print("✅ Ngưỡng tối ưu từng class:", np.round(best_thresholds, 3))

    # === (8) Đánh giá cuối cùng ===
    print("\n🔹 Bước 8: Đánh giá cuối cùng với ngưỡng tối ưu...")
    y_pred = (y_pred_proba > best_thresholds).astype(int)
    print(classification_report(
        y_true, y_pred, target_names=[f"Class_{i}" for i in range(y_pred.shape[1])]
    ))

    print("\n🎯 HOÀN TẤT PIPELINE — MÔ HÌNH SẴN SÀNG ĐÁNH GIÁ VÀ DỰ ĐOÁN.")


# ============================================================
if __name__ == "__main__":
    # Thay đổi mode tùy nhu cầu: "preprocess", "analyze", hoặc "train"
    main(mode="train")
