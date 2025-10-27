# processing/feature_extraction.py
"""
Trích xuất và xử lý tín hiệu ECG từ file .mat -> lưu dưới dạng .npy
Bao gồm:
 - Đọc danh sách đường dẫn file ECG (.mat)
 - Lọc nhiễu, chuẩn hóa, kiểm tra NaN/Inf
 - Lưu thành mảng numpy (12, N)
"""

import os
import numpy as np
from processing.denoised_ecg import load_ecg_mat, normalize_to_mV, denoise_all_leads


def process_and_save_ecg_files(X_paths, save_dir="processed_ecg_omaigah", fs=500, verbose=True):
    """
    Tiền xử lý toàn bộ ECG trong danh sách X_paths và lưu ra .npy.

    Args:
        X_paths (list[str]): Danh sách đường dẫn file .hea hoặc .mat gốc.
        save_dir (str): Thư mục lưu file npy sau xử lý.
        fs (int): Tần số lấy mẫu.
        verbose (bool): In log chi tiết nếu True.
    """
    os.makedirs(save_dir, exist_ok=True)
    valid_indices = []

    for i, path in enumerate(X_paths):
        base, ext = os.path.splitext(path)
        mat_path = base + ".mat"

        if not os.path.exists(mat_path):
            if verbose:
                print(f"[Bỏ qua] Không tìm thấy file .mat: {mat_path}")
            continue

        try:
            # 1️⃣ Load và chuyển sang mV
            signals = load_ecg_mat(mat_path)  # shape (12, N)
            raw_mV = normalize_to_mV(signals, gain=1000.0)

            # 2️⃣ Lọc nhiễu
            denoised = denoise_all_leads(raw_mV, fs)

            # 3️⃣ Kiểm tra NaN/Inf
            if np.isnan(denoised).any() or np.isinf(denoised).any():
                if verbose:
                    print(f"[Cảnh báo] NaN/Inf tại record {i}")
                continue

            # 4️⃣ Chuẩn hóa z-score từng lead
            sig_z = (denoised - denoised.mean(axis=1, keepdims=True)) / (
                denoised.std(axis=1, keepdims=True) + 1e-8
            )

            # 5️⃣ Giới hạn giá trị để tránh outlier
            sig_z = np.clip(sig_z, -10, 10)

            # 6️⃣ Kiểm tra lại NaN/Inf
            if np.isnan(sig_z).any() or np.isinf(sig_z).any():
                if verbose:
                    print(f"[Cảnh báo] NaN/Inf sau chuẩn hóa tại record {i}")
                continue

            # 7️⃣ Lưu file numpy
            np.save(os.path.join(save_dir, f"record_{i}.npy"), sig_z)
            valid_indices.append(i)

            if verbose and (i + 1) % 100 == 0:
                print(f"✓ Đã xử lý {i+1}/{len(X_paths)} file")

        except Exception as e:
            if verbose:
                print(f"[Lỗi] record {i}: {e}")
            continue

    print(f"✅ Hoàn tất! Lưu {len(valid_indices)} bản ghi hợp lệ vào '{save_dir}'.")
    return valid_indices


# === Ví dụ chạy độc lập ===
if __name__ == "__main__":
    # Ví dụ giả định: bạn đã có danh sách file ECG
    sample_paths = [
        r"E:\NCKH - 2026\ECG Project\data\raw\WFDBRecords\01\010\JS00001.hea",
    ]
    process_and_save_ecg_files(sample_paths, save_dir="processed_ecg_omaigah")
