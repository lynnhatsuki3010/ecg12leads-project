# processing/feature_extraction_georgia.py
"""
Trích xuất và xử lý tín hiệu ECG từ Georgia (.mat) -> lưu dưới dạng .npy
Tương đương feature_extraction.py của Chapman.

Bao gồm:
 - Đọc danh sách đường dẫn file ECG từ metadata Georgia
 - Đọc tín hiệu từ file .mat — tái dùng load_ecg_mat() vì format giống Chapman
 - Lọc nhiễu (Notch + Bandpass 0.5-50Hz) — tái dùng process_ecg_signal()
 - Cắt/Pad về độ dài cố định (Fix Length = 5000 samples = 10s)
 - Chuẩn hóa Z-score
 - Kiểm tra NaN/Inf và lưu thành mảng numpy (12, 5000)

Yêu cầu:
 - Phải chạy preprocess_georgia.py trước để có file metadata
"""

import os
import numpy as np
from processing.denoised_ecg import load_ecg_mat, process_ecg_signal


def process_and_save_georgia_files(df_metadata, save_dir="processed_georgia",
                                    fs=500, target_length=5000,
                                    verbose=True):
    """
    Tiền xử lý toàn bộ ECG Georgia theo pipeline chuẩn và lưu ra .npy.
    Tương đương process_and_save_ecg_files() trong feature_extraction.py.

    Args:
        df_metadata (pd.DataFrame): Bảng metadata từ preprocess_georgia.py
                                    Phải có cột 'file_path' và 'record_id'
        save_dir (str): Thư mục lưu file npy sau xử lý
        fs (int): Tần số lấy mẫu (mặc định 500Hz)
        target_length (int): Độ dài cố định (mặc định 5000 samples = 10s)
        verbose (bool): In log tiến độ nếu True

    Returns:
        valid_indices (list[int]): Danh sách index trong df_metadata đã xử lý thành công
    """
    os.makedirs(save_dir, exist_ok=True)
    valid_indices = []

    total_files = len(df_metadata)
    if verbose:
        print(f"🚀 Bắt đầu xử lý {total_files} bản ghi Georgia...")
        print(f"   Config: FS={fs}, Target Length={target_length}, Save Dir={save_dir}")

    for i, (_, row) in enumerate(df_metadata.iterrows()):
        # file_path trỏ tới .hea, chuyển sang .mat
        hea_path  = row["file_path"]
        base      = os.path.splitext(hea_path)[0]
        mat_path  = base + ".mat"
        file_name = row["record_id"]   # Ví dụ: E00001

        if not os.path.exists(mat_path):
            if verbose:
                print(f"[Bỏ qua] Không tìm thấy file .mat: {mat_path}")
            continue

        try:
            # 1️⃣ Load dữ liệu thô — tái dùng load_ecg_mat() vì format giống Chapman
            raw_signals = load_ecg_mat(mat_path)   # shape (12, N)

            if raw_signals is None:
                if verbose:
                    print(f"[Lỗi] Không đọc được dữ liệu từ {mat_path}")
                continue

            # 2️⃣ Pipeline xử lý: Lọc nhiễu -> Fix Length -> Z-score
            # Tái dùng hoàn toàn process_ecg_signal() từ denoised_ecg.py
            processed_sig = process_ecg_signal(
                raw_signals,
                fs=fs,
                use_zscore=True,
                target_len=target_length
            )

            # 3️⃣ Kiểm tra NaN/Inf (Safety check)
            if np.isnan(processed_sig).any() or np.isinf(processed_sig).any():
                if verbose:
                    print(f"[Cảnh báo] Phát hiện NaN/Inf tại record {file_name} (index {i}) -> Bỏ qua.")
                continue

            # 4️⃣ Giới hạn giá trị (Clip) để loại bỏ outlier cực đoan
            processed_sig = np.clip(processed_sig, -10, 10)

            # 5️⃣ Lưu file numpy — giữ nguyên tên gốc: E00001.npy
            save_path = os.path.join(save_dir, f"{file_name}.npy")
            np.save(save_path, processed_sig)

            valid_indices.append(i)

            # Log tiến độ mỗi 100 files
            if verbose and (i + 1) % 100 == 0:
                print(f"✓ Đã xử lý {i+1}/{total_files} file")

        except Exception as e:
            if verbose:
                print(f"[Ngoại lệ] Lỗi tại file {file_name} (index {i}): {str(e)}")
            continue

    # Lưu danh sách indices hợp lệ — giống Chapman
    indices_path = os.path.join(os.path.dirname(save_dir), "valid_indices_georgia.npy")
    np.save(indices_path, np.array(valid_indices))

    if verbose:
        print(f"✅ Hoàn tất! Đã lưu {len(valid_indices)}/{total_files} bản ghi hợp lệ vào '{save_dir}'.")
        print(f"   Danh sách valid_indices đã lưu tại: {indices_path}")

    return valid_indices


# === Ví dụ chạy độc lập (Test) ===
if __name__ == "__main__":
    import pandas as pd
    from processing.preprocess_georgia import (
        create_metadata_from_georgia,
        map_snomed_codes_georgia,
        clean_metadata_georgia,
        filter_target_labels_georgia,
    )

    GEORGIA_ROOT = r"E:\NCKH - 2026\ECG Project\data\raw\Georgia"
    SAVE_DIR     = r"E:\NCKH - 2026\ECG Project\data\processed_georgia\ecg_leads12_georgia"

    df = create_metadata_from_georgia(GEORGIA_ROOT)
    df = map_snomed_codes_georgia(df)
    df = clean_metadata_georgia(df)
    df = filter_target_labels_georgia(df)

    process_and_save_georgia_files(df, save_dir=SAVE_DIR)
