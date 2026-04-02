# processing/feature_extraction.py
"""
Trích xuất và xử lý tín hiệu ECG từ file .mat -> lưu dưới dạng .npy
Bao gồm:
 - Đọc danh sách đường dẫn file ECG (.mat)
 - Lọc nhiễu (Notch + Bandpass 0.5-50Hz)
 - Cắt/Pad về độ dài cố định (Fix Length)
 - Chuẩn hóa Z-score
 - Kiểm tra NaN/Inf và lưu thành mảng numpy
"""

import os
import numpy as np
from processing.denoised_ecg import load_ecg_mat, process_ecg_signal

def process_and_save_ecg_files(X_paths, save_dir="processed_ecg_v2", 
                               fs=500, target_length=5000, 
                               mode="raw", # <--- THÊM Ở ĐÂY
                               verbose=True):
    """
    Tiền xử lý toàn bộ ECG trong danh sách X_paths theo pipeline chuẩn và lưu ra .npy.

    Args:
        X_paths (list[str]): Danh sách đường dẫn file gốc (có thể là .hea hoặc .mat).
        save_dir (str): Thư mục lưu file npy sau xử lý.
        fs (int): Tần số lấy mẫu (mặc định 500Hz).
        target_length (int): Độ dài cố định của tín hiệu (mặc định 5000 samples = 10s).
        verbose (bool): In log tiến độ nếu True.
        
    Returns:
        valid_indices (list[int]): Danh sách index của các file xử lý thành công.
    """
    os.makedirs(save_dir, exist_ok=True)
    valid_indices = []
    
    # Mapping tên file gốc sang tên file mới để tránh trùng lặp nếu cần
    # Ở đây dùng logic: JS00001.mat -> JS00001.npy
    
    total_files = len(X_paths)
    if verbose:
        print(f"🚀 Bắt đầu xử lý {total_files} bản ghi...")
        print(f"   Config: FS={fs}, Target Length={target_length}, Save Dir={save_dir}")

    for i, path in enumerate(X_paths):
        # Xử lý đường dẫn file: Đảm bảo trỏ tới file .mat
        base, ext = os.path.splitext(path)
        mat_path = base + ".mat"
        file_name = os.path.basename(base) # Ví dụ: JS00001

        if not os.path.exists(mat_path):
            if verbose:
                print(f"[Bỏ qua] Không tìm thấy file .mat: {mat_path}")
            continue

        try:
            # 1️⃣ Load dữ liệu thô
            raw_signals = load_ecg_mat(mat_path)  # shape (12, N)
            
            if raw_signals is None:
                if verbose: print(f"[Lỗi] Không đọc được dữ liệu từ {mat_path}")
                continue

            # 2️⃣ Pipeline xử lý: Lọc nhiễu -> Fix Length -> Z-score
            # Hàm process_ecg_signal đã được update trong denoised_ecg.py
            processed_sig = process_ecg_signal(
                raw_signals, 
                fs=fs, 
                #use_zscore=True, 
                mode=mode, # <--- TRUYỀN VÀO ĐÂY
                target_len=target_length
            )

            # 3️⃣ Kiểm tra NaN/Inf (Safety check)
            if np.isnan(processed_sig).any() or np.isinf(processed_sig).any():
                if verbose:
                    print(f"[Cảnh báo] Phát hiện NaN/Inf tại record {file_name} (index {i}) -> Bỏ qua.")
                continue

            # 4️⃣ Giới hạn giá trị (Clip) để loại bỏ outlier cực đoan
            # Z-score thường nằm trong khoảng -3 đến 3, clip -10 đến 10 là an toàn
            processed_sig = np.clip(processed_sig, -10, 10)

            # 5️⃣ Lưu file numpy
            # Tên file output giữ nguyên tên gốc cho dễ trace: JS00001.npy
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

    # Lưu danh sách indices hợp lệ vào cùng thư mục để tiện load labels tương ứng sau này
    indices_path = os.path.join(os.path.dirname(save_dir), "valid_indices.npy")
    np.save(indices_path, np.array(valid_indices))
    
    if verbose:
        print(f"✅ Hoàn tất! Đã lưu {len(valid_indices)}/{total_files} bản ghi hợp lệ vào '{save_dir}'.")
        print(f"   Danh sách valid_indices đã lưu tại: {indices_path}")
        
    return valid_indices


# update 070126 + add functions
from processing.hand_crafted_features import ECGFeatureExtractor

def extract_and_save_hand_features(
    processed_dir,
    save_dir,
    fs=500,
    verbose=True
):
    """
    Extract hand-crafted features từ processed ECG signals.
    
    Args:
        processed_dir: Thư mục chứa .npy files (output từ process_and_save_ecg_files)
        save_dir: Thư mục lưu hand features
        fs: Sampling frequency
        verbose: Print progress
    """
    import os
    import numpy as np
    from glob import glob
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Get all .npy files
    npy_files = sorted(glob(os.path.join(processed_dir, "*.npy")))
    
    if verbose:
        print(f"🚀 Extracting hand-crafted features from {len(npy_files)} files...")
        print(f"   Input dir: {processed_dir}")
        print(f"   Output dir: {save_dir}")
    
    extractor = ECGFeatureExtractor(fs=fs)
    
    for i, npy_file in enumerate(npy_files):
        if verbose and (i+1) % 100 == 0:
            print(f"✓ Extracted {i+1}/{len(npy_files)} features")
        
        try:
            # Load signal
            signal = np.load(npy_file)
            
            # Extract features
            features = extractor.extract_all_features(signal)
            feat_array = extractor.features_to_array(features)
            
            # Save với cùng tên file
            base_name = os.path.basename(npy_file)
            save_path = os.path.join(save_dir, base_name)
            np.save(save_path, feat_array)
            
        except Exception as e:
            print(f"❌ Error processing {npy_file}: {e}")
            continue
    
    if verbose:
        print(f"✅ Completed! Saved {len(npy_files)} feature files to {save_dir}")
        
        
# === Ví dụ chạy độc lập (Test) ===
if __name__ == "__main__":
    # Test thử với 1 file giả định
    sample_paths = [
        r"E:\NCKH - 2026\ECG Project\data\raw\WFDBRecords\01\010\JS00001.hea",
    ]
    # process_and_save_ecg_files(sample_paths, save_dir="processed_ecg_test_v2")





# # processing/feature_extraction.py
# """
# Trích xuất và xử lý tín hiệu ECG từ file .mat -> lưu dưới dạng .npy
# Bao gồm:
#  - Đọc danh sách đường dẫn file ECG (.mat)
#  - Lọc nhiễu, chuẩn hóa, kiểm tra NaN/Inf
#  - Lưu thành mảng numpy (12, N)
# """

# import os
# import numpy as np
# from processing.denoised_ecg import load_ecg_mat, normalize_to_mV, denoise_all_leads


# def process_and_save_ecg_files(X_paths, save_dir="processed_ecg_omaigah", fs=500, verbose=True):
#     """
#     Tiền xử lý toàn bộ ECG trong danh sách X_paths và lưu ra .npy.

#     Args:
#         X_paths (list[str]): Danh sách đường dẫn file .hea hoặc .mat gốc.
#         save_dir (str): Thư mục lưu file npy sau xử lý.
#         fs (int): Tần số lấy mẫu.
#         verbose (bool): In log chi tiết nếu True.
#     """
#     os.makedirs(save_dir, exist_ok=True)
#     valid_indices = []

#     for i, path in enumerate(X_paths):
#         base, ext = os.path.splitext(path)
#         mat_path = base + ".mat"

#         if not os.path.exists(mat_path):
#             if verbose:
#                 print(f"[Bỏ qua] Không tìm thấy file .mat: {mat_path}")
#             continue

#         try:
#             # 1️⃣ Load và chuyển sang mV
#             signals = load_ecg_mat(mat_path)  # shape (12, N)
#             raw_mV = normalize_to_mV(signals, gain=1000.0)

#             # 2️⃣ Lọc nhiễu
#             denoised = denoise_all_leads(raw_mV, fs)

#             # 3️⃣ Kiểm tra NaN/Inf
#             if np.isnan(denoised).any() or np.isinf(denoised).any():
#                 if verbose:
#                     print(f"[Cảnh báo] NaN/Inf tại record {i}")
#                 continue

#             # 4️⃣ Chuẩn hóa z-score từng lead
#             sig_z = (denoised - denoised.mean(axis=1, keepdims=True)) / (
#                 denoised.std(axis=1, keepdims=True) + 1e-8
#             )

#             # 5️⃣ Giới hạn giá trị để tránh outlier
#             sig_z = np.clip(sig_z, -10, 10)

#             '''# 6️⃣ Kiểm tra lại NaN/Inf
#             if np.isnan(sig_z).any() or np.isinf(sig_z).any():
#                 if verbose:
#                     print(f"[Cảnh báo] NaN/Inf sau chuẩn hóa tại record {i}")
#                 continue'''
            
#             # Chọn lead muốn lưu (ví dụ lead I = 0)
#             # Chọn các lead V1–V6
#             lead_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
#             six_leads = sig_z[lead_indices, :]   # shape (6, N)

#             # 6️⃣ Kiểm tra lại NaN/Inf cho lead đã chọn
#             if np.isnan(six_leads).any() or np.isinf(six_leads).any():
#                 if verbose:
#                     print(f"[Cảnh báo] NaN/Inf sau chuẩn hóa tại record {i}")
#                 continue

#             # 7️⃣ Lưu file numpy
#             np.save(os.path.join(save_dir, f"record_{i}.npy"), six_leads)
#             valid_indices.append(i)

#             if verbose and (i + 1) % 100 == 0:
#                 print(f"✓ Đã xử lý {i+1}/{len(X_paths)} file")

#         except Exception as e:
#             if verbose:
#                 print(f"[Lỗi] record {i}: {e}")
#             continue

#     print(f"✅ Hoàn tất! Lưu {len(valid_indices)} bản ghi hợp lệ vào '{save_dir}'.")
#     return valid_indices


# # === Ví dụ chạy độc lập ===
# if __name__ == "__main__":
#     # Ví dụ giả định: bạn đã có danh sách file ECG
#     sample_paths = [
#         r"E:\NCKH - 2026\ECG Project\data\raw\WFDBRecords\01\010\JS00001.hea",
#     ]
#     process_and_save_ecg_files(sample_paths, save_dir="processed_ecg_omaigah")
