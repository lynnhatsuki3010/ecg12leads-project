# processing/feature_extraction_v2.py
"""
PHIÊN BẢN CẢI TIẾN: Sử dụng Constant Scaling cho LVH
"""
import os
import numpy as np
from processing.denoised_ecg_v2 import load_ecg_mat, process_ecg_signal

def process_and_save_ecg_files(X_paths, save_dir="processed_ecg_v3", 
                               fs=500, target_length=5000, 
                               normalize_method='constant',
                               verbose=True):
    """
    Tiền xử lý toàn bộ ECG với Constant Scaling (giữ nguyên biên độ cho LVH)
    
    Args:
        X_paths: Danh sách đường dẫn file gốc
        save_dir: Thư mục lưu file npy
        fs: Tần số lấy mẫu (500Hz)
        target_length: Độ dài cố định (5000 samples = 10s)
        normalize_method: 'constant' (khuyến nghị cho LVH) hoặc 'zscore'
        verbose: In log
        
    Returns:
        valid_indices: Danh sách index của các file xử lý thành công
    """
    os.makedirs(save_dir, exist_ok=True)
    valid_indices = []
    
    total_files = len(X_paths)
    if verbose:
        print(f"🚀 Bắt đầu xử lý {total_files} bản ghi...")
        print(f"   📊 Config:")
        print(f"      - Sampling rate: {fs}Hz")
        print(f"      - Target length: {target_length} samples")
        print(f"      - Normalize method: {normalize_method}")
        print(f"      - Save directory: {save_dir}")
        
        if normalize_method == 'constant':
            print(f"   ✅ Sử dụng Constant Scaling - GIỮ NGUYÊN TỶ LỆ BIÊN ĐỘ")
            print(f"      → Tốt cho: LVH, RBBB, LBBB, và các bệnh lý về biên độ")
        else:
            print(f"   ⚠️ Sử dụng Z-score - MẤT THÔNG TIN BIÊN ĐỘ TUYỆT ĐỐI")
            print(f"      → Chỉ tốt cho: Các bệnh lý về nhịp (AFIB, AF, ST, SB...)")

    for i, path in enumerate(X_paths):
        base, ext = os.path.splitext(path)
        mat_path = base + ".mat"
        file_name = os.path.basename(base)

        if not os.path.exists(mat_path):
            if verbose:
                print(f"[Bỏ qua] Không tìm thấy file .mat: {mat_path}")
            continue

        try:
            # 1️⃣ Load dữ liệu thô
            raw_signals = load_ecg_mat(mat_path)
            
            if raw_signals is None:
                if verbose: 
                    print(f"[Lỗi] Không đọc được dữ liệu từ {mat_path}")
                continue

            # 2️⃣ Pipeline xử lý với Constant Scaling
            processed_sig = process_ecg_signal(
                raw_signals, 
                fs=fs, 
                normalize_method=normalize_method,
                target_len=target_length
            )

            # 3️⃣ Kiểm tra NaN/Inf
            if np.isnan(processed_sig).any() or np.isinf(processed_sig).any():
                if verbose:
                    print(f"[Cảnh báo] Phát hiện NaN/Inf tại {file_name} (index {i})")
                continue

            # 4️⃣ Clip giá trị
            # Với Constant Scaling: range thường là [-1, 1]
            # Với Z-score: range thường là [-10, 10]
            if normalize_method == 'constant':
                processed_sig = np.clip(processed_sig, -2, 2)  # An toàn hơn cho biên độ
            else:
                processed_sig = np.clip(processed_sig, -10, 10)

            # 5️⃣ Lưu file numpy
            save_path = os.path.join(save_dir, f"{file_name}.npy")
            np.save(save_path, processed_sig)
            
            valid_indices.append(i)

            # Log tiến độ
            if verbose and (i + 1) % 100 == 0:
                print(f"✓ Đã xử lý {i+1}/{total_files} file")

        except Exception as e:
            if verbose:
                print(f"[Ngoại lệ] Lỗi tại {file_name} (index {i}): {str(e)}")
            continue

    # Lưu danh sách indices hợp lệ
    indices_path = os.path.join(os.path.dirname(save_dir), "valid_indices_v3.npy")
    np.save(indices_path, np.array(valid_indices))
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"✅ HOÀN TẤT!")
        print(f"   📁 Đã lưu {len(valid_indices)}/{total_files} bản ghi hợp lệ")
        print(f"   📂 Output directory: {save_dir}")
        print(f"   📋 Valid indices: {indices_path}")
        print(f"{'='*60}\n")
        
    return valid_indices


# === VÍ DỤ SỬ DỤNG ===
if __name__ == "__main__":
    # Test với file mẫu
    sample_paths = [
        r"E:\NCKH - 2026\ECG Project\data\raw\WFDBRecords\01\010\JS00001.hea",
    ]
    
    # Xử lý với Constant Scaling (khuyến nghị)
    print("🔄 Xử lý với CONSTANT SCALING (giữ nguyên biên độ)...")
    process_and_save_ecg_files(
        sample_paths, 
        save_dir="processed_ecg_constant_scaling",
        normalize_method='constant'
    )
    
    # So sánh với Z-score (để thấy sự khác biệt)
    print("\n🔄 Xử lý với Z-SCORE (để so sánh)...")
    process_and_save_ecg_files(
        sample_paths, 
        save_dir="processed_ecg_zscore",
        normalize_method='zscore'
    )