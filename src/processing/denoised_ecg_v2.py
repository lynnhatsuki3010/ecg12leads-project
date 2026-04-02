# processing/denoised_ecg_v2.py
"""
PHIÊN BẢN CẢI TIẾN: Sử dụng Constant Scaling thay vì Z-score
Mục đích: Giữ nguyên tỷ lệ biên độ cho các bệnh lý như LVH
"""
import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt, iirnotch

# --- 1. Load dữ liệu .mat ---
def load_ecg_mat(mat_file):
    """Đọc file .mat chứa ECG, trả về signals shape (12, N)"""
    try:
        mat = scipy.io.loadmat(mat_file)
        if 'val' in mat:
            signals = mat['val']
        elif 'data' in mat: 
            signals = mat['data']
        else:
            raise ValueError(f"Không tìm thấy key 'val' hoặc 'data' trong {mat_file}")
        return signals.astype(float)
    except Exception as e:
        return None

# --- 2. Bộ lọc thông dải ---
def butter_bandpass(lowcut, highcut, fs, order=4):
    ny = 0.5 * fs
    b, a = butter(order, [lowcut / ny, highcut / ny], btype='band')
    return b, a

def apply_bandpass(sig, fs, lowcut=0.5, highcut=50.0, order=4):
    """Lọc Bandpass 0.5 - 50Hz"""
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, sig)

# --- 3. Bộ lọc Notch ---
def apply_notch(sig, fs, notch_freq=50.0, quality=30.0):
    b, a = iirnotch(notch_freq / (fs / 2), quality)
    return filtfilt(b, a, sig)

# --- 4a. Z-score Normalization (CŨ - KHÔNG KHUYẾN KHÍCH CHO LVH) ---
def z_score_normalize(signal):
    """
    ⚠️ CẢNH BÁO: Phương pháp này phá hủy thông tin biên độ tuyệt đối
    Không nên dùng cho các bệnh lý phụ thuộc biên độ như LVH
    """
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-6:
        return np.zeros_like(signal)
    return (signal - mean) / std

# --- 4b. Constant Scaling (MỚI - KHUYẾN NGHỊ) ---
def constant_scale_normalize(signal, gain=1000.0):
    """
    ✅ Chuẩn hóa GIỮ NGUYÊN tỷ lệ biên độ (quan trọng cho LVH, RBBB, LBBB...)
    
    Pipeline:
    1. Chia cho gain (1000) để đưa về đơn vị mV
    2. Chia cho 5.0 để đưa giá trị về khoảng [-1, 1] phù hợp Deep Learning
    
    Ví dụ so sánh:
    ┌─────────────────┬──────────────┬─────────────────┐
    │ Trường hợp      │ Z-score      │ Constant Scale  │
    ├─────────────────┼──────────────┼─────────────────┤
    │ Bình thường (1mV)│ ~0.0 ± 1.0   │ 0.2            │
    │ LVH (3mV)       │ ~0.0 ± 1.0   │ 0.6 (3x)       │
    └─────────────────┴──────────────┴─────────────────┘
    
    Với Z-score: Cả 2 trường hợp đều có std ≈ 1.0 → Mất thông tin biên độ
    Với Constant Scale: Tỷ lệ 3:1 được giữ nguyên → Model nhận biết được LVH
    """
    sig_mv = signal / gain  # ADC → mV
    return sig_mv / 5.0      # Scale về [-1, 1]

# --- 5. Fix Length ---
def fix_length(signal, target_length=5000):
    """Cắt hoặc pad tín hiệu về độ dài cố định"""
    C, L = signal.shape
    if L > target_length:
        start = (L - target_length) // 2
        return signal[:, start:start+target_length]
    elif L < target_length:
        pad_len = target_length - L
        return np.pad(signal, ((0,0), (0, pad_len)), 'constant')
    else:
        return signal

# --- 6. Pipeline xử lý chính (CẢI TIẾN) ---
def process_ecg_signal(raw_signal, fs=500, normalize_method='constant', target_len=5000):
    """
    Pipeline đầy đủ: Notch -> Bandpass -> Fix Length -> Normalize
    
    Args:
        raw_signal: Tín hiệu thô (12, N)
        fs: Tần số lấy mẫu
        normalize_method: 'constant' (khuyến nghị) hoặc 'zscore' (legacy)
        target_len: Độ dài cố định
        
    Returns:
        processed: Tín hiệu đã xử lý (12, target_len)
    """
    num_leads, num_samples = raw_signal.shape
    processed = np.zeros_like(raw_signal)

    # 1. Lọc nhiễu từng lead
    for i in range(num_leads):
        sig = apply_notch(raw_signal[i, :], fs, notch_freq=50.0)
        sig = apply_bandpass(sig, fs, lowcut=0.5, highcut=50.0)
        processed[i, :] = sig

    # 2. Cắt/Pad về độ dài cố định
    if target_len is not None:
        processed = fix_length(processed, target_length=target_len)

    # 3. Chuẩn hóa
    if normalize_method == 'constant':
        # ✅ KHUYẾN NGHỊ: Giữ nguyên tỷ lệ biên độ
        processed = constant_scale_normalize(processed, gain=1000.0)
    elif normalize_method == 'zscore':
        # ⚠️ Legacy: Chỉ dùng nếu không quan tâm biên độ
        for i in range(num_leads):
            processed[i, :] = z_score_normalize(processed[i, :])
    else:
        raise ValueError(f"normalize_method phải là 'constant' hoặc 'zscore', nhận: {normalize_method}")
            
    return processed

# --- 7. Wrapper function để tương thích code cũ ---
def process_ecg_signal_legacy(raw_signal, fs=500, use_zscore=True, target_len=5000):
    """
    ⚠️ HÀM CŨ - ĐỂ TƯƠNG THÍCH NGƯỢC
    Khuyến nghị: Dùng process_ecg_signal() với normalize_method='constant'
    """
    if use_zscore:
        return process_ecg_signal(raw_signal, fs, 'zscore', target_len)
    else:
        return process_ecg_signal(raw_signal, fs, 'constant', target_len)