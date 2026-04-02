# processing/denoised_ecg.py
import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt, iirnotch, lfilter

from processing.denoised_ecg_v2 import constant_scale_normalize

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
        # print(f"Lỗi đọc file {mat_file}: {e}") # Có thể comment lại để đỡ spam log
        return None

# --- 2. Bộ lọc thông dải ---
def butter_bandpass(lowcut, highcut, fs, order=4):
    ny = 0.5 * fs
    b, a = butter(order, [lowcut / ny, highcut / ny], btype='band')
    return b, a

# def apply_bandpass(sig, fs, lowcut=0.5, highcut=50.0, order=4):
#     """Lọc Bandpass 0.5 - 50Hz"""
#     b, a = butter_bandpass(lowcut, highcut, fs, order)
#     return filtfilt(b, a, sig)

# --- 3. Bộ lọc Notch ---
def apply_notch(sig, fs, notch_freq=50.0, quality=30.0):
    b, a = iirnotch(notch_freq / (fs / 2), quality)
    return filtfilt(b, a, sig)

# --- 4. Z-score Normalization ---
def z_score_normalize(signal):
    """Chuẩn hóa (x - mean) / std"""
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-6:
        return np.zeros_like(signal)
    return (signal - mean) / std

# --- 5. Fix Length ---
def fix_length(signal, target_length=5000):
    """
    Cắt hoặc pad tín hiệu về độ dài cố định.
    """
    C, L = signal.shape
    if L > target_length:
        start = (L - target_length) // 2
        return signal[:, start:start+target_length]
    elif L < target_length:
        pad_len = target_length - L
        return np.pad(signal, ((0,0), (0, pad_len)), 'constant')
    else:
        return signal

# --- 6. Pipeline xử lý chính ---
# def process_ecg_signal(raw_signal, fs=500, use_zscore=True, target_len=5000):
#     """
#     Pipeline đầy đủ: Notch -> Bandpass -> Fix Length -> Normalize
#     """
#     num_leads, num_samples = raw_signal.shape
#     processed = np.zeros_like(raw_signal)

#     # 1. Lọc nhiễu từng lead
#     for i in range(num_leads):
#         sig = apply_notch(raw_signal[i, :], fs, notch_freq=50.0)
#         sig = apply_bandpass(sig, fs, lowcut=0.5, highcut=50.0)
#         processed[i, :] = sig

#     # 2. Cắt/Pad về độ dài cố định
#     if target_len is not None:
#         # SỬA LỖI Ở DÒNG DƯỚI NÀY: target_length=target_len
#         processed = fix_length(processed, target_length=target_len)

#     # 3. Chuẩn hóa Z-score
#     if use_zscore:
#         for i in range(num_leads):
#             processed[i, :] = z_score_normalize(processed[i, :])
            
#     return processed

def apply_bandpass(sig, fs, lowcut=0.5, highcut=50.0, order=4, zero_phase=True):
    ny = 0.5 * fs
    b, a = butter(order, [lowcut / ny, highcut / ny], btype='band')
    if zero_phase:
        return filtfilt(b, a, sig)
    else:
        return lfilter(b, a, sig) 
def process_ecg_signal(raw_signal, fs=500, mode="raw", target_len=5000):
    num_leads, num_samples = raw_signal.shape
    processed = np.zeros_like(raw_signal)
    
    for i in range(num_leads):
        sig = raw_signal[i, :]
        if mode == "raw": 
            pass # Không lọc
        elif mode == "bandpass_only": 
            sig = apply_bandpass(sig, fs, zero_phase=False)
        elif mode == "bandpass_notch": 
            sig = apply_notch(sig, fs, notch_freq=50.0)
            sig = apply_bandpass(sig, fs, zero_phase=False)
        elif mode == "bandpass_notch_zerophase": 
            sig = apply_notch(sig, fs, notch_freq=50.0)
            sig = apply_bandpass(sig, fs, zero_phase=True)
        processed[i, :] = sig
    # Phải giữ Fix Length và Z-score / Constant Scale để size tensor đúng (12, 5000)
    processed = fix_length(processed, target_length=target_len)
    processed = constant_scale_normalize(processed, gain=1000.0)
    return processed




###################################################3

# # processing/denoised_ecg.py
# import numpy as np
# import scipy.io
# from scipy.signal import butter, filtfilt, iirnotch

# # --- 1. Load dữ liệu .mat ---
# def load_ecg_mat(mat_file):
#     """Đọc file .mat chứa ECG, trả về signals shape (12, N)"""
#     try:
#         mat = scipy.io.loadmat(mat_file)
#         if 'val' in mat:
#             signals = mat['val']
#         elif 'data' in mat: 
#             signals = mat['data']
#         else:
#             raise ValueError(f"Không tìm thấy key 'val' hoặc 'data' trong {mat_file}")
#         return signals.astype(float)
#     except Exception as e:
#         # print(f"Lỗi đọc file {mat_file}: {e}") # Có thể comment lại để đỡ spam log
#         return None

# # --- 2. Bộ lọc thông dải ---
# def butter_bandpass(lowcut, highcut, fs, order=4):
#     ny = 0.5 * fs
#     b, a = butter(order, [lowcut / ny, highcut / ny], btype='band')
#     return b, a

# def apply_bandpass(sig, fs, lowcut=0.5, highcut=50.0, order=4):
#     """Lọc Bandpass 0.5 - 50Hz"""
#     b, a = butter_bandpass(lowcut, highcut, fs, order)
#     return filtfilt(b, a, sig)

# # --- 3. Bộ lọc Notch ---
# def apply_notch(sig, fs, notch_freq=50.0, quality=30.0):
#     b, a = iirnotch(notch_freq / (fs / 2), quality)
#     return filtfilt(b, a, sig)

# # --- 4. Z-score Normalization (việc chuẩn hóa Zscore có khả năng sẽ gây nhiễu cho nhãn LVH) ---
# # def z_score_normalize(signal):
# #     """Chuẩn hóa (x - mean) / std"""
# #     mean = np.mean(signal)
# #     std = np.std(signal)
# #     if std < 1e-6:
# #         return np.zeros_like(signal)
# #     return (signal - mean) / std

# # --- 4. Constant Scaling (Chuẩn hóa giữ biên độ) ---
# def constant_scale_normalize(signal, gain=1000.0):
#     """
#     Chuẩn hóa giữ nguyên tỷ lệ biên độ (quan trọng cho LVH).
    
#     Quy trình:
#     1. Chia cho gain (1000) để đưa về đơn vị mV.
#     2. Chia tiếp cho 5.0 để đưa giá trị về khoảng [-1, 1] (phù hợp cho Deep Learning).
#        (Vì biên độ ECG sinh lý thường không vượt quá 5mV)
#     """
#     # Chuyển sang mV
#     sig_mv = signal / gain
    
#     # Chuẩn hóa về range [-1, 1]
#     # Sóng R bình thường (1mV) -> 0.2
#     # Sóng R LVH (3mV) -> 0.6
#     return sig_mv / 5.0 

# # --- 5. Fix Length (ĐÃ SỬA LỖI TÊN THAM SỐ) ---
# def fix_length(signal, target_length=5000):
#     """
#     Cắt hoặc pad tín hiệu về độ dài cố định.
#     """
#     C, L = signal.shape
#     if L > target_length:
#         start = (L - target_length) // 2
#         return signal[:, start:start+target_length]
#     elif L < target_length:
#         pad_len = target_length - L
#         return np.pad(signal, ((0,0), (0, pad_len)), 'constant')
#     else:
#         return signal

# # --- 6. Pipeline xử lý chính ---
# def process_ecg_signal(raw_signal, fs=500, use_zscore=True, target_len=5000):
#     """
#     Pipeline đầy đủ: Notch -> Bandpass -> Fix Length -> Normalize
#     """
#     num_leads, num_samples = raw_signal.shape
#     processed = np.zeros_like(raw_signal)

#     # 1. Lọc nhiễu từng lead
#     for i in range(num_leads):
#         sig = apply_notch(raw_signal[i, :], fs, notch_freq=50.0)
#         sig = apply_bandpass(sig, fs, lowcut=0.5, highcut=50.0)
#         processed[i, :] = sig

#     # 2. Cắt/Pad về độ dài cố định
#     if target_len is not None:
#         # SỬA LỖI Ở DÒNG DƯỚI NÀY: target_length=target_len
#         processed = fix_length(processed, target_length=target_len)

#    # 3. Chuẩn hóa Constant Scaling (Luôn luôn dùng)
#     # Gain=1000 là chuẩn chung của WFDB/Chapman. 
#     # Nếu dataset của bạn khác (ví dụ gain 200), hãy sửa số này.
#     processed = constant_scale_normalize(processed, gain=1000.0)
            
#     return processed


# # lọc nhiễu ecg
# import numpy as np
# import scipy.io
# from scipy.signal import butter, filtfilt, iirnotch


# # --- 1. Load dữ liệu .mat ---
# def load_ecg_mat(mat_file):
#     """Đọc file .mat chứa ECG, trả về signals shape (12, N)"""
#     mat = scipy.io.loadmat(mat_file)
#     signals = mat['val']  # 12 x N
#     return signals.astype(float)


# # --- 2. Bộ lọc thông dải (0.5–40Hz) ---
# def butter_bandpass(lowcut, highcut, fs, order=4):
#     ny = 0.5 * fs
#     b, a = butter(order, [lowcut / ny, highcut / ny], btype='band')
#     return b, a

# def apply_bandpass(sig, fs, lowcut=0.5, highcut=40.0, order=4):
#     """Lọc giữ lại tần số sinh lý ECG (0.5–40Hz)"""
#     b, a = butter_bandpass(lowcut, highcut, fs, order)
#     return filtfilt(b, a, sig)


# # --- 3. Bộ lọc Notch để loại nhiễu điện lưới 50Hz ---
# def apply_notch(sig, fs, notch_freq=50.0, quality=30.0):
#     """Loại nhiễu điện 50Hz (VN/EU powerline)"""
#     b, a = iirnotch(notch_freq / (fs / 2), quality)
#     return filtfilt(b, a, sig)


# # --- 4. Chuyển đổi sang mV ---
# def normalize_to_mV(sig, gain=1000):
#     """Chuyển đổi từ đơn vị raw ADC sang mV"""
#     return sig / gain


# # --- 5. Lọc nhiễu 1 lead ---
# def denoise_ecg_lead(sig, fs):
#     """Pipeline lọc nhiễu cơ bản cho 1 lead ECG"""
#     x = apply_notch(sig, fs, notch_freq=50.0, quality=30.0)
#     x = apply_bandpass(x, fs, lowcut=0.5, highcut=40.0, order=3)
#     return x


# # --- 6. Áp dụng cho toàn bộ 12 leads ---
# def denoise_all_leads(signals, fs):
#     """Lọc nhiễu toàn bộ 12-leads ECG"""
#     denoised = np.zeros_like(signals, dtype=float)
#     for i in range(signals.shape[0]):
#         denoised[i, :] = denoise_ecg_lead(signals[i, :], fs)
#     return denoised