# lọc nhiễu ecg
import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt, iirnotch


# --- 1. Load dữ liệu .mat ---
def load_ecg_mat(mat_file):
    """Đọc file .mat chứa ECG, trả về signals shape (12, N)"""
    mat = scipy.io.loadmat(mat_file)
    signals = mat['val']  # 12 x N
    return signals.astype(float)


# --- 2. Bộ lọc thông dải (0.5–40Hz) ---
def butter_bandpass(lowcut, highcut, fs, order=4):
    ny = 0.5 * fs
    b, a = butter(order, [lowcut / ny, highcut / ny], btype='band')
    return b, a

def apply_bandpass(sig, fs, lowcut=0.5, highcut=40.0, order=4):
    """Lọc giữ lại tần số sinh lý ECG (0.5–40Hz)"""
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, sig)


# --- 3. Bộ lọc Notch để loại nhiễu điện lưới 50Hz ---
def apply_notch(sig, fs, notch_freq=50.0, quality=30.0):
    """Loại nhiễu điện 50Hz (VN/EU powerline)"""
    b, a = iirnotch(notch_freq / (fs / 2), quality)
    return filtfilt(b, a, sig)


# --- 4. Chuyển đổi sang mV ---
def normalize_to_mV(sig, gain=1000):
    """Chuyển đổi từ đơn vị raw ADC sang mV"""
    return sig / gain


# --- 5. Lọc nhiễu 1 lead ---
def denoise_ecg_lead(sig, fs):
    """Pipeline lọc nhiễu cơ bản cho 1 lead ECG"""
    x = apply_notch(sig, fs, notch_freq=50.0, quality=30.0)
    x = apply_bandpass(x, fs, lowcut=0.5, highcut=40.0, order=3)
    return x


# --- 6. Áp dụng cho toàn bộ 12 leads ---
def denoise_all_leads(signals, fs):
    """Lọc nhiễu toàn bộ 12-leads ECG"""
    denoised = np.zeros_like(signals, dtype=float)
    for i in range(signals.shape[0]):
        denoised[i, :] = denoise_ecg_lead(signals[i, :], fs)
    return denoised