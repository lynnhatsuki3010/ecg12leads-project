# solutions/afib_af_separator.py
"""
GIẢI PHÁP: Phân biệt AFIB (Rung nhĩ) và AF (Cuồng nhĩ)

Vấn đề: Model nhầm lẫn AFIB và AF do 2 bệnh có đặc điểm tương tự
Nguyên nhân: Cả 2 đều là rối loạn nhịp ở tâm nhĩ, nhưng khác nhau về:
  - AFIB: Hoàn toàn bất thường, không có sóng P rõ ràng
  - AF: Có sóng F (flutter waves) đều đặn với tần số 250-350 bpm

Giải pháp: Thêm feature engineering để model học được sự khác biệt
"""

import numpy as np
from scipy import signal
from scipy.stats import entropy

class AFIBvAF_FeatureExtractor:
    """
    Trích xuất features để phân biệt AFIB và AF
    """
    def __init__(self, fs=500):
        self.fs = fs
        
    def compute_rr_variability(self, ecg_signal):
        """
        Tính độ biến thiên RR interval
        AFIB: RR interval hoàn toàn bất thường (entropy cao)
        AF: RR interval có pattern (entropy thấp hơn)
        """
        # Detect R peaks (đơn giản)
        lead_ii = ecg_signal[1, :]  # Lead II thường rõ nhất
        peaks, _ = signal.find_peaks(lead_ii, height=0.3, distance=self.fs//3)
        
        if len(peaks) < 3:
            return 0, 0
        
        # Tính RR intervals
        rr_intervals = np.diff(peaks) / self.fs * 1000  # Convert to ms
        
        # Độ biến thiên (SDNN - Standard Deviation of NN intervals)
        sdnn = np.std(rr_intervals)
        
        # Entropy (đo độ hỗn loạn)
        rr_entropy = entropy(np.histogram(rr_intervals, bins=20)[0] + 1e-10)
        
        return sdnn, rr_entropy
    
    def compute_f_wave_regularity(self, ecg_signal):
        """
        Phát hiện sóng F (Flutter waves) trong AF
        AF: Có sóng F đều đặn ở 250-350 bpm (4-6 Hz)
        AFIB: Không có pattern rõ ràng
        """
        # Lấy lead V1 (tốt nhất để thấy atrial activity)
        lead_v1 = ecg_signal[6, :]  # V1 là lead thứ 7
        
        # FFT để phát hiện dominant frequency
        freqs = np.fft.rfftfreq(len(lead_v1), 1/self.fs)
        fft_vals = np.abs(np.fft.rfft(lead_v1))
        
        # Tìm peak trong dải 4-6 Hz (AF flutter range)
        mask = (freqs >= 4) & (freqs <= 6)
        if mask.sum() == 0:
            return 0
        
        flutter_power = np.max(fft_vals[mask])
        total_power = np.sum(fft_vals)
        
        # Tỷ lệ power trong dải flutter
        flutter_ratio = flutter_power / (total_power + 1e-10)
        
        return flutter_ratio
    
    def compute_p_wave_presence(self, ecg_signal):
        """
        Kiểm tra sự hiện diện của sóng P
        AFIB: Không có sóng P rõ ràng
        AF: Có thể có sóng F nhưng không có sóng P bình thường
        """
        # Lấy lead II
        lead_ii = ecg_signal[1, :]
        
        # Detect R peaks
        peaks, _ = signal.find_peaks(lead_ii, height=0.3, distance=self.fs//3)
        
        if len(peaks) < 2:
            return 0
        
        # Kiểm tra vùng trước R peak (nơi sóng P xuất hiện)
        p_wave_scores = []
        for peak in peaks[1:]:  # Bỏ qua peak đầu
            start = max(0, peak - int(0.2 * self.fs))  # 200ms trước R
            end = peak - int(0.05 * self.fs)            # 50ms trước R
            
            if start >= end:
                continue
                
            segment = lead_ii[start:end]
            
            # Sóng P có biên độ nhỏ, tính score dựa trên variance
            p_score = np.std(segment)
            p_wave_scores.append(p_score)
        
        return np.mean(p_wave_scores) if p_wave_scores else 0
    
    def extract_all_features(self, ecg_signal):
        """
        Trích xuất tất cả features
        Returns:
            dict: Dictionary chứa các features
        """
        sdnn, rr_entropy = self.compute_rr_variability(ecg_signal)
        flutter_ratio = self.compute_f_wave_regularity(ecg_signal)
        p_wave_score = self.compute_p_wave_presence(ecg_signal)
        
        features = {
            'sdnn': sdnn,                    # Cao → AFIB
            'rr_entropy': rr_entropy,        # Cao → AFIB
            'flutter_ratio': flutter_ratio,  # Cao → AF
            'p_wave_score': p_wave_score     # Cao → Bình thường, Thấp → AFIB/AF
        }
        
        # Tính composite score
        afib_score = (sdnn * 0.4 + rr_entropy * 0.4 - flutter_ratio * 0.2)
        af_score = (flutter_ratio * 0.6 - sdnn * 0.2 - rr_entropy * 0.2)
        
        features['afib_score'] = afib_score
        features['af_score'] = af_score
        
        return features


# === CÁCH TÍCH HỢP VÀO MODEL ===
"""
Có 2 cách sử dụng:

CÁCH 1: Feature Enhancement (Thêm features vào input của model)
---------------------------------------------------------------
from afib_af_separator import AFIBvAF_FeatureExtractor

# Trong Dataset __getitem__
extractor = AFIBvAF_FeatureExtractor(fs=500)
features = extractor.extract_all_features(signal)

# Concatenate features vào signal
extra_features = np.array([
    features['sdnn'], 
    features['rr_entropy'], 
    features['flutter_ratio']
])
# Thêm vào model như một embedding layer riêng


CÁCH 2: Post-processing (Sửa predictions dựa trên features)
------------------------------------------------------------
# Sau khi model predict
for i, signal in enumerate(test_signals):
    features = extractor.extract_all_features(signal)
    
    # Nếu model dự đoán AFIB nhưng có flutter_ratio cao → Sửa thành AF
    if predictions[i, AFIB_IDX] > 0.5 and features['flutter_ratio'] > 0.3:
        predictions[i, AFIB_IDX] = 0.3
        predictions[i, AF_IDX] = 0.7
        
    # Nếu model dự đoán AF nhưng RR entropy cao → Sửa thành AFIB
    if predictions[i, AF_IDX] > 0.5 and features['rr_entropy'] > 2.5:
        predictions[i, AF_IDX] = 0.3
        predictions[i, AFIB_IDX] = 0.7
"""


# === VÍ DỤ SỬ DỤNG ===
if __name__ == "__main__":
    # Test với signal giả định
    test_signal = np.random.randn(12, 5000)
    
    extractor = AFIBvAF_FeatureExtractor(fs=500)
    features = extractor.extract_all_features(test_signal)
    
    print("📊 Features trích xuất:")
    for key, value in features.items():
        print(f"   {key}: {value:.4f}")
    
    print("\n🔍 Phân tích:")
    if features['afib_score'] > features['af_score']:
        print("   → Khả năng cao là AFIB (Rung nhĩ)")
    else:
        print("   → Khả năng cao là AF (Cuồng nhĩ)")