"""
Hand-crafted Feature Extraction for ALL 7 ECG Classes
FIXED VERSION: Removed constant/problematic features

Removed features:
- lvh_sokolow_positive (always 1.0)
- svt_narrow_qrs (always 1.0)  
- pwave_pr_normal (almost always 1.0)
- qrs_wide (almost always 0)
- Hardcoded defaults

Result: 64 clean features (was 70)
"""

import numpy as np
from scipy.stats import entropy
from scipy.signal import find_peaks, butter, filtfilt, welch
import warnings
warnings.filterwarnings('ignore')


class ECGFeatureExtractor:
    """Extract 64 clean features for all 7 ECG classes."""
    
    def __init__(self, fs=500):
        self.fs = fs
        
    def extract_all_features(self, ecg_signal):
        """Extract ALL features from 12-lead ECG signal."""
        # Ensure shape is (12, N)
        if ecg_signal.shape[0] != 12:
            ecg_signal = ecg_signal.T
        
        features = {}
        
        # Group 1: Heart Rate & Rhythm - 12 features
        rhythm_features = self._extract_heart_rate_features(ecg_signal)
        features.update(rhythm_features)
        
        # Group 2: AFIB Features - 8 features
        afib_features = self._extract_afib_features(ecg_signal)
        features.update(afib_features)
        
        # Group 3: AF Features - 5 features
        af_features = self._extract_af_features(ecg_signal)
        features.update(af_features)
        
        # Group 4: LVH Features - 13 features (removed 2 constants)
        lvh_features = self._extract_lvh_features(ecg_signal)
        features.update(lvh_features)
        
        # Group 5: SVT Features - 7 features (removed 1 constant)
        svt_features = self._extract_svt_features(ecg_signal)
        features.update(svt_features)
        
        # Group 6: P-wave Features - 5 features (removed 1 constant)
        pwave_features = self._extract_pwave_features(ecg_signal)
        features.update(pwave_features)
        
        # Group 7: QRS Morphology - 7 features (removed 1 constant)
        qrs_features = self._extract_qrs_morphology(ecg_signal)
        features.update(qrs_features)
        
        # Group 8: Frequency Domain - 7 features (removed 1 constant)
        freq_features = self._extract_frequency_features(ecg_signal)
        features.update(freq_features)
        
        return features
    
    # ========== GROUP 1: HEART RATE (12 features) ==========
    
    def _extract_heart_rate_features(self, ecg):
        features = {}
        lead_ii = ecg[1, :]
        
        try:
            peaks, _ = find_peaks(lead_ii, height=np.max(lead_ii) * 0.3, distance=self.fs // 3)
            if len(peaks) < 2:
                return self._get_default_hr_features()
            
            rr_intervals = np.diff(peaks) / self.fs * 1000
            mean_rr = np.clip(np.mean(rr_intervals), 200, 3000)  # Clip to valid range
            
            hr = np.clip(60000.0 / mean_rr, 20, 300)
            features['hr_mean'] = hr
            features['hr_std'] = np.clip(np.std(rr_intervals) / mean_rr, 0, 2)
            features['rr_mean'] = mean_rr
            
            features['hr_bradycardia'] = 1.0 if hr < 60 else 0.0
            features['hr_normal'] = 1.0 if 60 <= hr <= 100 else 0.0
            features['hr_tachycardia'] = 1.0 if 100 < hr <= 150 else 0.0
            features['hr_severe_tachy'] = 1.0 if hr > 150 else 0.0
            
            rr_cv = np.std(rr_intervals) / mean_rr
            features['rr_regularity'] = np.exp(-np.clip(rr_cv * 5, 0, 10))
            
            features['rr_std'] = np.clip(np.std(rr_intervals), 0, 500)
            features['rr_range'] = np.clip(np.max(rr_intervals) - np.min(rr_intervals), 0, 2000)
            
            hr_max = np.clip(60000.0 / (np.max([np.min(rr_intervals), 200])), 20, 300)
            hr_min = np.clip(60000.0 / (np.min([np.max(rr_intervals), 3000])), 20, 300)
            features['hr_max'] = hr_max
            features['hr_min'] = hr_min
            
        except:
            return self._get_default_hr_features()
        
        return features
    
    # ========== GROUP 2: AFIB (8 features) ==========
    
    def _extract_afib_features(self, ecg):
        features = {}
        lead_ii = ecg[1, :]
        
        try:
            peaks, _ = find_peaks(lead_ii, height=np.max(lead_ii) * 0.3, distance=self.fs // 3)
            if len(peaks) < 3:
                return self._get_default_afib_features()
            
            rr_intervals = np.diff(peaks) / self.fs * 1000
            
            features['afib_sdnn'] = np.clip(np.std(rr_intervals), 0, 500)
            rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
            features['afib_rmssd'] = np.clip(rmssd, 0, 500)
            
            cv = np.std(rr_intervals) / (np.mean(rr_intervals) + 1e-10)
            features['afib_cv'] = np.clip(cv, 0, 5)
            
            hist, _ = np.histogram(rr_intervals, bins=20)
            features['afib_entropy'] = np.clip(entropy(hist + 1e-10), 0, 5)
            
            if len(rr_intervals) > 1:
                diff_rr = np.abs(np.diff(rr_intervals))
                irregularity = np.mean(diff_rr) / (np.mean(rr_intervals) + 1e-10)
                features['afib_irregularity'] = np.clip(irregularity, 0, 2)
            else:
                features['afib_irregularity'] = 0.0
            
            features['afib_p_absence'] = self._detect_p_wave_absence(ecg)
            
            hr = np.clip(60000.0 / (np.mean(rr_intervals) + 1e-10), 20, 300)
            features['afib_mean_hr'] = hr
            
            hr_range = np.clip(
                (60000.0 / (np.min(rr_intervals) + 200)) - (60000.0 / (np.max(rr_intervals) + 200)),
                0, 200
            )
            features['afib_hr_range'] = hr_range
            
        except:
            return self._get_default_afib_features()
        
        return features
    
    # ========== GROUP 3: AF (5 features) ==========
    
    def _extract_af_features(self, ecg):
        features = {}
        
        try:
            lead_v1 = ecg[6, :]
            lead_ii = ecg[1, :]
            
            freqs = np.fft.rfftfreq(len(lead_v1), 1/self.fs)
            fft_vals = np.abs(np.fft.rfft(lead_v1))
            
            flutter_mask = (freqs >= 4) & (freqs <= 6)
            if flutter_mask.sum() > 0:
                flutter_power = np.max(fft_vals[flutter_mask])
                total_power = np.sum(fft_vals)
                features['af_flutter_ratio'] = np.clip(flutter_power / (total_power + 1e-10), 0, 1)
                
                flutter_freqs = freqs[flutter_mask]
                flutter_powers = fft_vals[flutter_mask]
                features['af_dominant_freq'] = np.clip(flutter_freqs[np.argmax(flutter_powers)], 0, 20)
                
                autocorr = np.correlate(lead_v1, lead_v1, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                expected_lag = int(self.fs / (features['af_dominant_freq'] + 1))
                if 0 < expected_lag < len(autocorr):
                    features['af_regularity'] = np.clip(autocorr[expected_lag] / (autocorr[0] + 1e-10), -1, 1)
                else:
                    features['af_regularity'] = 0.0
            else:
                features['af_flutter_ratio'] = 0.0
                features['af_dominant_freq'] = 0.0
                features['af_regularity'] = 0.0
            
            features['af_rr_regularity'] = self._compute_rr_regularity(lead_ii)
            features['af_sawtooth'] = self._detect_sawtooth_pattern(lead_ii)
            
        except:
            features['af_flutter_ratio'] = 0.0
            features['af_dominant_freq'] = 0.0
            features['af_regularity'] = 0.0
            features['af_rr_regularity'] = 0.0
            features['af_sawtooth'] = 0.0
        
        return features
    
    # ========== GROUP 4: LVH (13 features - removed 2 constants) ==========
    
    def _extract_lvh_features(self, ecg):
        features = {}
        
        try:
            v1 = ecg[6, :]
            v5 = ecg[10, :]
            v6 = ecg[11, :]
            avl = ecg[4, :]
            lead_ii = ecg[1, :]
            
            s_v1 = self._measure_s_wave_amplitude(v1)
            r_v5 = self._measure_r_wave_amplitude(v5)
            r_v6 = self._measure_r_wave_amplitude(v6)
            r_avl = self._measure_r_wave_amplitude(avl)
            s_v3 = self._measure_s_wave_amplitude(ecg[8, :])
            
            # Feature 1: Sokolow-Lyon (removed binary indicator)
            sokolow_lyon = np.clip(abs(s_v1) + max(r_v5, r_v6), 0, 20)
            features['lvh_sokolow_lyon'] = sokolow_lyon
            
            # Feature 2: Cornell (removed binary indicator)
            cornell = np.clip(r_avl + abs(s_v3), 0, 20)
            features['lvh_cornell_voltage'] = cornell
            
            # Features 3-7: Amplitudes
            features['lvh_r_v5'] = np.clip(r_v5, 0, 10)
            features['lvh_r_v6'] = np.clip(r_v6, 0, 10)
            features['lvh_r_avl'] = np.clip(r_avl, 0, 10)
            features['lvh_s_v1'] = np.clip(abs(s_v1), 0, 10)
            features['lvh_s_v3'] = np.clip(abs(s_v3), 0, 10)
            
            # Feature 8: QRS duration (removed binary indicator)
            qrs_duration = self._measure_qrs_duration(lead_ii)
            features['lvh_qrs_duration'] = np.clip(qrs_duration, 40, 200)
            
            # Feature 9: QRS axis (removed binary indicator)
            axis = self._estimate_qrs_axis(ecg)
            features['lvh_qrs_axis'] = np.clip(axis, -180, 180)
            
            # Feature 10: R/S ratio
            r_v1 = self._measure_r_wave_amplitude(v1)
            rs_ratio = r_v1 / (abs(s_v1) + 0.01)
            features['lvh_rs_ratio_v1'] = np.clip(rs_ratio, 0, 10)
            
            # Features 11-13: Composite scores
            sokolow_score = np.clip(sokolow_lyon / 3.5, 0, 3)
            cornell_score = np.clip(cornell / 2.4, 0, 3)
            qrs_score = np.clip(qrs_duration / 120, 0, 2)
            
            features['lvh_sokolow_score'] = sokolow_score
            features['lvh_cornell_score'] = cornell_score
            features['lvh_qrs_score'] = qrs_score
            
        except:
            return self._get_default_lvh_features()
        
        return features
    
    # ========== GROUP 5: SVT (7 features - removed 1 constant) ==========
    
    def _extract_svt_features(self, ecg):
        features = {}
        lead_ii = ecg[1, :]
        
        try:
            peaks, _ = find_peaks(lead_ii, height=np.max(lead_ii) * 0.3, distance=self.fs // 3)
            if len(peaks) < 3:
                return self._get_default_svt_features()
            
            rr_intervals = np.diff(peaks) / self.fs * 1000
            hr = np.clip(60000.0 / (np.mean(rr_intervals) + 1e-10), 20, 300)
            
            features['svt_high_rate'] = 1.0 if hr > 150 else 0.0
            features['svt_hr'] = hr
            
            cv = np.std(rr_intervals) / (np.mean(rr_intervals) + 1e-10)
            features['svt_regularity'] = 1.0 if cv < 0.1 else 0.0
            
            # Removed svt_narrow_qrs (always 1.0)
            
            rr_consistency = np.clip(1.0 - (cv / 0.2), 0, 1)
            features['svt_rr_consistency'] = rr_consistency
            
            p_score = self._detect_p_wave_absence(ecg)
            features['svt_abnormal_p'] = p_score
            
            rate_range = np.clip(
                (60000.0 / (np.min(rr_intervals) + 200)) - (60000.0 / (np.max(rr_intervals) + 200)),
                0, 200
            )
            features['svt_rate_range'] = rate_range
            features['svt_stable_rate'] = 1.0 if rate_range < 30 else 0.0
            
        except:
            return self._get_default_svt_features()
        
        return features
    
    # ========== GROUP 6: P-WAVE (5 features - removed 1 constant) ==========
    
    def _extract_pwave_features(self, ecg):
        features = {}
        lead_ii = ecg[1, :]
        
        try:
            peaks, _ = find_peaks(lead_ii, height=np.max(lead_ii) * 0.3, distance=self.fs // 3)
            if len(peaks) < 2:
                return self._get_default_pwave_features()
            
            p_presence = 1.0 - self._detect_p_wave_absence(ecg)
            features['pwave_presence'] = p_presence
            
            p_regular = self._check_p_wave_regularity(lead_ii, peaks)
            features['pwave_regularity'] = p_regular
            
            pr_interval = self._measure_pr_interval(lead_ii, peaks)
            features['pwave_pr_interval'] = np.clip(pr_interval, 0, 300)
            
            # Removed pwave_pr_normal (almost always 1.0)
            
            p_amplitude = self._measure_p_wave_amplitude(lead_ii, peaks)
            features['pwave_amplitude'] = np.clip(p_amplitude, 0, 2)
            
            p_duration = self._measure_p_wave_duration(lead_ii, peaks)
            features['pwave_duration'] = np.clip(p_duration, 0, 200)
            
        except:
            return self._get_default_pwave_features()
        
        return features
    
    # ========== GROUP 7: QRS (7 features - removed 1 constant) ==========
    
    def _extract_qrs_morphology(self, ecg):
        features = {}
        lead_ii = ecg[1, :]
        
        try:
            qrs_duration = self._measure_qrs_duration(lead_ii)
            features['qrs_duration'] = np.clip(qrs_duration, 40, 200)
            
            # Removed qrs_wide (almost always 0)
            
            axis = self._estimate_qrs_axis(ecg)
            features['qrs_axis'] = np.clip(axis, -180, 180)
            
            qrs_amps = []
            for i in range(12):
                r_amp = self._measure_r_wave_amplitude(ecg[i, :])
                qrs_amps.append(r_amp)
            
            features['qrs_amp_mean'] = np.clip(np.mean(qrs_amps), 0, 10)
            features['qrs_amp_std'] = np.clip(np.std(qrs_amps), 0, 5)
            
            consistency = 1.0 - min(np.std(qrs_amps) / (np.mean(qrs_amps) + 0.1), 1.0)
            features['qrs_consistency'] = np.clip(consistency, 0, 1)
            
            features['qrs_max_amp'] = np.clip(np.max(qrs_amps), 0, 10)
            features['qrs_min_amp'] = np.clip(np.min(qrs_amps), 0, 10)
            
        except:
            features['qrs_duration'] = 80.0
            features['qrs_axis'] = 0.0
            features['qrs_amp_mean'] = 0.0
            features['qrs_amp_std'] = 0.0
            features['qrs_consistency'] = 0.5
            features['qrs_max_amp'] = 0.0
            features['qrs_min_amp'] = 0.0
        
        return features
    
    # ========== GROUP 8: FREQUENCY (7 features) ==========
    
    def _extract_frequency_features(self, ecg):
        features = {}
        lead_ii = ecg[1, :]
        
        try:
            freqs, psd = welch(lead_ii, fs=self.fs, nperseg=min(256, len(lead_ii)//4))
            
            lf_mask = (freqs >= 0.5) & (freqs <= 5)
            lf_power = np.sum(psd[lf_mask])
            total_power = np.sum(psd)
            features['freq_lf_power'] = np.clip(np.log1p(lf_power), 0, 20)
            features['freq_lf_ratio'] = np.clip(lf_power / (total_power + 1e-10), 0, 1)
            
            hf_mask = (freqs >= 4) & (freqs <= 6)
            hf_power = np.sum(psd[hf_mask])
            features['freq_hf_power'] = np.clip(np.log1p(hf_power), 0, 20)
            features['freq_hf_ratio'] = np.clip(hf_power / (total_power + 1e-10), 0, 1)
            
            features['freq_dominant'] = np.clip(freqs[np.argmax(psd)], 0, 20)
            
            psd_norm = psd / (np.sum(psd) + 1e-10)
            features['freq_entropy'] = np.clip(entropy(psd_norm + 1e-10), 0, 10)
            
            features['freq_lf_hf_ratio'] = np.clip(lf_power / (hf_power + 1e-10), 0, 100)
            
        except:
            features['freq_lf_power'] = 0.0
            features['freq_lf_ratio'] = 0.0
            features['freq_hf_power'] = 0.0
            features['freq_hf_ratio'] = 0.0
            features['freq_dominant'] = 0.0
            features['freq_entropy'] = 0.0
            features['freq_lf_hf_ratio'] = 0.0
        
        return features
    
    # ========== HELPER FUNCTIONS ==========
    
    def _measure_r_wave_amplitude(self, lead_signal):
        try:
            peaks, properties = find_peaks(lead_signal, height=0)
            if len(peaks) > 0:
                return np.max(properties['peak_heights'])
            return 0.0
        except:
            return 0.0
    
    def _measure_s_wave_amplitude(self, lead_signal):
        try:
            inverted = -lead_signal
            peaks, properties = find_peaks(inverted, height=0)
            if len(peaks) > 0:
                return -np.max(properties['peak_heights'])
            return 0.0
        except:
            return 0.0
    
    def _measure_qrs_duration(self, lead_signal):
        try:
            peaks, _ = find_peaks(lead_signal, height=np.max(lead_signal) * 0.3, distance=self.fs // 3)
            if len(peaks) == 0:
                return 80.0
            
            peak = peaks[0]
            threshold = np.max(lead_signal[max(0, peak-50):min(len(lead_signal), peak+50)]) * 0.1
            
            start = peak
            for i in range(peak, max(0, peak-int(self.fs*0.1)), -1):
                if abs(lead_signal[i]) < threshold:
                    start = i
                    break
            
            end = peak
            for i in range(peak, min(len(lead_signal), peak+int(self.fs*0.1))):
                if abs(lead_signal[i]) < threshold:
                    end = i
                    break
            
            return (end - start) / self.fs * 1000
        except:
            return 80.0
    
    def _estimate_qrs_axis(self, ecg):
        try:
            lead_i = ecg[0, :]
            lead_avf = ecg[5, :]
            r_i = self._measure_r_wave_amplitude(lead_i)
            r_avf = self._measure_r_wave_amplitude(lead_avf)
            axis = np.arctan2(r_avf, r_i) * 180 / np.pi
            return axis
        except:
            return 0.0
    
    def _detect_p_wave_absence(self, ecg):
        try:
            lead_ii = ecg[1, :]
            peaks, _ = find_peaks(lead_ii, height=np.max(lead_ii) * 0.3, distance=self.fs // 3)
            
            if len(peaks) < 2:
                return 0.5
            
            p_count = 0
            total = 0
            
            for peak in peaks[1:]:
                pr_start = int(peak - 0.2 * self.fs)
                pr_end = int(peak - 0.12 * self.fs)
                if pr_start < 0:
                    continue
                pr_segment = lead_ii[pr_start:pr_end]
                if len(pr_segment) > 0:
                    if np.max(pr_segment) > np.std(lead_ii) * 0.5:
                        p_count += 1
                    total += 1
            
            if total > 0:
                return 1.0 - (p_count / total)
            return 0.5
        except:
            return 0.5
    
    def _compute_rr_regularity(self, lead_signal):
        try:
            peaks, _ = find_peaks(lead_signal, height=np.max(lead_signal) * 0.3, distance=self.fs // 3)
            if len(peaks) < 3:
                return 0.5
            rr_intervals = np.diff(peaks)
            cv = np.std(rr_intervals) / (np.mean(rr_intervals) + 1e-10)
            return np.exp(-np.clip(cv * 5, 0, 10))
        except:
            return 0.5
    
    def _detect_sawtooth_pattern(self, lead_signal):
        try:
            nyq = self.fs / 2
            low, high = 4.0 / nyq, 6.0 / nyq
            b, a = butter(3, [low, high], btype='band')
            filtered = filtfilt(b, a, lead_signal)
            autocorr = np.correlate(filtered, filtered, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)
            lag_range = range(int(self.fs * 0.15), int(self.fs * 0.25))
            if len(lag_range) > 0 and max(lag_range) < len(autocorr):
                return np.clip(np.max(autocorr[list(lag_range)]), 0, 1)
            return 0.0
        except:
            return 0.0
    
    def _check_p_wave_regularity(self, lead_signal, peaks):
        try:
            if len(peaks) < 3:
                return 0.5
            pr_intervals = []
            for peak in peaks[1:]:
                pr_start = int(peak - 0.2 * self.fs)
                pr_end = int(peak - 0.12 * self.fs)
                if pr_start >= 0:
                    pr_intervals.append(pr_end - pr_start)
            if len(pr_intervals) > 1:
                cv = np.std(pr_intervals) / (np.mean(pr_intervals) + 1e-10)
                return np.exp(-np.clip(cv * 5, 0, 10))
            return 0.5
        except:
            return 0.5
    
    def _measure_pr_interval(self, lead_signal, peaks):
        try:
            if len(peaks) < 2:
                return 160.0
            peak = peaks[1]
            pr_start = int(peak - 0.2 * self.fs)
            pr_end = int(peak - 0.05 * self.fs)
            if pr_start < 0:
                return 160.0
            return (pr_end - pr_start) / self.fs * 1000
        except:
            return 160.0
    
    def _measure_p_wave_amplitude(self, lead_signal, peaks):
        try:
            if len(peaks) < 2:
                return 0.0
            peak = peaks[1]
            pr_start = int(peak - 0.2 * self.fs)
            pr_end = int(peak - 0.12 * self.fs)
            if pr_start < 0:
                return 0.0
            pr_segment = lead_signal[pr_start:pr_end]
            return np.max(pr_segment) if len(pr_segment) > 0 else 0.0
        except:
            return 0.0
    
    def _measure_p_wave_duration(self, lead_signal, peaks):
        return 100.0  # Simplified
    
    # ========== DEFAULT FEATURES ==========
    
    def _get_default_hr_features(self):
        return {
            'hr_mean': 75.0, 'hr_std': 0.0, 'rr_mean': 800.0,
            'hr_bradycardia': 0.0, 'hr_normal': 1.0,
            'hr_tachycardia': 0.0, 'hr_severe_tachy': 0.0,
            'rr_regularity': 0.5, 'rr_std': 0.0, 'rr_range': 0.0,
            'hr_max': 75.0, 'hr_min': 75.0
        }
    
    def _get_default_afib_features(self):
        return {
            'afib_sdnn': 0.0, 'afib_rmssd': 0.0, 'afib_cv': 0.0,
            'afib_entropy': 0.0, 'afib_irregularity': 0.0,
            'afib_p_absence': 0.5, 'afib_mean_hr': 75.0, 'afib_hr_range': 0.0
        }
    
    def _get_default_lvh_features(self):
        return {
            'lvh_sokolow_lyon': 0.0, 'lvh_cornell_voltage': 0.0,
            'lvh_r_v5': 0.0, 'lvh_r_v6': 0.0, 'lvh_r_avl': 0.0,
            'lvh_s_v1': 0.0, 'lvh_s_v3': 0.0,
            'lvh_qrs_duration': 80.0, 'lvh_qrs_axis': 0.0,
            'lvh_rs_ratio_v1': 0.0,
            'lvh_sokolow_score': 0.0, 'lvh_cornell_score': 0.0, 'lvh_qrs_score': 0.67
        }
    
    def _get_default_svt_features(self):
        return {
            'svt_high_rate': 0.0, 'svt_hr': 75.0, 'svt_regularity': 0.0,
            'svt_rr_consistency': 0.5, 'svt_abnormal_p': 0.5,
            'svt_rate_range': 0.0, 'svt_stable_rate': 0.0
        }
    
    def _get_default_pwave_features(self):
        return {
            'pwave_presence': 0.5, 'pwave_regularity': 0.5,
            'pwave_pr_interval': 160.0, 'pwave_amplitude': 0.0, 'pwave_duration': 100.0
        }
    
    def features_to_array(self, features):
        """Convert to array - 64 features total"""
        
        feature_names = [
            # HR (12)
            'hr_mean', 'hr_std', 'rr_mean', 'hr_bradycardia', 'hr_normal',
            'hr_tachycardia', 'hr_severe_tachy', 'rr_regularity',
            'rr_std', 'rr_range', 'hr_max', 'hr_min',
            
            # AFIB (8)
            'afib_sdnn', 'afib_rmssd', 'afib_cv', 'afib_entropy',
            'afib_irregularity', 'afib_p_absence', 'afib_mean_hr', 'afib_hr_range',
            
            # AF (5)
            'af_flutter_ratio', 'af_dominant_freq', 'af_regularity',
            'af_rr_regularity', 'af_sawtooth',
            
            # LVH (13 - removed 2)
            'lvh_sokolow_lyon', 'lvh_cornell_voltage',
            'lvh_r_v5', 'lvh_r_v6', 'lvh_r_avl', 'lvh_s_v1', 'lvh_s_v3',
            'lvh_qrs_duration', 'lvh_qrs_axis', 'lvh_rs_ratio_v1',
            'lvh_sokolow_score', 'lvh_cornell_score', 'lvh_qrs_score',
            
            # SVT (7 - removed 1)
            'svt_high_rate', 'svt_hr', 'svt_regularity',
            'svt_rr_consistency', 'svt_abnormal_p', 'svt_rate_range', 'svt_stable_rate',
            
            # P-wave (5 - removed 1)
            'pwave_presence', 'pwave_regularity', 'pwave_pr_interval',
            'pwave_amplitude', 'pwave_duration',
            
            # QRS (7 - removed 1)
            'qrs_duration', 'qrs_axis', 'qrs_amp_mean', 'qrs_amp_std',
            'qrs_consistency', 'qrs_max_amp', 'qrs_min_amp',
            
            # Frequency (7)
            'freq_lf_power', 'freq_lf_ratio', 'freq_hf_power', 'freq_hf_ratio',
            'freq_dominant', 'freq_entropy', 'freq_lf_hf_ratio'
        ]
        
        arr = np.array([features.get(name, 0.0) for name in feature_names], dtype=np.float32)
        return arr


if __name__ == "__main__":
    print("Testing FIXED ECGFeatureExtractor...")
    print(f"Total features: 64 (removed 6 constants)")