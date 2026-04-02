"""
ecg_quantitative_eval_v2.py
============================
Module đánh giá ĐỊNH LƯỢNG attention map cho ECG Classification.
Nhãn: SB, SR, AF, SVT, ST  |  Kiến trúc: ResNet18 + Attention + BiLSTM
Dataset: PhysioNet Challenge 2020  |  fs: 500 Hz  |  Input: Z-score

THAY ĐỔI SO VỚI V1:
    [NEW] BLOCK 4b — ACVS (Attention Clinical Validity Score)
          Trích đặc trưng từ vùng top 20% attention → so ngưỡng lâm sàng
    [FIX] detect_r_peaks: distance động theo label, bắt được SVT/ST (>180bpm)
    [FIX] Ngưỡng RR-CV SR/SB mở rộng lên 10% theo sinh lý thực tế
    [FIX] AF thêm RMSSD và pNN50 thay vì chỉ dùng CV
    [NEW] BLOCK 5b — ACVS baseline ratio: % mẫu "pass" từng tiêu chí lâm sàng
    [NEW] _build_summary bổ sung acvs_score và acvs_pass_rate per label

Import vào Jupyter:
    import sys
    sys.path.insert(0, r"E:\\NCKH - 2026\\ECG Project")
    from ecg_quantitative_eval_v2 import *
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════
# BLOCK 1 — PHÁT HIỆN R-PEAK (FIX: hỗ trợ nhịp nhanh SVT/ST)
# ══════════════════════════════════════════════════════════════

# Ngưỡng distance tối thiểu giữa 2 R-peak theo label (ms)
# SVT/ST có thể >180bpm → RR < 333ms → distance phải nhỏ hơn
LABEL_MIN_RR_MS = {
    'SR' : 450,   # ~133 bpm max
    'SB' : 600,   # nhịp chậm, khoảng rộng hơn
    'AF' : 300,   # AF có thể đập nhanh bất thường
    'ST' : 280,   # sinus tachy có thể >200bpm
    'SVT': 220,   # SVT 150-250bpm → min ~240ms
}

def detect_r_peaks(signal_1d, fs=500, label=None):
    """
    Phát hiện R-peaks từ 1 lead ECG đã Z-score.
    Khuyến nghị dùng Lead II (index 1).

    Parameters
    ----------
    signal_1d : np.ndarray  shape (seq_len,)
    fs        : int
    label     : str or None  nếu cung cấp, dùng distance động theo label

    Returns
    -------
    r_peaks : np.ndarray  indices của các R-peak
    """
    # Distance động theo label để không bỏ sót SVT/ST
    if label and label in LABEL_MIN_RR_MS:
        min_rr_ms = LABEL_MIN_RR_MS[label]
    else:
        min_rr_ms = 300  # fallback mặc định an toàn

    distance = int(fs * min_rr_ms / 1000)

    peaks, _ = find_peaks(
        signal_1d,
        height=0.7,           # Z-score threshold (hạ xuống để bắt được SVT yếu)
        distance=distance,
        prominence=0.5
    )
    return peaks


def get_qrs_windows(r_peaks, fs=500, pre_ms=60, post_ms=80):
    """
    Tạo cửa sổ QRS complex xung quanh mỗi R-peak.
    Chuẩn lâm sàng: QRS ~ 80-120ms => lấy pre=60ms, post=80ms.
    """
    pre  = int(pre_ms  / 1000 * fs)
    post = int(post_ms / 1000 * fs)
    return [(max(0, r - pre), r + post) for r in r_peaks]


def get_rr_intervals(r_peaks, fs=500):
    """
    Tính RR intervals từ mảng R-peak indices.

    Returns
    -------
    rr_ms      : np.ndarray  RR intervals (ms)
    rr_mean    : float       mean RR (ms)
    heart_rate : float       nhịp tim (bpm)
    """
    if len(r_peaks) < 2:
        return np.array([]), None, None
    rr_ms      = np.diff(r_peaks) / fs * 1000
    rr_mean    = float(np.mean(rr_ms))
    heart_rate = 60000.0 / rr_mean if rr_mean > 0 else 0.0
    return rr_ms, rr_mean, heart_rate


# ══════════════════════════════════════════════════════════════
# BLOCK 2 — METRIC 1: QRS ENERGY RATIO
# ══════════════════════════════════════════════════════════════

def compute_attention_energy_in_qrs(attn_weights_raw, r_peaks, signal_len, fs=500):
    """
    Tính % năng lượng attention tập trung trong vùng QRS.
    > 50%  => model đang nhìn đúng vùng QRS
    """
    attn_up = np.interp(
        np.arange(signal_len),
        np.linspace(0, signal_len - 1, len(attn_weights_raw)),
        attn_weights_raw
    )
    attn_up = np.abs(attn_up)

    qrs_mask = np.zeros(signal_len, dtype=bool)
    for (s, e) in get_qrs_windows(r_peaks, fs=fs):
        qrs_mask[s:e] = True

    energy_ratio = float(attn_up[qrs_mask].sum() / (attn_up.sum() + 1e-9))
    return energy_ratio, qrs_mask, attn_up


# ══════════════════════════════════════════════════════════════
# BLOCK 3 — METRIC 2: RR STATISTICS & CV (FIX ngưỡng)
# ══════════════════════════════════════════════════════════════

def compute_rr_stats(r_peaks, fs=500, gt_rr_ms=None):
    """
    Tính thống kê RR interval.

    Ngưỡng lâm sàng (đã cập nhật):
        SR  : RR mean 600-1000ms, CV < 10%  (sinh lý cho phép ~5-10%)
        SB  : RR mean > 1000ms,   CV < 10%
        AF  : CV > 15%, RMSSD > 50ms, pNN50 > 20%  (rối loạn thực sự)
        ST  : RR mean 400-600ms,  CV < 10%
        SVT : RR mean < 400ms,    CV < 10%

    Returns
    -------
    mae      : float or None
    rr_cv    : float  Coefficient of Variation (%)
    rr_stats : dict
    """
    rr_ms, rr_mean, hr = get_rr_intervals(r_peaks, fs)
    if len(rr_ms) == 0:
        return None, None, {}

    rr_std  = float(np.std(rr_ms))
    rr_cv   = rr_std / rr_mean * 100 if rr_mean else 0.0

    # RMSSD: root mean square of successive differences (chỉ số HRV)
    rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))) if len(rr_ms) >= 2 else 0.0

    # pNN50: % cặp RR liên tiếp chênh nhau > 50ms (đặc trưng AF)
    if len(rr_ms) >= 2:
        successive_diff = np.abs(np.diff(rr_ms))
        pnn50 = float(np.mean(successive_diff > 50) * 100)
    else:
        pnn50 = 0.0

    rr_stats = {
        'rr_mean_ms'  : round(rr_mean, 1),
        'rr_std_ms'   : round(rr_std, 1),
        'rr_cv_pct'   : round(rr_cv, 1),
        'rr_min_ms'   : round(float(np.min(rr_ms)), 1),
        'rr_max_ms'   : round(float(np.max(rr_ms)), 1),
        'heart_rate'  : round(hr, 1),
        'n_beats'     : len(rr_ms) + 1,
        'rmssd_ms'    : round(rmssd, 1),   # [NEW]
        'pnn50_pct'   : round(pnn50, 1),   # [NEW]
    }

    mae = None
    if gt_rr_ms is not None and len(gt_rr_ms) > 0:
        n = min(len(rr_ms), len(gt_rr_ms))
        mae = float(np.mean(np.abs(rr_ms[:n] - np.array(gt_rr_ms)[:n])))
        rr_stats['rr_mae_ms'] = round(mae, 1)

    return mae, float(rr_cv), rr_stats


# ══════════════════════════════════════════════════════════════
# BLOCK 4 — METRIC 3: TEMPORAL LOCALIZATION SCORE (TLS)
# ══════════════════════════════════════════════════════════════

LABEL_CLINICAL_REGIONS = {
    # Mỗi nhãn: vùng "đặc trưng lâm sàng" được định nghĩa trước R-peak
    # AF: pre rộng để bắt vùng baseline trước QRS (nơi sóng F xuất hiện)
    'SB' : {'name': 'QRS + Post-T Region', 'pre_ms': 60,  'post_ms': 400},  # khoảng nghỉ dài
    'SR' : {'name': 'Inter-R Pause',       'pre_ms': 80,  'post_ms': 200},  # khoảng giữa R-R
    'AF' : {'name': 'Pre-QRS Baseline',    'pre_ms': 250, 'post_ms': 80 },  # vùng sóng F
    'SVT': {'name': 'Narrow QRS Complex',  'pre_ms': 40,  'post_ms': 50 },  # QRS hẹp
    'ST' : {'name': 'Short Inter-R Pause', 'pre_ms': 50,  'post_ms': 150},  # khoảng ngắn nhưng đều
}


def compute_temporal_localization_score(attn_up, r_peaks, label_name,
                                         fs=500, signal_len=None):
    """
    Tính Temporal Localization Score (TLS).
    > 60%  => attention khớp tốt với đặc trưng lâm sàng
    """
    if signal_len is None:
        signal_len = len(attn_up)

    cfg  = LABEL_CLINICAL_REGIONS.get(label_name, LABEL_CLINICAL_REGIONS['SR'])
    pre  = int(cfg['pre_ms']  / 1000 * fs)
    post = int(cfg['post_ms'] / 1000 * fs)

    region_mask = np.zeros(signal_len, dtype=bool)
    for r in r_peaks:
        region_mask[max(0, r - pre): min(signal_len, r + post)] = True

    tls = float(attn_up[region_mask].sum() / (attn_up.sum() + 1e-9))

    region_info = {
        'label'           : label_name,
        'region_name'     : cfg['name'],
        'coverage_pct'    : round(region_mask.sum() / signal_len * 100, 1),
        'attn_energy_pct' : round(tls * 100, 1),
    }
    return tls, region_mask, region_info


# ══════════════════════════════════════════════════════════════
# BLOCK 4b — METRIC 4: ACVS (Attention Clinical Validity Score)  [NEW]
# ══════════════════════════════════════════════════════════════

# Định nghĩa tiêu chí lâm sàng cho từng nhãn
# Mỗi tiêu chí là 1 lambda nhận (rr_ms, rr_mean, rr_cv, signal_at_attn, rmssd, pnn50) → bool
ACVS_CRITERIA = {
    # SR: Nhịp xoang bình thường
    # - RR mean 600–1000ms (60–100bpm)
    # - RR đều (CV < 10%)
    'SR': [
        lambda d: 600 <= d['rr_mean'] <= 1000,    # HR 60-100bpm
        lambda d: d['rr_cv'] < 10.0,               # nhịp đều
        lambda d: d['attn_variance'] < 0.3,         # attention không quá lởm chởm (nhịp ổn)
    ],

    # SB: Sinus bradycardia
    # - RR mean > 1000ms (<60bpm)
    # - attention vào khoảng nghỉ DÀI → signal tại đó phải thấp (không phải đỉnh R)
    'SB': [
        lambda d: d['rr_mean'] > 1000,             # HR < 60bpm
        lambda d: d['rr_cv'] < 10.0,               # vẫn là nhịp xoang (đều)
        lambda d: d['attn_on_rpeak_ratio'] < 0.4,  # attention NOT trên R-peak (tập trung khoảng nghỉ)
    ],

    # AF: Atrial flutter / fibrillation
    # - RR không đều (CV > 15% hoặc pNN50 > 20%)
    # - RMSSD cao (HRV cao)
    # - Attention bám baseline, KHÔNG phải trên đỉnh R
    # - Variance tín hiệu tại vùng attention cao > ngưỡng (baseline răng cưa)
    'AF': [
        lambda d: d['rr_cv'] > 15.0 or d['pnn50'] > 20.0,  # nhịp không đều
        lambda d: d['rmssd'] > 40.0,                         # HRV cao
        lambda d: d['attn_on_rpeak_ratio'] < 0.3,            # attention KHÔNG trên R-peak
        lambda d: d['attn_variance'] > 0.15,                 # baseline lởm chởm
    ],

    # ST: Sinus tachycardia
    # - RR mean 400–600ms (100–150bpm)
    # - RR đều (nhịp xoang)
    # - Attention tập trung khoảng nghỉ ngắn
    'ST': [
        lambda d: 400 <= d['rr_mean'] <= 700,      # HR 86-150bpm (nới rộng thực tế)
        lambda d: d['rr_cv'] < 12.0,               # đều
        lambda d: d['heart_rate'] > 90,             # chắc chắn là nhịp nhanh
    ],

    # SVT: Supraventricular tachycardia
    # - RR mean < 400ms (>150bpm)
    # - RR khá đều (nhịp đều nhưng rất nhanh)
    # - Attention trải trên QRS hẹp → variance thấp tại vùng attention
    # - QRS width tại vùng attention phải hẹp (<120ms)
    'SVT': [
        lambda d: d['rr_mean'] < 450,              # HR > 133bpm
        lambda d: d['heart_rate'] > 130,            # xác nhận nhịp nhanh
        lambda d: d['qrs_width_ms'] < 140,          # QRS hẹp (SVT đặc trưng)
    ],
}

# Trọng số để tính ACVS tổng hợp (tiêu chí quan trọng hơn → weight cao hơn)
ACVS_WEIGHTS = {
    'SR' : [0.40, 0.35, 0.25],
    'SB' : [0.45, 0.30, 0.25],
    'AF' : [0.30, 0.25, 0.25, 0.20],
    'ST' : [0.40, 0.35, 0.25],
    'SVT': [0.40, 0.35, 0.25],
}


def compute_qrs_width(signal_1d, r_peaks, fs=500, threshold_ratio=0.5):
    """
    Ước tính độ rộng QRS (ms) bằng cách đo half-peak width quanh R-peak.

    Parameters
    ----------
    signal_1d       : np.ndarray  lead signal (Z-score)
    r_peaks         : np.ndarray  R-peak indices
    fs              : int
    threshold_ratio : float  ngưỡng % peak để đo width (default 50% = half-width)

    Returns
    -------
    qrs_width_ms : float  trung bình độ rộng QRS (ms)
    """
    if len(r_peaks) == 0:
        return 0.0

    widths = []
    for r in r_peaks:
        peak_val = signal_1d[r]
        if peak_val <= 0:
            continue
        threshold = threshold_ratio * peak_val

        # Tìm điểm trái vượt threshold
        left  = r
        while left > 0 and signal_1d[left] > threshold:
            left -= 1

        # Tìm điểm phải vượt threshold
        right = r
        while right < len(signal_1d) - 1 and signal_1d[right] > threshold:
            right += 1

        width_samples = right - left
        widths.append(width_samples / fs * 1000)  # đổi ra ms

    return float(np.mean(widths)) if widths else 0.0


def compute_acvs(signal_1d, attn_up, r_peaks, label_name, rr_stats, fs=500,
                 attn_top_pct=0.20):
    """
    Tính Attention Clinical Validity Score (ACVS).

    Logic:
        1. Lấy top {attn_top_pct}% attention → xác định vùng "quan trọng"
        2. Trích đặc trưng tín hiệu tại vùng đó
        3. So sánh với tiêu chí lâm sàng của nhãn gốc
        4. Trả về score [0,1] và chi tiết từng tiêu chí

    Parameters
    ----------
    signal_1d   : np.ndarray  shape (seq_len,)  — 1 lead ECG
    attn_up     : np.ndarray  shape (seq_len,)  — attention đã upsample
    r_peaks     : np.ndarray  R-peak indices
    label_name  : str
    rr_stats    : dict        từ compute_rr_stats()
    fs          : int
    attn_top_pct: float       top % attention để lấy vùng quan trọng

    Returns
    -------
    acvs        : float [0,1]  điểm tổng hợp
    criteria_pass : list[bool]  từng tiêu chí pass/fail
    detail      : dict         toàn bộ đặc trưng đã trích
    """
    if len(r_peaks) < 2:
        return None, [], {}

    # --- Bước 1: Xác định vùng attention cao ---
    threshold  = np.quantile(attn_up, 1.0 - attn_top_pct)
    high_mask  = attn_up >= threshold   # boolean mask

    # --- Bước 2: Trích đặc trưng từ vùng đó ---

    # 2a. Variance tín hiệu tại vùng attention cao
    # (cao = baseline lởm chởm như AF; thấp = vùng ổn định như SR/SB)
    attn_signal = signal_1d[high_mask]
    attn_variance = float(np.var(attn_signal)) if len(attn_signal) > 0 else 0.0

    # 2b. Tỉ lệ attention trùng với R-peak
    # Tạo R-peak mask: vùng ±20ms quanh mỗi R-peak
    rpeak_halo = int(0.020 * fs)   # 20ms
    rpeak_mask = np.zeros(len(signal_1d), dtype=bool)
    for r in r_peaks:
        rpeak_mask[max(0, r - rpeak_halo): min(len(signal_1d), r + rpeak_halo)] = True

    # % attention cao mà trùng với R-peak
    overlap = high_mask & rpeak_mask
    attn_on_rpeak_ratio = float(overlap.sum() / (high_mask.sum() + 1e-9))

    # 2c. RR tại vùng attention cao — lấy R-peaks nằm trong vùng attention cao
    r_in_attn = [r for r in r_peaks if high_mask[r]]
    if len(r_in_attn) >= 2:
        rr_attn    = np.diff(r_in_attn) / fs * 1000
        rr_attn_mean = float(np.mean(rr_attn))
        rr_attn_cv   = float(np.std(rr_attn) / rr_attn_mean * 100) if rr_attn_mean else 0.0
    else:
        # fallback: dùng toàn bộ RR
        rr_attn_mean = rr_stats.get('rr_mean_ms', 0.0) or 0.0
        rr_attn_cv   = rr_stats.get('rr_cv_pct', 0.0) or 0.0

    # 2d. QRS width ước tính
    qrs_width_ms = compute_qrs_width(signal_1d, r_peaks, fs=fs)

    # --- Gói toàn bộ đặc trưng vào dict ---
    feature_dict = {
        'rr_mean'              : rr_attn_mean,
        'rr_cv'                : rr_attn_cv,
        'heart_rate'           : rr_stats.get('heart_rate', 0.0) or 0.0,
        'rmssd'                : rr_stats.get('rmssd_ms', 0.0) or 0.0,
        'pnn50'                : rr_stats.get('pnn50_pct', 0.0) or 0.0,
        'attn_variance'        : attn_variance,
        'attn_on_rpeak_ratio'  : attn_on_rpeak_ratio,
        'qrs_width_ms'         : qrs_width_ms,
        'n_r_in_attn_region'   : len(r_in_attn),
    }

    # --- Bước 3: Đánh giá từng tiêu chí lâm sàng ---
    criteria = ACVS_CRITERIA.get(label_name, [])
    weights  = ACVS_WEIGHTS.get(label_name, [1.0 / len(criteria)] * len(criteria))

    criteria_pass = []
    for criterion in criteria:
        try:
            result = bool(criterion(feature_dict))
        except Exception:
            result = False
        criteria_pass.append(result)

    # --- Bước 4: Tính ACVS tổng hợp (weighted average) ---
    if not criteria_pass:
        acvs = 0.0
    else:
        w = np.array(weights[:len(criteria_pass)], dtype=float)
        w = w / w.sum()   # chuẩn hóa tổng = 1
        acvs = float(np.dot(criteria_pass, w))

    detail = {
        'acvs'           : round(acvs, 4),
        'criteria_pass'  : criteria_pass,
        'n_criteria'     : len(criteria),
        'n_pass'         : sum(criteria_pass),
        **{f'feat_{k}': round(float(v), 4) if isinstance(v, (int, float)) else v
           for k, v in feature_dict.items()}
    }

    return acvs, criteria_pass, detail


# ══════════════════════════════════════════════════════════════
# BLOCK 5 — ĐÁNH GIÁ TOÀN TẬP TEST (bổ sung ACVS)
# ══════════════════════════════════════════════════════════════

def evaluate_attention_quantitative(
    model,
    test_loader,
    y_test,
    target_names,
    device,
    fs=500,
    lead_for_rpeak=1,
    n_samples_max=None,
    verbose=True,
):
    """
    Chạy đánh giá định lượng toàn tập test.
    Tự động gọi get_attention_weights() từ visualize/attention_map.py.

    Returns
    -------
    all_results : list[dict]  kết quả per-sample (bao gồm ACVS)
    summary     : dict        thống kê per-label + overall
    """
    from visualize.attention_map import get_attention_weights

    model.eval()
    model = model.to(device)

    if isinstance(y_test, torch.Tensor):
        y_test_np = y_test.cpu().numpy()
    else:
        y_test_np = np.asarray(y_test)

    all_results = []
    sample_idx  = 0

    for batch_signals, _ in tqdm(test_loader, desc="Đánh giá định lượng"):
        batch_signals = batch_signals.to(device)

        with torch.no_grad():
            outputs = model(batch_signals)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = torch.sigmoid(outputs).cpu().numpy()

        for b in range(len(batch_signals)):
            if n_samples_max and sample_idx >= n_samples_max:
                break

            signal_tensor = batch_signals[b]
            signal_np     = signal_tensor.cpu().numpy()
            seq_len       = signal_np.shape[1]

            gt_vec    = y_test_np[sample_idx]
            pred_vec  = probs[b]
            gt_idx    = int(np.argmax(gt_vec))
            pred_idx  = int(np.argmax(pred_vec))
            gt_name   = target_names[gt_idx]
            pred_name = target_names[pred_idx]

            try:
                attn_w = get_attention_weights(model, signal_tensor)
            except Exception:
                sample_idx += 1
                continue

            lead_sig = signal_np[lead_for_rpeak]

            # FIX: detect R-peak với distance động theo nhãn gốc
            r_peaks = detect_r_peaks(lead_sig, fs=fs, label=gt_name)
            valid   = len(r_peaks) >= 2

            attn_up = np.interp(
                np.arange(seq_len),
                np.linspace(0, seq_len - 1, len(attn_w)),
                attn_w
            )

            qrs_ratio = None
            if valid:
                qrs_ratio, _, _ = compute_attention_energy_in_qrs(
                    attn_w, r_peaks, seq_len, fs=fs
                )

            _, rr_cv, rr_stats = compute_rr_stats(r_peaks, fs=fs)

            tls = None
            if valid:
                tls, _, _ = compute_temporal_localization_score(
                    attn_up, r_peaks, gt_name, fs=fs, signal_len=seq_len
                )

            # [NEW] Tính ACVS
            acvs        = None
            acvs_detail = {}
            if valid:
                acvs, criteria_pass, acvs_detail = compute_acvs(
                    signal_1d=lead_sig,
                    attn_up=attn_up,
                    r_peaks=r_peaks,
                    label_name=gt_name,
                    rr_stats=rr_stats,
                    fs=fs
                )

            row = {
                'sample_idx'       : sample_idx,
                'gt_label'         : gt_name,
                'pred_label'       : pred_name,
                'correct'          : (gt_idx == pred_idx),
                'pred_conf'        : round(float(pred_vec[pred_idx]), 4),
                'n_r_peaks'        : len(r_peaks),
                'qrs_energy_ratio' : round(qrs_ratio, 4)  if qrs_ratio is not None else None,
                'tls'              : round(tls, 4)         if tls       is not None else None,
                'acvs'             : round(acvs, 4)        if acvs      is not None else None,  # [NEW]
                'rr_mean_ms'       : rr_stats.get('rr_mean_ms'),
                'rr_cv_pct'        : rr_stats.get('rr_cv_pct'),
                'heart_rate'       : rr_stats.get('heart_rate'),
                'rmssd_ms'         : rr_stats.get('rmssd_ms'),          # [NEW]
                'pnn50_pct'        : rr_stats.get('pnn50_pct'),         # [NEW]
            }
            # Gắn thêm chi tiết ACVS từng tiêu chí
            if acvs_detail:
                row.update({
                    'acvs_n_pass'    : acvs_detail.get('n_pass'),
                    'acvs_n_crit'    : acvs_detail.get('n_criteria'),
                    'feat_rr_mean'   : acvs_detail.get('feat_rr_mean'),
                    'feat_rr_cv'     : acvs_detail.get('feat_rr_cv'),
                    'feat_attn_var'  : acvs_detail.get('feat_attn_variance'),
                    'feat_rpeak_ratio': acvs_detail.get('feat_attn_on_rpeak_ratio'),
                    'feat_qrs_width' : acvs_detail.get('feat_qrs_width_ms'),
                })

            all_results.append(row)
            sample_idx += 1

        if n_samples_max and sample_idx >= n_samples_max:
            break

    summary = _build_summary(all_results, target_names, verbose=verbose)
    return all_results, summary


# ══════════════════════════════════════════════════════════════
# BLOCK 5b — BUILD SUMMARY (bổ sung ACVS)
# ══════════════════════════════════════════════════════════════

def _build_summary(all_results, target_names, verbose=True):
    valid = [r for r in all_results if r.get('qrs_energy_ratio') is not None]

    def safe_mean(lst):
        lst = [x for x in lst if x is not None]
        return float(np.mean(lst)) if lst else 0.0

    summary = {
        'overall': {
            'n_samples'           : len(all_results),
            'n_valid'             : len(valid),
            'accuracy_pct'        : round(np.mean([r['correct'] for r in all_results]) * 100, 2),
            'qrs_energy_mean_pct' : round(safe_mean([r['qrs_energy_ratio'] for r in valid]) * 100, 2),
            'tls_mean_pct'        : round(safe_mean([r['tls']              for r in valid]) * 100, 2),
            'acvs_mean'           : round(safe_mean([r['acvs']             for r in valid]), 4),  # [NEW]
        }
    }

    for label in target_names:
        sub = [r for r in valid if r['gt_label'] == label]
        if not sub:
            continue

        # ACVS pass rate: % mẫu có ACVS >= 0.6 (đạt ít nhất 60% tiêu chí)
        acvs_list     = [r['acvs'] for r in sub if r['acvs'] is not None]
        acvs_pass_rate = float(np.mean([1 if (a is not None and a >= 0.6) else 0
                                         for a in acvs_list])) * 100

        summary[label] = {
            'n_samples'           : len(sub),
            'accuracy_pct'        : round(np.mean([r['correct']            for r in sub]) * 100, 2),
            'qrs_energy_pct'      : round(safe_mean([r['qrs_energy_ratio'] for r in sub]) * 100, 2),
            'tls_pct'             : round(safe_mean([r['tls']              for r in sub]) * 100, 2),
            'acvs_mean'           : round(safe_mean(acvs_list), 4),          # [NEW]
            'acvs_pass_rate_pct'  : round(acvs_pass_rate, 2),               # [NEW] % mẫu ACVS>=0.6
            'rr_cv_mean_pct'      : round(safe_mean([r['rr_cv_pct']        for r in sub]), 2),
            'heart_rate_mean_bpm' : round(safe_mean([r['heart_rate']       for r in sub]), 1),
            'rmssd_mean_ms'       : round(safe_mean([r['rmssd_ms']         for r in sub]), 1),  # [NEW]
            'pnn50_mean_pct'      : round(safe_mean([r['pnn50_pct']        for r in sub]), 1),  # [NEW]
        }

    if verbose:
        _print_summary_table(summary, target_names)
    return summary


def _print_summary_table(summary, target_names):
    ov = summary['overall']
    print("\n" + "=" * 80)
    print("  KẾT QUẢ ĐÁNH GIÁ ĐỊNH LƯỢNG ATTENTION MAP (v2 + ACVS)")
    print("=" * 80)
    print(f"  Tổng mẫu   : {ov['n_samples']}  (hợp lệ: {ov['n_valid']})")
    print(f"  Accuracy   : {ov['accuracy_pct']}%")
    print(f"  QRS Energy : {ov['qrs_energy_mean_pct']}%   (>50% là tốt)")
    print(f"  TLS Score  : {ov['tls_mean_pct']}%   (>60% là tốt)")
    print(f"  ACVS Mean  : {ov['acvs_mean']:.4f}   (>0.60 là tốt)")   # [NEW]
    print()
    print(f"  {'Label':<7} {'N':>5} {'Acc%':>7} {'QRS%':>7} {'TLS%':>7} "
          f"{'ACVS':>7} {'Pass%':>7} {'RR-CV%':>8} {'RMSSD':>7}")
    print("  " + "-" * 70)
    for label in target_names:
        if label not in summary:
            continue
        s = summary[label]
        print(f"  {label:<7} {s['n_samples']:>5} {s['accuracy_pct']:>7.1f}"
              f" {s['qrs_energy_pct']:>7.1f} {s['tls_pct']:>7.1f}"
              f" {s['acvs_mean']:>7.4f} {s['acvs_pass_rate_pct']:>7.1f}"
              f" {s['rr_cv_mean_pct']:>8.1f} {s['rmssd_mean_ms']:>7.1f}")
    print()
    print("  Ghi chú:")
    print("  QRS%    > 50%   => Model tập trung đúng vùng QRS [OK]")
    print("  TLS%    > 60%   => Attention khớp với đặc trưng lâm sàng [OK]")
    print("  ACVS    > 0.60  => Đặc trưng tại vùng attention thỏa tiêu chí lâm sàng [OK]")
    print("  Pass%   > 70%   => Phần lớn mẫu vượt ngưỡng ACVS=0.6 [OK]")
    print("  RR-CV AF        => Nên cao (>15%) vì AF có RR không đều")
    print("  RMSSD AF        => Nên cao (>40ms) vì HRV cao trong AF")
    print("=" * 80)


# ══════════════════════════════════════════════════════════════
# BLOCK 6 — VISUALIZE: BIỂU ĐỒ TỔNG HỢP (bổ sung panel ACVS)
# ══════════════════════════════════════════════════════════════

def plot_quantitative_summary(summary, target_names, save_path=None):
    """
    5-subplot (2x3): Accuracy | QRS Energy | TLS | ACVS | ACVS Pass Rate | RR-CV
    """
    labels = [l for l in target_names if l in summary]
    acc      = [summary[l]['accuracy_pct']       for l in labels]
    qrs      = [summary[l]['qrs_energy_pct']     for l in labels]
    tls      = [summary[l]['tls_pct']            for l in labels]
    acvs_v   = [summary[l]['acvs_mean'] * 100    for l in labels]   # đổi sang %
    acvs_pr  = [summary[l]['acvs_pass_rate_pct'] for l in labels]
    rrcv     = [summary[l]['rr_cv_mean_pct']     for l in labels]
    x        = np.arange(len(labels))
    colors   = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0'][:len(labels)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Quantitative Attention Evaluation — ECG (SB/SR/AF/SVT/ST) v2",
                 fontsize=14, fontweight='bold')

    def _bar(ax, vals, title, threshold=None, overall=None, ylabel="%"):
        bars = ax.bar(x, vals, color=colors, edgecolor='black', alpha=0.85, width=0.55)
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel(ylabel); ax.set_ylim(0, max(list(vals) + [10]) * 1.30)
        ax.grid(axis='y', alpha=0.3)
        if threshold is not None:
            ax.axhline(threshold, color='orange', ls='--', lw=1.5,
                       label=f'Threshold {threshold}%')
        if overall is not None:
            ax.axhline(overall, color='red', ls='--', lw=1.5,
                       label=f'Overall {overall:.1f}')
        if threshold is not None or overall is not None:
            ax.legend(fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')

    _bar(axes[0, 0], acc,
         "Classification Accuracy (%)",
         overall=summary['overall']['accuracy_pct'])

    _bar(axes[0, 1], qrs,
         "QRS Energy Ratio (%)", #\n>50%: model nhìn đúng vùng QRS
         threshold=50,
         overall=summary['overall']['qrs_energy_mean_pct'])

    _bar(axes[0, 2], tls,
         "Temporal Localization Score (%)", #\n>60%: attention khớp đặc trưng lâm sàng
         threshold=60,
         overall=summary['overall']['tls_mean_pct'])

    _bar(axes[1, 0], acvs_v,
         "ACVS Mean (%)", #\n>60%: đặc trưng vùng attention thỏa tiêu chí
         threshold=60,
         overall=summary['overall']['acvs_mean'] * 100)

    _bar(axes[1, 1], acvs_pr,
         "ACVS Pass Rate (%)", #\n>70%: phần lớn mẫu vượt ngưỡng ACVS=0.6
         threshold=70)

    _bar(axes[1, 2], rrcv,
         "RR Interval CV (%)", # — Độ Không Đều Nhịp Tim\nAF nên cao nhất (>15%)
         threshold=15)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"[OK] Lưu: {save_path}")
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════
# BLOCK 7 — VISUALIZE: CHI TIẾT 1 MẪU (bổ sung ACVS panel)
# ══════════════════════════════════════════════════════════════

def plot_single_sample(signal_np, attn_weights, r_peaks,
                       gt_label, pred_label, pred_conf,
                       qrs_energy_ratio, tls, rr_stats,
                       acvs=None, acvs_detail=None,
                       fs=500, lead_for_rpeak=1, save_path=None):
    """
    Vẽ đầy đủ 1 mẫu ECG với các metric định lượng (bao gồm ACVS).
    """
    n_leads, seq_len = signal_np.shape
    time = np.arange(seq_len) / fs
    lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

    attn_up = np.interp(
        np.arange(seq_len),
        np.linspace(0, seq_len - 1, len(attn_weights)),
        attn_weights
    )
    high_attn   = attn_up > np.quantile(attn_up, 0.8)
    qrs_windows = get_qrs_windows(r_peaks, fs=fs)
    correct     = gt_label == pred_label
    status      = "[ĐÚNG]" if correct else "[SAI]"
    title_color = '#1B5E20' if correct else '#B71C1C'

    fig = plt.figure(figsize=(18, 16))
    gs  = fig.add_gridspec(5, 3, hspace=0.55, wspace=0.38)

    # Lead II chính
    ax0       = fig.add_subplot(gs[0, :2])
    lead_main = signal_np[lead_for_rpeak]
    ax0.plot(time, lead_main, 'k-', lw=1.0, zorder=2)
    for (s, e) in qrs_windows:
        ax0.axvspan(time[s], time[min(e, seq_len - 1)],
                    alpha=0.15, color='green', label='QRS window')
    if len(r_peaks) > 0:
        ax0.scatter(time[r_peaks], lead_main[r_peaks],
                    color='red', s=60, zorder=5, marker='v', label='R-peaks')
    ax0.scatter(time[high_attn], lead_main[high_attn],
                color='blue', s=8, alpha=0.5, zorder=3, label='High Attention (top 20%)')
    ax0.set_title(
        f"Lead II  |  GT: {gt_label}  ->  Pred: {pred_label} ({pred_conf * 100:.1f}%)  {status}",
        fontweight='bold', fontsize=12, color=title_color
    )
    ax0.set_ylabel("Z-score"); ax0.set_xlabel("Time (s)")
    ax0.grid(alpha=0.3)
    h, l = ax0.get_legend_handles_labels()
    ax0.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(),
               fontsize=8, loc='upper right')

    # Metric box (bổ sung ACVS)
    ax_m = fig.add_subplot(gs[0, 2])
    ax_m.axis('off')
    acvs_str      = f"{acvs * 100:.1f}%" if acvs is not None else "N/A"
    acvs_pass_str = ""
    if acvs_detail and 'criteria_pass' in acvs_detail:
        icons = ['✓' if p else '✗' for p in acvs_detail['criteria_pass']]
        acvs_pass_str = " [" + " ".join(icons) + "]"

    txt = (
        "METRICS\n"
        + "-" * 28 + "\n"
        + f"QRS Energy : {qrs_energy_ratio * 100:.1f}%\n"
        + f"TLS Score  : {tls * 100:.1f}%\n"
        + f"ACVS       : {acvs_str}{acvs_pass_str}\n"
        + "-" * 28 + "\n"
        + f"RR Mean    : {rr_stats.get('rr_mean_ms', 'N/A')} ms\n"
        + f"RR Std     : {rr_stats.get('rr_std_ms',  'N/A')} ms\n"
        + f"RR CV      : {rr_stats.get('rr_cv_pct',  'N/A')} %\n"
        + f"RMSSD      : {rr_stats.get('rmssd_ms',   'N/A')} ms\n"
        + f"pNN50      : {rr_stats.get('pnn50_pct',  'N/A')} %\n"
        + f"Heart Rate : {rr_stats.get('heart_rate',  'N/A')} bpm\n"
        + f"N beats    : {rr_stats.get('n_beats',     'N/A')}\n"
    )
    ax_m.text(0.05, 0.97, txt, transform=ax_m.transAxes,
              fontsize=9, va='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # Attention heatmap
    ax_a = fig.add_subplot(gs[1, :2])
    ax_a.fill_between(time, attn_up, color='crimson', alpha=0.7)
    for (s, e) in qrs_windows:
        ax_a.axvspan(time[s], time[min(e, seq_len - 1)], alpha=0.2, color='green')
    ax_a.set_title("Attention Weight theo Thời Gian  (xanh lá = vùng QRS)",
                   fontweight='bold')
    ax_a.set_ylabel("Attention"); ax_a.set_xlabel("Time (s)")
    ax_a.grid(alpha=0.3)

    # RR bar chart
    ax_rr = fig.add_subplot(gs[1, 2])
    rr_ms, rr_mean, _ = get_rr_intervals(r_peaks, fs=fs)
    if len(rr_ms) > 0:
        bar_c = ['#E53935' if abs(rr - rr_mean) > 0.1 * rr_mean
                 else '#43A047' for rr in rr_ms]
        ax_rr.bar(range(len(rr_ms)), rr_ms, color=bar_c, edgecolor='black', alpha=0.85)
        ax_rr.axhline(rr_mean, color='navy', ls='--', lw=1.5,
                      label=f'Mean {rr_mean:.0f} ms')
        ax_rr.set_title("RR Intervals  (đỏ = lệch >10%)", fontweight='bold')
        ax_rr.set_xlabel("Beat index"); ax_rr.set_ylabel("RR (ms)")
        ax_rr.legend(fontsize=8); ax_rr.grid(axis='y', alpha=0.3)
    else:
        ax_rr.text(0.5, 0.5, 'Không đủ R-peaks', ha='center', va='center')
        ax_rr.axis('off')

    # [NEW] ACVS bar chart từng tiêu chí
    ax_acvs = fig.add_subplot(gs[2, :])
    if acvs_detail and 'criteria_pass' in acvs_detail:
        crit_labels = [f"Crit {i+1}" for i in range(acvs_detail['n_criteria'])]
        crit_vals   = [1.0 if p else 0.0 for p in acvs_detail['criteria_pass']]
        crit_colors = ['#43A047' if p else '#E53935' for p in acvs_detail['criteria_pass']]
        ax_acvs.bar(crit_labels, crit_vals, color=crit_colors, edgecolor='black', width=0.4)
        ax_acvs.set_ylim(0, 1.3)
        ax_acvs.set_title(
            f"ACVS Từng Tiêu Chí — {gt_label}  |  Score: {acvs * 100:.1f}%  "
            f"({acvs_detail['n_pass']}/{acvs_detail['n_criteria']} pass)",
            fontweight='bold', fontsize=11
        )
        ax_acvs.set_ylabel("Pass (1) / Fail (0)")
        for i, (bar, v) in enumerate(zip(ax_acvs.patches, crit_vals)):
            label_txt = "PASS" if v == 1 else "FAIL"
            ax_acvs.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.05,
                         label_txt, ha='center', fontsize=11, fontweight='bold',
                         color='green' if v == 1 else 'red')
        ax_acvs.grid(axis='y', alpha=0.3)
    else:
        ax_acvs.text(0.5, 0.5, 'ACVS N/A', ha='center', va='center')
        ax_acvs.axis('off')

    # 6 leads phụ
    show = min(6, n_leads)
    for i in range(show):
        row = 3 + i // 3
        col = i % 3
        ax_l = fig.add_subplot(gs[row, col])
        ld   = signal_np[i]
        ax_l.plot(time, ld, 'k-', lw=0.7)
        ax_l.scatter(time[high_attn], ld[high_attn],
                     color='blue', s=4, alpha=0.4)
        lname = lead_names[i] if i < len(lead_names) else f'L{i+1}'
        ax_l.set_title(f"Lead {lname}", fontsize=9, fontweight='bold')
        ax_l.set_ylabel("Z-score", fontsize=7)
        ax_l.tick_params(labelsize=7)
        ax_l.grid(alpha=0.3)

    fig.suptitle(f"Chi Tiết Mẫu  |  GT: {gt_label}  |  Pred: {pred_label}  {status}",
                 fontsize=13, fontweight='bold', color=title_color)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Lưu: {save_path}")
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════
# BLOCK 8 — SAVE RESULTS TO CSV
# ══════════════════════════════════════════════════════════════

def save_results_csv(all_results, save_path="quantitative_results_v2.csv"):
    """Lưu toàn bộ kết quả per-sample ra CSV."""
    import csv
    if not all_results:
        print("Không có kết quả để lưu.")
        return
    fieldnames = list(all_results[0].keys())
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"[OK] Đã lưu {len(all_results)} dòng vào: {save_path}")


# ══════════════════════════════════════════════════════════════
# BLOCK 9 — QUICK EVAL 1 MẪU (helper tiện lợi)
# ══════════════════════════════════════════════════════════════

def eval_single_sample(model, signal_tensor, gt_label, target_names,
                        device, fs=500, lead_for_rpeak=1, plot=True, save_path=None):
    """
    Đánh giá nhanh 1 mẫu bất kỳ.
    Trả về toàn bộ metrics và vẽ biểu đồ nếu plot=True.

    Parameters
    ----------
    model         : ResNet18_LSTM_Attn đã load
    signal_tensor : torch.Tensor (12, seq_len)
    gt_label      : str
    target_names  : list[str]
    device        : torch.device
    fs            : int
    lead_for_rpeak: int
    plot          : bool
    save_path     : str or None

    Returns
    -------
    result : dict  toàn bộ metrics
    """
    from visualize.attention_map import get_attention_weights

    model.eval()
    signal_np = signal_tensor.cpu().numpy()
    seq_len   = signal_np.shape[1]

    with torch.no_grad():
        output = model(signal_tensor.unsqueeze(0).to(device))
        if isinstance(output, tuple):
            output = output[0]
        prob = torch.sigmoid(output).cpu().numpy()[0]

    pred_idx  = int(np.argmax(prob))
    pred_name = target_names[pred_idx]
    pred_conf = float(prob[pred_idx])

    attn_w = get_attention_weights(model, signal_tensor)
    attn_up = np.interp(
        np.arange(seq_len),
        np.linspace(0, seq_len - 1, len(attn_w)),
        attn_w
    )

    lead_sig = signal_np[lead_for_rpeak]
    r_peaks  = detect_r_peaks(lead_sig, fs=fs, label=gt_label)

    qrs_ratio, _, _ = compute_attention_energy_in_qrs(attn_w, r_peaks, seq_len, fs)
    _, rr_cv, rr_stats = compute_rr_stats(r_peaks, fs)
    tls, _, region_info = compute_temporal_localization_score(attn_up, r_peaks, gt_label, fs, seq_len)
    acvs, criteria_pass, acvs_detail = compute_acvs(lead_sig, attn_up, r_peaks, gt_label, rr_stats, fs)

    result = {
        'gt_label'       : gt_label,
        'pred_label'     : pred_name,
        'correct'        : gt_label == pred_name,
        'pred_conf'      : round(pred_conf, 4),
        'qrs_energy'     : round(qrs_ratio, 4),
        'tls'            : round(tls, 4),
        'acvs'           : round(acvs, 4) if acvs is not None else None,
        'acvs_detail'    : acvs_detail,
        'rr_stats'       : rr_stats,
        'region_info'    : region_info,
    }

    if plot:
        plot_single_sample(
            signal_np, attn_w, r_peaks,
            gt_label, pred_name, pred_conf,
            qrs_ratio, tls, rr_stats,
            acvs=acvs, acvs_detail=acvs_detail,
            fs=fs, lead_for_rpeak=lead_for_rpeak,
            save_path=save_path
        )

    return result




# """
# ecg_quantitative_eval.py
# ========================
# Module đánh giá ĐỊNH LƯỢNG attention map cho ECG Classification.
# Nhãn: SB, SR, AF, SVT, ST  |  Kiến trúc: ResNet18 + Attention + BiLSTM
# Dataset: PhysioNet Challenge 2020  |  fs: 500 Hz  |  Input: Z-score

# Đường dẫn project:
#     PROJECT_ROOT = r"E:\\NCKH - 2026\\ECG Project"
#     Data splits : data/processed/splits_leads12_160125/
#     Signal npy  : data/processed/ecg_leads12_160125/
#     Model       : models/resnet18_lstm_mixup_swa_160125.pth

# Import vào Jupyter:
#     import sys
#     sys.path.insert(0, r"E:\\NCKH - 2026\\ECG Project")
#     from ecg_quantitative_eval import *
# """

# import os
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.signal import find_peaks
# from tqdm import tqdm


# # ══════════════════════════════════════════════════════════════
# # BLOCK 1 — PHÁT HIỆN R-PEAK & CÁC SÓNG LÂM SÀNG
# # ══════════════════════════════════════════════════════════════

# def detect_r_peaks(signal_1d, fs=500):
#     """
#     Phát hiện R-peaks từ 1 lead ECG đã Z-score.
#     Khuyến nghị dùng Lead II (index 1) hoặc V5 (index 10).

#     Parameters
#     ----------
#     signal_1d : np.ndarray  shape (seq_len,)
#     fs        : int  tần số lấy mẫu (Hz)

#     Returns
#     -------
#     r_peaks : np.ndarray  indices của các R-peak
#     """
#     peaks, _ = find_peaks(
#         signal_1d,
#         height=0.8,               # Z-score threshold cho đỉnh R
#         distance=int(fs * 0.45),  # tối thiểu 450ms giữa 2 đỉnh (~133 bpm max)
#         prominence=0.6            # đỉnh R phải nổi bật hơn nền
#     )
#     return peaks


# def get_qrs_windows(r_peaks, fs=500, pre_ms=60, post_ms=80):
#     """
#     Tạo cửa sổ QRS complex xung quanh mỗi R-peak.
#     Chuẩn lâm sàng: QRS ~ 80-120ms => lấy pre=60ms, post=80ms.

#     Returns
#     -------
#     windows : list of (start_idx, end_idx)
#     """
#     pre  = int(pre_ms  / 1000 * fs)
#     post = int(post_ms / 1000 * fs)
#     return [(max(0, r - pre), r + post) for r in r_peaks]


# def get_rr_intervals(r_peaks, fs=500):
#     """
#     Tính RR intervals từ mảng R-peak indices.

#     Returns
#     -------
#     rr_ms      : np.ndarray  RR intervals (ms)
#     rr_mean    : float       mean RR (ms)
#     heart_rate : float       nhip tim (bpm)
#     """
#     if len(r_peaks) < 2:
#         return np.array([]), None, None
#     rr_ms      = np.diff(r_peaks) / fs * 1000
#     rr_mean    = float(np.mean(rr_ms))
#     heart_rate = 60000.0 / rr_mean if rr_mean > 0 else 0.0
#     return rr_ms, rr_mean, heart_rate


# # ══════════════════════════════════════════════════════════════
# # BLOCK 2 — METRIC 1: QRS ENERGY RATIO
# # ══════════════════════════════════════════════════════════════

# def compute_attention_energy_in_qrs(attn_weights_raw, r_peaks, signal_len, fs=500):
#     """
#     Tinh % nang luong attention tap trung trong vung QRS.

#     Y nghia:
#         > 50%  => model dang nhin dung vung QRS
#         < 30%  => attention rai deu, model chua hoc duoc pattern nhip tim

#     Parameters
#     ----------
#     attn_weights_raw : np.ndarray  shape (seq_len_feature,)
#     r_peaks          : np.ndarray  R-peak indices tren signal goc
#     signal_len       : int         do dai signal goc
#     fs               : int

#     Returns
#     -------
#     energy_ratio : float [0,1]
#     qrs_mask     : np.ndarray bool
#     attn_up      : np.ndarray  attention da upsample ve signal_len
#     """
#     attn_up = np.interp(
#         np.arange(signal_len),
#         np.linspace(0, signal_len - 1, len(attn_weights_raw)),
#         attn_weights_raw
#     )
#     attn_up = np.abs(attn_up)

#     qrs_mask = np.zeros(signal_len, dtype=bool)
#     for (s, e) in get_qrs_windows(r_peaks, fs=fs):
#         qrs_mask[s:e] = True

#     energy_ratio = float(attn_up[qrs_mask].sum() / (attn_up.sum() + 1e-9))
#     return energy_ratio, qrs_mask, attn_up


# # ══════════════════════════════════════════════════════════════
# # BLOCK 3 — METRIC 2: RR STATISTICS & CV
# # ══════════════════════════════════════════════════════════════

# def compute_rr_stats(r_peaks, fs=500, gt_rr_ms=None):
#     """
#     Tinh thong ke RR interval.

#     Nguong lam sang tham khao:
#         SR  : RR mean 600-1000ms, CV < 5%
#         SB  : RR mean > 1000ms,   CV < 5%
#         AF  : CV > 10%  (nhip khong deu)
#         ST  : RR mean 400-600ms,  CV < 5%
#         SVT : RR mean < 400ms,    CV < 5%

#     Returns
#     -------
#     mae      : float or None  MAE so voi ground truth (ms)
#     rr_cv    : float          Coefficient of Variation (%)
#     rr_stats : dict
#     """
#     rr_ms, rr_mean, hr = get_rr_intervals(r_peaks, fs)
#     if len(rr_ms) == 0:
#         return None, None, {}

#     rr_std = float(np.std(rr_ms))
#     rr_cv  = rr_std / rr_mean * 100 if rr_mean else 0.0

#     rr_stats = {
#         'rr_mean_ms' : round(rr_mean, 1),
#         'rr_std_ms'  : round(rr_std, 1),
#         'rr_cv_pct'  : round(rr_cv, 1),
#         'rr_min_ms'  : round(float(np.min(rr_ms)), 1),
#         'rr_max_ms'  : round(float(np.max(rr_ms)), 1),
#         'heart_rate' : round(hr, 1),
#         'n_beats'    : len(rr_ms) + 1,
#     }

#     mae = None
#     if gt_rr_ms is not None and len(gt_rr_ms) > 0:
#         n = min(len(rr_ms), len(gt_rr_ms))
#         mae = float(np.mean(np.abs(rr_ms[:n] - np.array(gt_rr_ms)[:n])))
#         rr_stats['rr_mae_ms'] = round(mae, 1)

#     return mae, float(rr_cv), rr_stats


# # ══════════════════════════════════════════════════════════════
# # BLOCK 4 — METRIC 3: TEMPORAL LOCALIZATION SCORE (TLS)
# # ══════════════════════════════════════════════════════════════

# # Vung lam sang quan trong cho tung nhan (pre/post R-peak, don vi ms)
# # Thu tu nhan: SB, SR, AF, SVT, ST  (khop voi target_names trong training)
# LABEL_CLINICAL_REGIONS = {
#     'SB' : {'name': 'QRS + Post-T Region', 'pre_ms': 60,  'post_ms': 300},
#     'SR' : {'name': 'QRS Complex',         'pre_ms': 60,  'post_ms': 80 },
#     'AF' : {'name': 'P-wave Loss + QRS',   'pre_ms': 200, 'post_ms': 80 },
#     'SVT': {'name': 'Narrow QRS Complex',  'pre_ms': 40,  'post_ms': 50 },
#     'ST' : {'name': 'Rapid QRS Complex',   'pre_ms': 50,  'post_ms': 60 },
# }


# def compute_temporal_localization_score(attn_up, r_peaks, label_name,
#                                          fs=500, signal_len=None):
#     """
#     Tinh Temporal Localization Score (TLS).
#     Do attention model co khop voi vung dac trung lam sang cua tung nhan khong.

#     Y nghia:
#         > 60%  => attention khop tot voi dac trung lam sang
#         < 40%  => model dang hoc sai dac trung

#     Parameters
#     ----------
#     attn_up    : np.ndarray  attention da upsample (shape = signal_len)
#     r_peaks    : np.ndarray
#     label_name : str  'SR' | 'SB' | 'AF' | 'ST' | 'SVT'
#     fs         : int
#     signal_len : int

#     Returns
#     -------
#     tls         : float [0,1]
#     region_mask : np.ndarray bool
#     region_info : dict
#     """
#     if signal_len is None:
#         signal_len = len(attn_up)

#     cfg  = LABEL_CLINICAL_REGIONS.get(label_name, LABEL_CLINICAL_REGIONS['SR'])
#     pre  = int(cfg['pre_ms']  / 1000 * fs)
#     post = int(cfg['post_ms'] / 1000 * fs)

#     region_mask = np.zeros(signal_len, dtype=bool)
#     for r in r_peaks:
#         region_mask[max(0, r - pre): min(signal_len, r + post)] = True

#     tls = float(attn_up[region_mask].sum() / (attn_up.sum() + 1e-9))

#     region_info = {
#         'label'           : label_name,
#         'region_name'     : cfg['name'],
#         'coverage_pct'    : round(region_mask.sum() / signal_len * 100, 1),
#         'attn_energy_pct' : round(tls * 100, 1),
#     }
#     return tls, region_mask, region_info


# # ══════════════════════════════════════════════════════════════
# # BLOCK 5 — DANH GIA TOAN TAP TEST
# # ══════════════════════════════════════════════════════════════

# def evaluate_attention_quantitative(
#     model,
#     test_loader,
#     y_test,
#     target_names,
#     device,
#     fs=500,
#     lead_for_rpeak=1,
#     n_samples_max=None,
#     verbose=True,
# ):
#     """
#     Chay danh gia dinh luong toan tap test.
#     Tu dong goi get_attention_weights() tu visualize/attention_map.py.

#     Parameters
#     ----------
#     model          : ResNet18_LSTM_Attn da load
#     test_loader    : DataLoader tap test
#     y_test         : np.ndarray  (N, num_classes)
#     target_names   : list[str]  ['SR', 'SB', 'AF', 'ST', 'SVT']
#     device         : torch.device
#     fs             : int  (default 500)
#     lead_for_rpeak : int  lead de detect R-peak (default 1 = Lead II)
#     n_samples_max  : int or None  gioi han so mau
#     verbose        : bool

#     Returns
#     -------
#     all_results : list[dict]  ket qua per-sample
#     summary     : dict        thong ke per-label + overall
#     """
#     from visualize.attention_map import get_attention_weights

#     model.eval()
#     model = model.to(device)

#     if isinstance(y_test, torch.Tensor):
#         y_test_np = y_test.cpu().numpy()
#     else:
#         y_test_np = np.asarray(y_test)

#     all_results = []
#     sample_idx  = 0

#     for batch_signals, _ in tqdm(test_loader, desc="Danh gia dinh luong"):
#         batch_signals = batch_signals.to(device)

#         with torch.no_grad():
#             outputs = model(batch_signals)
#             if isinstance(outputs, tuple):
#                 outputs = outputs[0]
#             probs = torch.sigmoid(outputs).cpu().numpy()

#         for b in range(len(batch_signals)):
#             if n_samples_max and sample_idx >= n_samples_max:
#                 break

#             signal_tensor = batch_signals[b]
#             signal_np     = signal_tensor.cpu().numpy()
#             seq_len       = signal_np.shape[1]

#             gt_vec    = y_test_np[sample_idx]
#             pred_vec  = probs[b]
#             gt_idx    = int(np.argmax(gt_vec))
#             pred_idx  = int(np.argmax(pred_vec))
#             gt_name   = target_names[gt_idx]
#             pred_name = target_names[pred_idx]

#             try:
#                 attn_w = get_attention_weights(model, signal_tensor)
#             except Exception:
#                 sample_idx += 1
#                 continue

#             lead_sig = signal_np[lead_for_rpeak]
#             r_peaks  = detect_r_peaks(lead_sig, fs=fs)
#             valid    = len(r_peaks) >= 2

#             attn_up = np.interp(
#                 np.arange(seq_len),
#                 np.linspace(0, seq_len - 1, len(attn_w)),
#                 attn_w
#             )

#             qrs_ratio = None
#             if valid:
#                 qrs_ratio, _, _ = compute_attention_energy_in_qrs(
#                     attn_w, r_peaks, seq_len, fs=fs
#                 )

#             _, rr_cv, rr_stats = compute_rr_stats(r_peaks, fs=fs)

#             tls = None
#             if valid:
#                 tls, _, _ = compute_temporal_localization_score(
#                     attn_up, r_peaks, gt_name, fs=fs, signal_len=seq_len
#                 )

#             all_results.append({
#                 'sample_idx'       : sample_idx,
#                 'gt_label'         : gt_name,
#                 'pred_label'       : pred_name,
#                 'correct'          : (gt_idx == pred_idx),
#                 'pred_conf'        : round(float(pred_vec[pred_idx]), 4),
#                 'n_r_peaks'        : len(r_peaks),
#                 'qrs_energy_ratio' : round(qrs_ratio, 4) if qrs_ratio is not None else None,
#                 'tls'              : round(tls, 4)        if tls       is not None else None,
#                 'rr_mean_ms'       : rr_stats.get('rr_mean_ms'),
#                 'rr_cv_pct'        : rr_stats.get('rr_cv_pct'),
#                 'heart_rate'       : rr_stats.get('heart_rate'),
#             })
#             sample_idx += 1

#         if n_samples_max and sample_idx >= n_samples_max:
#             break

#     summary = _build_summary(all_results, target_names, verbose=verbose)
#     return all_results, summary


# def _build_summary(all_results, target_names, verbose=True):
#     valid = [r for r in all_results if r['qrs_energy_ratio'] is not None]

#     def safe_mean(lst):
#         lst = [x for x in lst if x is not None]
#         return float(np.mean(lst)) if lst else 0.0

#     summary = {
#         'overall': {
#             'n_samples'           : len(all_results),
#             'n_valid'             : len(valid),
#             'accuracy_pct'        : round(np.mean([r['correct'] for r in all_results]) * 100, 2),
#             'qrs_energy_mean_pct' : round(safe_mean([r['qrs_energy_ratio'] for r in valid]) * 100, 2),
#             'tls_mean_pct'        : round(safe_mean([r['tls']              for r in valid]) * 100, 2),
#         }
#     }

#     for label in target_names:
#         sub = [r for r in valid if r['gt_label'] == label]
#         if not sub:
#             continue
#         summary[label] = {
#             'n_samples'           : len(sub),
#             'accuracy_pct'        : round(np.mean([r['correct']          for r in sub]) * 100, 2),
#             'qrs_energy_pct'      : round(safe_mean([r['qrs_energy_ratio'] for r in sub]) * 100, 2),
#             'tls_pct'             : round(safe_mean([r['tls']              for r in sub]) * 100, 2),
#             'rr_cv_mean_pct'      : round(safe_mean([r['rr_cv_pct']       for r in sub]), 2),
#             'heart_rate_mean_bpm' : round(safe_mean([r['heart_rate']       for r in sub]), 1),
#         }

#     if verbose:
#         _print_summary_table(summary, target_names)
#     return summary


# def _print_summary_table(summary, target_names):
#     ov = summary['overall']
#     print("\n" + "=" * 68)
#     print("  KET QUA DANH GIA DINH LUONG ATTENTION MAP")
#     print("=" * 68)
#     print(f"  Tong mau  : {ov['n_samples']}  (hop le: {ov['n_valid']})")
#     print(f"  Accuracy  : {ov['accuracy_pct']}%")
#     print(f"  QRS Energy: {ov['qrs_energy_mean_pct']}%   (>50% la tot)")
#     print(f"  TLS Score : {ov['tls_mean_pct']}%   (>60% la tot)")
#     print()
#     print(f"  {'Label':<7} {'N':>5} {'Acc%':>7} {'QRS%':>7} {'TLS%':>7} {'RR-CV%':>8} {'HR(bpm)':>9}")
#     print("  " + "-" * 51)
#     for label in target_names:
#         if label not in summary:
#             continue
#         s = summary[label]
#         print(f"  {label:<7} {s['n_samples']:>5} {s['accuracy_pct']:>7.1f}"
#               f" {s['qrs_energy_pct']:>7.1f} {s['tls_pct']:>7.1f}"
#               f" {s['rr_cv_mean_pct']:>8.1f} {s['heart_rate_mean_bpm']:>9.1f}")
#     print()
#     print("  Ghi chu:")
#     print("  QRS%  > 50%  => Model tap trung dung vung QRS [OK]")
#     print("  TLS%  > 60%  => Attention khop voi dac trung lam sang [OK]")
#     print("  RR-CV  AF    => Nen cao (>10%) vi AF co RR khong deu")
#     print("  HR SVT/ST    => Nen cao (>100 bpm)")
#     print("=" * 68)


# # ══════════════════════════════════════════════════════════════
# # BLOCK 6 — VISUALIZE: BIEU DO TONG HOP
# # ══════════════════════════════════════════════════════════════

# def plot_quantitative_summary(summary, target_names, save_path=None):
#     """
#     4-subplot: Accuracy | QRS Energy | TLS | RR-CV — theo tung nhan.

#     Parameters
#     ----------
#     summary      : dict  tu evaluate_attention_quantitative()
#     target_names : list[str]
#     save_path    : str or None
#     """
#     labels = [l for l in target_names if l in summary]
#     acc    = [summary[l]['accuracy_pct']        for l in labels]
#     qrs    = [summary[l]['qrs_energy_pct']      for l in labels]
#     tls    = [summary[l]['tls_pct']             for l in labels]
#     rrcv   = [summary[l]['rr_cv_mean_pct']      for l in labels]
#     x      = np.arange(len(labels))
#     colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0'][:len(labels)]

#     fig, axes = plt.subplots(2, 2, figsize=(14, 9))
#     fig.suptitle("Danh Gia Dinh Luong Attention — ECG (SB/SR/AF/SVT/ST)",
#                  fontsize=14, fontweight='bold')

#     def _bar(ax, vals, title, threshold=None, overall=None):
#         bars = ax.bar(x, vals, color=colors, edgecolor='black', alpha=0.85, width=0.55)
#         ax.set_title(title, fontweight='bold', fontsize=11)
#         ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
#         ax.set_ylabel("%"); ax.set_ylim(0, max(vals + [10]) * 1.25)
#         ax.grid(axis='y', alpha=0.3)
#         if threshold is not None:
#             ax.axhline(threshold, color='orange', ls='--', lw=1.5,
#                        label=f'Threshold {threshold}%')
#         if overall is not None:
#             ax.axhline(overall, color='red', ls='--', lw=1.5,
#                        label=f'Overall {overall:.1f}%')
#         if threshold is not None or overall is not None:
#             ax.legend(fontsize=8)
#         for bar, v in zip(bars, vals):
#             ax.text(bar.get_x() + bar.get_width() / 2,
#                     bar.get_height() + 0.5,
#                     f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')

#     _bar(axes[0, 0], acc,
#          "Classification Accuracy (%)",
#          overall=summary['overall']['accuracy_pct'])

#     _bar(axes[0, 1], qrs,
#          "QRS Energy Ratio (%)\n<-- >50%: model nhin dung vung QRS",
#          threshold=50,
#          overall=summary['overall']['qrs_energy_mean_pct'])

#     _bar(axes[1, 0], tls,
#          "Temporal Localization Score (%)\n<-- >60%: attention khop dac trung lam sang",
#          threshold=60,
#          overall=summary['overall']['tls_mean_pct'])

#     _bar(axes[1, 1], rrcv,
#          "RR Interval CV (%) — Do Khong Deu Nhip Tim\n<-- AF nen cao nhat (>10%)",
#          threshold=10)

#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=200, bbox_inches='tight')
#         print(f"[OK] Luu: {save_path}")
#     plt.show()
#     return fig


# # ══════════════════════════════════════════════════════════════
# # BLOCK 7 — VISUALIZE: CHI TIET 1 MAU
# # ══════════════════════════════════════════════════════════════

# def plot_single_sample(signal_np, attn_weights, r_peaks,
#                        gt_label, pred_label, pred_conf,
#                        qrs_energy_ratio, tls, rr_stats,
#                        fs=500, lead_for_rpeak=1, save_path=None):
#     """
#     Ve day du 1 mau ECG voi cac metric dinh luong.
#     Gom: Lead II + R-peaks + QRS shading | Attention heatmap | RR bar | 6 leads | Metric box.

#     Parameters
#     ----------
#     signal_np        : np.ndarray  (leads, seq_len)
#     attn_weights     : np.ndarray  raw attention tu get_attention_weights()
#     r_peaks          : np.ndarray
#     gt_label         : str
#     pred_label       : str
#     pred_conf        : float  [0,1]
#     qrs_energy_ratio : float
#     tls              : float
#     rr_stats         : dict  tu compute_rr_stats()
#     fs               : int
#     lead_for_rpeak   : int
#     save_path        : str or None
#     """
#     n_leads, seq_len = signal_np.shape
#     time = np.arange(seq_len) / fs
#     lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

#     attn_up = np.interp(
#         np.arange(seq_len),
#         np.linspace(0, seq_len - 1, len(attn_weights)),
#         attn_weights
#     )
#     high_attn   = attn_up > np.quantile(attn_up, 0.8)
#     qrs_windows = get_qrs_windows(r_peaks, fs=fs)
#     correct     = gt_label == pred_label
#     status      = "[DUNG]" if correct else "[SAI]"
#     title_color = '#1B5E20' if correct else '#B71C1C'

#     fig = plt.figure(figsize=(18, 14))
#     gs  = fig.add_gridspec(4, 3, hspace=0.5, wspace=0.38)

#     # Lead II chinh
#     ax0       = fig.add_subplot(gs[0, :2])
#     lead_main = signal_np[lead_for_rpeak]
#     ax0.plot(time, lead_main, 'k-', lw=1.0, zorder=2)
#     for (s, e) in qrs_windows:
#         ax0.axvspan(time[s], time[min(e, seq_len - 1)],
#                     alpha=0.15, color='green', label='QRS window')
#     if len(r_peaks) > 0:
#         ax0.scatter(time[r_peaks], lead_main[r_peaks],
#                     color='red', s=60, zorder=5, marker='v', label='R-peaks')
#     ax0.scatter(time[high_attn], lead_main[high_attn],
#                 color='blue', s=8, alpha=0.5, zorder=3, label='High Attention (top 20%)')
#     ax0.set_title(
#         f"Lead II  |  GT: {gt_label}  ->  Pred: {pred_label} ({pred_conf * 100:.1f}%)  {status}",
#         fontweight='bold', fontsize=12, color=title_color
#     )
#     ax0.set_ylabel("Z-score"); ax0.set_xlabel("Time (s)")
#     ax0.grid(alpha=0.3)
#     h, l = ax0.get_legend_handles_labels()
#     ax0.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(),
#                fontsize=8, loc='upper right')

#     # Metric box
#     ax_m = fig.add_subplot(gs[0, 2])
#     ax_m.axis('off')
#     txt = (
#         "METRICS\n"
#         + "-" * 26 + "\n"
#         + f"QRS Energy : {qrs_energy_ratio * 100:.1f}%\n"
#         + f"TLS Score  : {tls * 100:.1f}%\n"
#         + "-" * 26 + "\n"
#         + f"RR Mean    : {rr_stats.get('rr_mean_ms', 'N/A')} ms\n"
#         + f"RR Std     : {rr_stats.get('rr_std_ms',  'N/A')} ms\n"
#         + f"RR CV      : {rr_stats.get('rr_cv_pct',  'N/A')} %\n"
#         + f"Heart Rate : {rr_stats.get('heart_rate',  'N/A')} bpm\n"
#         + f"N beats    : {rr_stats.get('n_beats',     'N/A')}\n"
#     )
#     ax_m.text(0.05, 0.97, txt, transform=ax_m.transAxes,
#               fontsize=10, va='top', fontfamily='monospace',
#               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

#     # Attention heatmap
#     ax_a = fig.add_subplot(gs[1, :2])
#     ax_a.fill_between(time, attn_up, color='crimson', alpha=0.7)
#     for (s, e) in qrs_windows:
#         ax_a.axvspan(time[s], time[min(e, seq_len - 1)], alpha=0.2, color='green')
#     ax_a.set_title("Attention Weight theo Thoi Gian  (xanh la = vung QRS)",
#                    fontweight='bold')
#     ax_a.set_ylabel("Attention"); ax_a.set_xlabel("Time (s)")
#     ax_a.grid(alpha=0.3)

#     # RR bar chart
#     ax_rr = fig.add_subplot(gs[1, 2])
#     rr_ms, rr_mean, _ = get_rr_intervals(r_peaks, fs=fs)
#     if len(rr_ms) > 0:
#         bar_c = ['#E53935' if abs(rr - rr_mean) > 0.1 * rr_mean
#                  else '#43A047' for rr in rr_ms]
#         ax_rr.bar(range(len(rr_ms)), rr_ms, color=bar_c, edgecolor='black', alpha=0.85)
#         ax_rr.axhline(rr_mean, color='navy', ls='--', lw=1.5,
#                       label=f'Mean {rr_mean:.0f} ms')
#         ax_rr.set_title("RR Intervals  (do = lech >10%)", fontweight='bold')
#         ax_rr.set_xlabel("Beat index"); ax_rr.set_ylabel("RR (ms)")
#         ax_rr.legend(fontsize=8); ax_rr.grid(axis='y', alpha=0.3)
#     else:
#         ax_rr.text(0.5, 0.5, 'Khong du R-peaks', ha='center', va='center')
#         ax_rr.axis('off')

#     # 6 leads phu
#     show = min(6, n_leads)
#     for i in range(show):
#         row = 2 + i // 3
#         col = i % 3
#         ax_l = fig.add_subplot(gs[row, col])
#         ld   = signal_np[i]
#         ax_l.plot(time, ld, 'k-', lw=0.7)
#         ax_l.scatter(time[high_attn], ld[high_attn],
#                      color='blue', s=4, alpha=0.4)
#         lname = lead_names[i] if i < len(lead_names) else f'L{i+1}'
#         ax_l.set_title(f"Lead {lname}", fontsize=9, fontweight='bold')
#         ax_l.set_ylabel("Z-score", fontsize=7)
#         ax_l.tick_params(labelsize=7)
#         ax_l.grid(alpha=0.3)

#     fig.suptitle(f"Chi Tiet Mau  |  GT: {gt_label}  |  Pred: {pred_label}  {status}",
#                  fontsize=13, fontweight='bold', color=title_color)

#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         print(f"[OK] Luu: {save_path}")
#     plt.show()
#     return fig


# # ══════════════════════════════════════════════════════════════
# # BLOCK 8 — SAVE RESULTS TO CSV
# # ══════════════════════════════════════════════════════════════

# def save_results_csv(all_results, save_path="quantitative_results.csv"):
#     """
#     Luu toan bo ket qua per-sample ra CSV.

#     Parameters
#     ----------
#     all_results : list[dict]  tu evaluate_attention_quantitative()
#     save_path   : str
#     """
#     import csv
#     if not all_results:
#         print("Khong co ket qua de luu.")
#         return
#     fieldnames = list(all_results[0].keys())
#     with open(save_path, 'w', newline='', encoding='utf-8') as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(all_results)
#     print(f"[OK] Da luu {len(all_results)} dong vao: {save_path}")
