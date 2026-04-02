###NEW 280226 ####
"""
attention_map.py — Trực quan Attention cho ResNet18_LSTM_Attn
=============================================================

KIẾN TRÚC MODEL:
    Input (12, 5000)
        → ResNet18 backbone  → downsample ~32x
        → LSTM               → seq_len = 157 timesteps
        → Attention layer    → output[1]: (1, 157, 1)   ← attention scores
        → FC                 → (num_classes,)

VẤN ĐỀ GỐC (đã fix):
    Mỗi attention timestep tương đương ~32 samples tín hiệu gốc.
    QRS complex rộng ~50ms = 25 samples < 1 LSTM timestep.
    → Không thể "khoanh" đúng QRS bằng attention ở resolution này.
    → Giải pháp: upsample bằng Gaussian kernel thay vì linear interp
      để giữ đỉnh attention sắc nét hơn khi map về signal domain.

CÁCH DÙNG:
    from visualize.attention_map import get_attention_weights, plot_attention_dots
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


# ══════════════════════════════════════════════════════════════
# BLOCK 1 — LẤY ATTENTION WEIGHTS
# ══════════════════════════════════════════════════════════════

def get_attention_weights(model, signal, upsample_method='gaussian', return_raw=False):
    """
    Lấy attention weights từ ResNet18_LSTM_Attn model.

    Kiến trúc model:
        signal (12, 5000) → ResNet18 → LSTM → Attention → (context, scores)
        scores shape: (1, 157, 1) → downsample ~32x từ signal gốc

    Parameters
    ----------
    model           : ResNet18_LSTM_Attn đã load
    signal          : torch.Tensor  shape (12, seq_len)
    upsample_method : str
        'gaussian'  — Gaussian kernel (mặc định, sắc nét hơn ở đỉnh)
        'linear'    — Linear interpolation (mượt mà, phân tán hơn)
        'nearest'   — Nearest neighbor (giữ bậc thang, dễ thấy segment)
    return_raw      : bool  nếu True, trả về thêm raw weights (157,)

    Returns
    -------
    weights_up  : np.ndarray  shape (seq_len,) chuẩn hóa [0,1]
    weights_raw : np.ndarray  shape (157,)     — chỉ trả về nếu return_raw=True
    """
    model.eval()
    _captured = {}

    def hook_fn(module, input, output):
        # Attention layer trả về tuple: (context_vector, attention_scores)
        # attention_scores shape: (batch, seq_len_lstm, 1) = (1, 157, 1)
        if isinstance(output, tuple) and len(output) > 1:
            _captured['scores'] = output[1].detach().cpu()

    handle = model.attention.register_forward_hook(hook_fn)
    device = next(model.parameters()).device

    with torch.no_grad():
        _ = model(signal.unsqueeze(0).to(device))
    handle.remove()

    # --- Lấy raw scores (157,) ---
    if 'scores' not in _captured:
        # Fallback nếu hook không bắt được
        seq_len = signal.shape[1]
        weights_up = np.ones(seq_len) / seq_len
        if return_raw:
            return weights_up, weights_up
        return weights_up

    raw = _captured['scores'].squeeze().numpy()   # (157,)

    # Normalize raw về [0, 1]
    raw_min, raw_max = raw.min(), raw.max()
    if raw_max > raw_min:
        raw_norm = (raw - raw_min) / (raw_max - raw_min + 1e-9)
    else:
        raw_norm = np.ones_like(raw) / len(raw)

    # --- Upsample 157 → 5000 ---
    signal_len = signal.shape[1]
    lstm_len   = len(raw_norm)

    if upsample_method == 'gaussian':
        weights_up = _upsample_gaussian(raw_norm, signal_len)
    elif upsample_method == 'nearest':
        weights_up = _upsample_nearest(raw_norm, signal_len)
    else:  # linear
        weights_up = np.interp(
            np.arange(signal_len),
            np.linspace(0, signal_len - 1, lstm_len),
            raw_norm
        )

    # Normalize lại sau upsample
    w_min, w_max = weights_up.min(), weights_up.max()
    if w_max > w_min:
        weights_up = (weights_up - w_min) / (w_max - w_min + 1e-9)

    if return_raw:
        return weights_up, raw_norm
    return weights_up


def _upsample_gaussian(raw_norm, target_len, sigma_factor=0.6):
    """
    Upsample bằng cách đặt Gaussian tại vị trí tương ứng trong signal domain.
    Mỗi attention timestep được "trải" ra thành một Gaussian hẹp,
    giữ được peak location tốt hơn linear interp.

    sigma_factor: độ rộng Gaussian, đơn vị = 1 LSTM timestep
                  0.5 = hẹp (sắc nét) | 1.0 = rộng (mượt)
    """
    lstm_len   = len(raw_norm)
    result     = np.zeros(target_len, dtype=np.float32)
    step       = target_len / lstm_len                      # ~31.8 samples / timestep
    sigma      = sigma_factor * step                        # Gaussian width in samples

    # Đặt từng attention score vào vị trí tương ứng trên trục signal
    for i, w in enumerate(raw_norm):
        center = (i + 0.5) * step                          # center sample index
        result[int(min(center, target_len - 1))] = w

    # Làm mượt bằng Gaussian filter
    result = gaussian_filter1d(result, sigma=sigma)
    return result


def _upsample_nearest(raw_norm, target_len):
    """Nearest neighbor: mỗi sample trong signal domain lấy giá trị
    của LSTM timestep gần nhất."""
    lstm_len = len(raw_norm)
    indices  = (np.arange(target_len) * lstm_len / target_len).astype(int)
    indices  = np.clip(indices, 0, lstm_len - 1)
    return raw_norm[indices]


# ══════════════════════════════════════════════════════════════
# BLOCK 2 — VẼ ECG + ATTENTION
# ══════════════════════════════════════════════════════════════

def plot_attention_dots(signal, attn_weights=None, attn_weights_raw=None,
                        sample_idx=None, label_true=None, label_pred=None,
                        fs=500, paper_speed=25, show_rr=True, lead_names=None,
                        show_attention_panel=True):
    """
    Vẽ ECG 12-lead với attention overlay + panel attention gốc (157 timestep).

    Parameters
    ----------
    signal             : torch.Tensor hoặc np.ndarray  shape (12, seq_len)
    attn_weights       : np.ndarray  shape (seq_len,)  đã upsample — từ get_attention_weights()
    attn_weights_raw   : np.ndarray  shape (157,)      raw attention — dùng return_raw=True
    sample_idx         : int    hiển thị trên title
    label_true         : str    nhãn thật
    label_pred         : str    nhãn dự đoán (nếu có)
    fs                 : int    tần số lấy mẫu (Hz)
    paper_speed        : int    25 hoặc 50 mm/s
    show_rr            : bool   vẽ R-peaks
    show_attention_panel : bool vẽ thêm 1 panel attention heatmap phía dưới
    """
    if lead_names is None:
        n = signal.shape[0]
        if n == 12:
            lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
        else:
            lead_names = [f'L{i+1}' for i in range(n)]

    # Chuyển sang numpy nếu cần
    if torch.is_tensor(signal):
        signal_np = signal.cpu().numpy()
    else:
        signal_np = np.asarray(signal)

    n_leads, L = signal_np.shape
    time = np.arange(L) / fs

    # Grid chuẩn ECG
    if paper_speed == 25:
        small_box_s, big_box_s = 0.04, 0.2
    else:
        small_box_s, big_box_s = 0.02, 0.1

    # Chuẩn bị attention upsampled
    if attn_weights is not None:
        attn_len = len(attn_weights)
        if attn_len != L:
            attn_up = np.interp(np.arange(L), np.linspace(0, L - 1, attn_len), attn_weights)
        else:
            attn_up = np.asarray(attn_weights, dtype=float)
        threshold_attn = np.quantile(attn_up, 0.80)   # top 20%
        high_attn_idx  = np.where(attn_up > threshold_attn)[0]
    else:
        attn_up       = None
        high_attn_idx = np.array([], dtype=int)

    # ── Layout ──
    n_extra_rows = 1 if show_attention_panel else 0
    total_rows   = n_leads + n_extra_rows
    fig_height   = 2.2 * n_leads + (2.5 if show_attention_panel else 0)

    fig = plt.figure(figsize=(18, fig_height))
    height_ratios = [2.2] * n_leads + ([2.5] if show_attention_panel else [])
    gs = gridspec.GridSpec(
        total_rows, 1,
        figure=fig,
        hspace=0.25,
        height_ratios=height_ratios
    )

    axes_leads = [fig.add_subplot(gs[i]) for i in range(n_leads)]

    # ── Vẽ từng lead ──
    for i, ax in enumerate(axes_leads):
        lead = signal_np[i]
        ax.set_facecolor('white')
        ax.set_ylabel(lead_names[i], rotation=0, labelpad=38,
                      fontsize=11, fontweight='bold')

        y_min, y_max = lead.min(), lead.max()
        margin = (y_max - y_min) * 0.18
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_xlim(time[0], time[-1])
        ax.tick_params(labelbottom=False, labelsize=7)

        # Lưới thời gian
        ax.set_xticks(np.arange(0, time[-1] + big_box_s, big_box_s))
        ax.grid(which='major', axis='x', color='#e08080', linewidth=0.7, alpha=0.6)
        ax.set_xticks(np.arange(0, time[-1] + small_box_s, small_box_s), minor=True)
        ax.grid(which='minor', axis='x', color='#f0b0b0', linewidth=0.3, alpha=0.4)
        ax.grid(True, axis='y', linestyle='--', alpha=0.25, color='gray')

        # Vẽ attention heatmap (fill dưới đường, màu cam nhạt)
        if attn_up is not None and i == 1:   # chỉ tô Lead II để không rối
            attn_scaled = attn_up / attn_up.max() * (y_max - y_min) * 0.25 + y_min - margin
            ax.fill_between(time, attn_scaled, y_min - margin,
                            color='#FF8C00', alpha=0.25, zorder=1,
                            label='Attention (Lead II overlay)')

        # Chấm attention cao (top 20%)
        if len(high_attn_idx) > 0 and attn_up is not None:
            ax.scatter(
                time[high_attn_idx], lead[high_attn_idx],
                color='#1565C0', alpha=0.55, s=10,
                zorder=3, label='High Attention (top 20%)'
            )

        # Sóng ECG
        ax.plot(time, lead, color='black', linewidth=1.0, zorder=2)

        # R-peaks
        if show_rr:
            r_peaks = _detect_r_peaks(lead, fs)
            if len(r_peaks) > 0:
                ax.scatter(
                    time[r_peaks], lead[r_peaks],
                    color='red', s=45, marker='v',
                    zorder=4, label='R-peaks', alpha=0.8
                )

        # Legend chỉ ở lead đầu
        if i == 0:
            h, l = ax.get_legend_handles_labels()
            by_label = dict(zip(l, h))
            if by_label:
                ax.legend(by_label.values(), by_label.keys(),
                          loc='upper right', fontsize=8, framealpha=0.7)

    axes_leads[-1].tick_params(labelbottom=True)
    axes_leads[-1].set_xlabel("Time (s)", fontsize=11, fontweight='bold')

    # ── Panel attention gốc (157 timestep) ──
    if show_attention_panel:
        ax_attn = fig.add_subplot(gs[n_leads])
        ax_attn.set_facecolor('#f8f8f8')

        if attn_weights_raw is not None:
            # Vẽ raw 157-point attention
            lstm_len   = len(attn_weights_raw)
            lstm_time  = np.linspace(0, L / fs, lstm_len)
            ax_attn.bar(lstm_time, attn_weights_raw,
                        width=(L / fs) / lstm_len * 0.85,
                        color='#FF6B35', alpha=0.8, label='Attention (157 LSTM timesteps)')
            ax_attn.set_xlim(time[0], time[-1])
            ax_attn.set_ylim(0, attn_weights_raw.max() * 1.2)
            ax_attn.set_ylabel("Attention\n(raw)", fontsize=9)
            ax_attn.legend(fontsize=8, loc='upper right')

            # Annotate top-3 peaks của attention
            top3_idx = np.argsort(attn_weights_raw)[-3:]
            for idx in top3_idx:
                t_center = lstm_time[idx]
                ax_attn.annotate(
                    f't={t_center:.2f}s\n({attn_weights_raw[idx]:.3f})',
                    xy=(t_center, attn_weights_raw[idx]),
                    xytext=(t_center, attn_weights_raw[idx] * 1.05),
                    fontsize=7, ha='center', color='#B71C1C', fontweight='bold'
                )

        elif attn_up is not None:
            # Fallback: vẽ upsampled
            ax_attn.fill_between(time, attn_up, color='#FF6B35', alpha=0.7,
                                 label='Attention (upsampled)')
            ax_attn.set_xlim(time[0], time[-1])
            ax_attn.legend(fontsize=8, loc='upper right')
            ax_attn.set_ylabel("Attention", fontsize=9)

        # Chú thích giải hạn chế
        note = (
            "⚠ Attention ở mức LSTM (157 timestep, ~32ms/step) — "
            "không đủ resolution để khoanh vùng QRS (~50-100ms).\n"
            "Sử dụng để xem model tập trung vào VÙNG NÀO của tín hiệu "
            "(đầu/giữa/cuối, thưa/dày nhịp), không phải chi tiết sóng P/Q/R/S/T."
        )
        ax_attn.set_title(note, fontsize=8, color='#555555', style='italic', pad=4)
        ax_attn.set_xlabel("Time (s)", fontsize=10)
        ax_attn.grid(axis='y', alpha=0.3)
        ax_attn.tick_params(labelsize=8)

    # ── Title tổng ──
    status_str = ""
    title_color = 'black'
    if label_true and label_pred:
        if label_true == label_pred:
            status_str = f"  ✓ ĐÚNG  (GT: {label_true} | Pred: {label_pred})"
            title_color = '#1B5E20'
        else:
            status_str = f"  ✗ SAI   (GT: {label_true} | Pred: {label_pred})"
            title_color = '#B71C1C'
    elif label_true:
        status_str = f"  Nhãn: {label_true}"

    title = f"ECG 12-lead + Attention Map  |  {paper_speed} mm/s"
    if sample_idx is not None:
        title += f"  |  Sample #{sample_idx}"
    title += status_str

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.002, color=title_color)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# BLOCK 3 — TIỆN ÍCH
# ══════════════════════════════════════════════════════════════

def _detect_r_peaks(signal_1d, fs=500):
    """Detect R-peaks từ 1 lead đã Z-score."""
    peaks, _ = find_peaks(
        signal_1d,
        height=0.8,
        distance=int(fs * 0.45),
        prominence=0.6
    )
    return peaks


def plot_attention_comparison(signal, model, fs=500, lead_for_rpeak=1):
    """
    So sánh 3 phương pháp upsample cạnh nhau:
    gaussian | linear | nearest
    Hữu ích để chọn method phù hợp với từng mẫu.

    Parameters
    ----------
    signal         : torch.Tensor  (12, seq_len)
    model          : ResNet18_LSTM_Attn
    fs             : int
    lead_for_rpeak : int  Lead để vẽ (mặc định = 1, Lead II)
    """
    methods = ['gaussian', 'linear', 'nearest']
    results = {}
    for m in methods:
        up, raw = get_attention_weights(model, signal,
                                        upsample_method=m, return_raw=True)
        results[m] = up
    raw_w = raw   # raw giống nhau cho mọi method

    if torch.is_tensor(signal):
        sig_np = signal.cpu().numpy()
    else:
        sig_np = np.asarray(signal)

    lead  = sig_np[lead_for_rpeak]
    L     = len(lead)
    time  = np.arange(L) / fs

    r_peaks = _detect_r_peaks(lead, fs)

    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)

    # --- Panel 0: Lead II tham chiếu ---
    ax = axes[0]
    ax.plot(time, lead, 'k-', lw=1.0)
    if len(r_peaks) > 0:
        ax.scatter(time[r_peaks], lead[r_peaks],
                   color='red', s=60, marker='v', zorder=5, label='R-peaks')
    ax.set_title("Lead II (tín hiệu tham chiếu)", fontweight='bold')
    ax.set_ylabel("Z-score"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # --- Panel 1-3: So sánh 3 methods ---
    colors = ['#FF6B35', '#1565C0', '#2E7D32']
    for j, (method, color) in enumerate(zip(methods, colors)):
        ax = axes[j + 1]
        attn_up = results[method]
        ax.fill_between(time, attn_up, color=color, alpha=0.6)
        # Đánh dấu R-peaks trên attention
        if len(r_peaks) > 0:
            ax.scatter(time[r_peaks], attn_up[r_peaks],
                       color='red', s=40, marker='v', zorder=5)
        # QRS windows
        qrs_pre  = int(0.06 * fs)
        qrs_post = int(0.08 * fs)
        for r in r_peaks:
            s = max(0, r - qrs_pre)
            e = min(L - 1, r + qrs_post)
            ax.axvspan(time[s], time[e], alpha=0.15, color='green')
        ax.set_title(f"Upsample method: '{method}'  — xanh lá = vùng QRS",
                     fontweight='bold', color=color)
        ax.set_ylabel("Attention"); ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time (s)", fontsize=11)
    fig.suptitle(
        "So sánh 3 phương pháp upsample attention (157 → 5000)\n"
        "Vùng xanh lá = QRS window (±60/80ms quanh R-peak)",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    return fig

#####160126####

# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.signal import find_peaks

# def get_attention_weights(model, signal):
#     """
#     Lấy attention weights từ ResNet18_LSTM_Attn model.
#     - signal: tensor [leads, length] (12, seq_len)
    
#     Returns:
#         weights: numpy array chuẩn hóa [0,1]
#     """
#     model.eval()
#     attention_weights = {}

#     def hook_fn(module, input, output):
#         # output từ Attention layer: (context, weights)
#         if isinstance(output, tuple) and len(output) > 1:
#             attention_weights['value'] = output[1].detach().cpu()

#     # Đăng ký forward hook tại layer attention
#     handle = model.attention.register_forward_hook(hook_fn)
    
#     device = next(model.parameters()).device
    
#     with torch.no_grad():
#         # Forward pass
#         signal_input = signal.unsqueeze(0).to(device)  # (1, 12, seq_len)
#         _ = model(signal_input)
    
#     handle.remove()

#     # Lấy weights, chuẩn hóa về [0,1]
#     weights = attention_weights.get('value', torch.zeros(signal.shape[1]))
#     weights = weights.squeeze().numpy()
    
#     # Normalize
#     if weights.max() > weights.min():
#         weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
    
#     return weights


# def plot_attention_dots(signal, attn_weights=None, sample_idx=None, label_true=None,
#                         fs=500, paper_speed=25, show_rr=True, lead_names=None):
#     """
#     Vẽ sóng ECG 12-lead.
#     - Thời gian: Chuẩn 25mm/s (hoặc 50mm/s tùy chọn).
#     - Biên độ: Tự động thích ứng Z-score hoặc mV.
    
#     Args:
#         signal: tensor [leads, length] (đã Z-score hoặc mV)
#         attn_weights: attention [seq_len] chuẩn hóa 0-1
#         fs: tần số lấy mẫu (Hz)
#         paper_speed: mm/s (Default 25 cho chuẩn y khoa thông dụng)
#         show_rr: bật/tắt hiển thị khoảng R-R
#     """
#     # Default lead names
#     if lead_names is None:
#         n_leads = signal.shape[0]
#         if n_leads == 12:
#             lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
#                          'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
#         elif n_leads == 6:
#             lead_names = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
#         else:
#             lead_names = [f'Lead {i+1}' for i in range(n_leads)]
    
#     n_leads, L = signal.shape
#     time = np.arange(L) / fs  # giây

#     # --- 1. Xử lý Attention Weights ---
#     if attn_weights is not None:
#         attn_len = len(attn_weights)
#         if attn_len != L:
#             # Interpolate nếu độ dài không khớp
#             attn_up = np.interp(
#                 np.arange(L), 
#                 np.linspace(0, L-1, attn_len), 
#                 attn_weights
#             )
#         else:
#             attn_up = attn_weights
        
#         # High attention indices (top 20% attention values)
#         high_attn_idx = np.where(attn_up > np.quantile(attn_up, 0.8))[0]

#     # --- 2. Cấu hình lưới (Grid) ---
#     # Thời gian: 25mm/s -> 1 ô nhỏ = 0.04s | 50mm/s -> 1 ô nhỏ = 0.02s
#     if paper_speed == 25:
#         small_box_s = 0.04
#         big_box_s = 0.2
#     else: # 50 mm/s
#         small_box_s = 0.02
#         big_box_s = 0.1

#     # Kiểm tra tín hiệu là Z-score hay mV?
#     # Z-score thường có mean~0, std~1, max thường < 10-20
#     # mV raw thường có thể lớn hơn nếu chưa scale, nhưng sau khi chia 1000 thì nhỏ (~1-2mV)
#     # Tuy nhiên, Z-score và mV (đã chia gain) có biên độ số học khá tương đồng (tầm -2 đến 2).
#     # ĐỂ AN TOÀN: Với Z-score, ta KHÔNG vẽ lưới đỏ mV chuẩn để tránh hiểu lầm.
#     # Ta sẽ vẽ lưới mờ (grid) thông thường.
    
#     # Ở đây mặc định coi như là Z-score (theo pipeline của bạn)
#     is_zscore = True 

#     fig, axes = plt.subplots(n_leads, 1, figsize=(16, 2.5*n_leads), sharex=True)
#     if n_leads == 1:
#         axes = [axes]

#     for i in range(n_leads):
#         lead = signal[i].cpu().numpy() if torch.is_tensor(signal[i]) else signal[i]
#         ax = axes[i]

#         ax.set_facecolor('white')
        
#         # Label trục Y
#         ylabel = lead_names[i]
#         # if is_zscore:
#         #     ylabel += "\n(Z-score)"
#         ax.set_ylabel(ylabel, rotation=0, labelpad=40, fontsize=12, fontweight='bold')

#         # Giới hạn trục Y: Tự động theo biên độ tín hiệu
#         y_min, y_max = lead.min(), lead.max()
#         margin = (y_max - y_min) * 0.15
#         ax.set_ylim(y_min - margin, y_max + margin)
#         ax.set_xlim(time[0], time[-1])

#         # --- VẼ LƯỚI ---
#         # 1. Lưới Dọc (Thời gian) - Luôn vẽ theo chuẩn 25mm/s hoặc 50mm/s
#         # Major ticks (0.2s)
#         ax.set_xticks(np.arange(0, time[-1], big_box_s))
#         ax.grid(which='major', axis='x', color='red', linewidth=0.8, alpha=0.5)
#         # Minor ticks (0.04s)
#         ax.set_xticks(np.arange(0, time[-1], small_box_s), minor=True)
#         ax.grid(which='minor', axis='x', color='pink', linewidth=0.3, alpha=0.5)

#         # 2. Lưới Ngang (Biên độ)
#         if is_zscore:
#             # Nếu là Z-score: Vẽ lưới nét đứt mờ, không theo chuẩn 0.1mV/0.5mV
#             ax.grid(True, axis='y', linestyle='--', alpha=0.3, color='gray')
#         else:
#             # Nếu là mV: Vẽ lưới chuẩn y tế
#             # (Code này để dành nếu sau này bạn vẽ tín hiệu raw)
#             small_box_mV = 0.1; big_box_mV = 0.5
#             ax.set_yticks(np.arange(np.floor(y_min), np.ceil(y_max), big_box_mV))
#             ax.grid(which='major', axis='y', color='red', linewidth=0.8, alpha=0.5)
#             ax.set_yticks(np.arange(np.floor(y_min), np.ceil(y_max), small_box_mV), minor=True)
#             ax.grid(which='minor', axis='y', color='pink', linewidth=0.3, alpha=0.5)

#         # --- VẼ ATTENTION (Màu xanh) ---
#         if attn_weights is not None:
#             ax.scatter(
#                 time[high_attn_idx],
#                 lead[high_attn_idx],
#                 color='blue', alpha=0.6, s=15, 
#                 zorder=3, label='High Attention'
#             )

#         # --- VẼ SÓNG ECG ---
#         ax.plot(time, lead, color='black', linewidth=1.2, zorder=2)

#         # --- R-PEAKS & R-R INTERVAL ---
#         if show_rr:
#             # Tinh chỉnh tham số find_peaks cho Z-score
#             # Z-score: mean ~ 0, std ~ 1. Đỉnh R thường > 1.5 - 2.0
#             peaks, _ = find_peaks(
#                 lead,
#                 distance=int(fs*0.5),      # > 500ms (nhịp < 120 bpm)
#                 height=1.0,                # Z-score > 1.0 (thấp để bắt nhịp yếu)
#                 prominence=0.8             # Nổi bật so với nền
#             )
            
#             if len(peaks) > 0:
#                 ax.scatter(time[peaks], lead[peaks], color='red', s=50, 
#                           marker='x', zorder=4, label='R peaks', linewidths=2)

#             # Vẽ đo khoảng R-R
#             # for j in range(len(peaks) - 1):
#             #     t1, t2 = time[peaks[j]], time[peaks[j + 1]]
#             #     rr_interval = t2 - t1
                
#             #     # Tính số ô nhỏ (theo tốc độ giấy)
#             #     n_small_boxes = rr_interval / small_box_s
                
#             #     mid_t = (t1 + t2) / 2
#             #     # Đặt text phía trên đỉnh cao nhất trong cặp
#             #     y_text = max(lead[peaks[j]], lead[peaks[j+1]]) + margin*0.8

#             #     ax.text(mid_t, y_text,
#             #             f"{n_small_boxes:.1f} ô\n({rr_interval*1000:.0f} ms)",
#             #             ha='center', va='bottom', fontsize=9, color='purple', 
#             #             fontweight='bold', zorder=5)

#             #     # Đường nối đứt đoạn
#             #     ax.plot([t1, t2], [lead[peaks[j]], lead[peaks[j+1]]],
#             #             linestyle='--', color='purple', alpha=0.5, linewidth=1.0)
        
#         # Legend (chỉ hiện ở lead đầu tiên)
#         if i == 0 and (attn_weights is not None or show_rr):
#             # Tạo dummy handles để legend đẹp hơn
#             handles, labels = ax.get_legend_handles_labels()
#             # Lọc bớt duplicate nếu có
#             by_label = dict(zip(labels, handles))
#             ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)

#     axes[-1].set_xlabel("Time (s)", fontsize=12, fontweight='bold')

#     # Title
#     title = f"ECG Attention Map with R-R Measurement (Paper Speed: {paper_speed} mm/s)"
#     if sample_idx is not None: title += f" | Sample #{sample_idx}"
#     if label_true is not None: title += f" | Label: {label_true}"
    
#     fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
#     plt.tight_layout(rect=[0, 0, 1, 0.99])
    
#     return fig 



######### OLD ############
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.signal import find_peaks

# def get_attention_weights(model, signal):
#     """
#     Lấy attention weights từ attention layer của model.
#     - signal: tensor [leads, length]
#     """
#     model.eval()
#     attention_weights = {}

#     def hook_fn(module, input, output):
#         # output: (context, weights)
#         if isinstance(output, tuple) and len(output) > 1:
#             attention_weights['value'] = output[1].detach().cpu()

#     # Đăng ký forward hook tại layer attention
#     handle = model.attention.register_forward_hook(hook_fn)
#     with torch.no_grad():
#         _ = model(signal.unsqueeze(0).to(next(model.parameters()).device))
#     handle.remove()

#     # Lấy weights, chuẩn hóa về [0,1]
#     weights = attention_weights.get('value', torch.zeros(signal.shape[1]))
#     weights = weights.squeeze().numpy()
#     weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
#     return weights


# def plot_attention_dots(signal, attn_weights=None, sample_idx=None, label_true=None,
#                         fs=500, paper_speed=50, show_rr=True):
#     """
#     Vẽ sóng ECG 12-lead với lưới chuẩn (50 mm/s, 10 mm/mV), chấm đỏ attention và đo khoảng R-R.
#     - signal: tensor [leads, length] (mV)
#     - attn_weights: attention [seq_len] chuẩn hóa 0-1
#     - fs: tần số lấy mẫu (Hz)
#     - paper_speed: mm/s (50mm/s chuẩn)
#     - show_rr: bật/tắt hiển thị khoảng R-R (True/False)
#     """
#     lead_names = ['V1','V2','V3','V4','V5','V6']
#     n_leads, L = signal.shape
#     time = np.arange(L) / fs  # giây

#     # Upsample attention về đúng chiều
#     if attn_weights is not None:
#         attn_up = np.interp(np.arange(L), np.linspace(0, L-1, len(attn_weights)), attn_weights)
#         high_attn_idx = np.where(attn_up > np.quantile(attn_up, 0.8))[0]

#     # --- Thông số lưới chuẩn ECG ---
#     small_box_s = 0.02  # 1 mm = 0.02 s @ 50 mm/s
#     big_box_s   = 0.1   # 5 mm = 0.1 s
#     small_box_mV = 0.1  # 1 ô nhỏ = 0.1 mV
#     big_box_mV   = 0.5  # 1 ô lớn = 0.5 mV

#     fig, axes = plt.subplots(n_leads, 1, figsize=(15, 2*n_leads), sharex=True)
#     if n_leads == 1:
#         axes = [axes]

#     for i in range(n_leads):
#         lead = signal[i].cpu().numpy()
#         ax = axes[i]

#         # Vẽ nền và nhãn
#         ax.set_facecolor('white')
#         ax.set_ylabel(lead_names[i] if i<len(lead_names) else f"L{i+1}", rotation=0, labelpad=30)

#         y_min, y_max = lead.min()*1.2, lead.max()*1.2
#         ax.set_ylim(y_min, y_max)
#         ax.set_xlim(time[0], time[-1])

#         # --- Lưới dọc (thời gian) ---
#         for t in np.arange(time[0], time[-1], small_box_s):
#             ax.axvline(t, color='lightpink', linewidth=0.3, zorder=1)
#         for t in np.arange(time[0], time[-1], big_box_s):
#             ax.axvline(t, color='red', linewidth=0.8, zorder=1)

#         # --- Lưới ngang (biên độ) ---
#         for y in np.arange(y_min, y_max, small_box_mV):
#             ax.axhline(y, color='lightpink', linewidth=0.3, zorder=1)
#         for y in np.arange(y_min, y_max, big_box_mV):
#             ax.axhline(y, color='red', linewidth=0.8, zorder=1)
            
#         # --- Chấm attention ---
#         if attn_weights is not None:
#             ax.scatter(
#                 time[high_attn_idx],
#                 lead[high_attn_idx],
#                 color='blue', alpha=0.8, s=12, edgecolors='none', zorder=3
#             )

#         # --- Vẽ sóng ECG ---
#         ax.plot(time, lead, color='black', linewidth=0.9, zorder=2)

#         # --- Phát hiện đỉnh R & đo khoảng R-R ---
#         if show_rr:
#             peaks, props = find_peaks(
#                 lead,
#                 distance=fs*0.6,  # >= 600 ms giữa 2 đỉnh R (loại T)
#                 height=np.mean(lead) + 0.6*np.std(lead),  # cao hơn trung bình 
#                 prominence=0.8*np.std(lead)  # đỉnh phải “nổi bật” hơn vùng xung quanh
#             )
#             # Vẽ đỉnh R
#             ax.scatter(time[peaks], lead[peaks], color='blue', s=25, zorder=4, label='R peaks')

#             # Đánh dấu khoảng R-R
#             for j in range(len(peaks) - 1):
#                 t1, t2 = time[peaks[j]], time[peaks[j + 1]]
#                 rr_interval = t2 - t1  # giây
#                 small_boxes = rr_interval / small_box_s
#                 mid_t = (t1 + t2) / 2

#                 ax.text(mid_t, y_max * 0.9,
#                         f"{small_boxes:.1f} ô\n({rr_interval*1000:.0f} ms)",
#                         ha='center', va='bottom', fontsize=8, color='purple', zorder=5)

#                 ax.plot([t1, t2], [lead[peaks[j]], lead[peaks[j+1]]],
#                         linestyle='--', color='purple', alpha=0.6, zorder=4)

#     axes[-1].set_xlabel("Time (s)")

#     # Tiêu đề tổng
#     title = "ECG 12-lead with Attention (red dots) & R–R measurement (boxes)"
#     if sample_idx is not None:
#         title += f" | Sample {sample_idx}"
#     if label_true is not None:
#         title += f" | True: {label_true}"
#     fig.suptitle(title, fontsize=14)
#     plt.tight_layout()
#     return fig
