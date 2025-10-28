import torch
import matplotlib.pyplot as plt
import numpy as np

def get_attention_weights(model, signal):
    """
    Lấy attention weights từ attention layer của model.
    - signal: tensor [leads, length]
    """
    model.eval()
    attention_weights = {}

    def hook_fn(module, input, output):
        # output: (context, weights)
        if isinstance(output, tuple) and len(output) > 1:
            attention_weights['value'] = output[1].detach().cpu()

    # Đăng ký forward hook tại layer attention
    handle = model.attention.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(signal.unsqueeze(0).to(next(model.parameters()).device))
    handle.remove()

    # Lấy weights, chuẩn hóa về [0,1]
    weights = attention_weights.get('value', torch.zeros(signal.shape[1]))
    weights = weights.squeeze().numpy()
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
    return weights


def plot_attention_dots(signal, attn_weights=None, sample_idx=None, label_true=None,
                            fs=500, paper_speed=50):
    """
    Vẽ sóng ECG 12-lead với lưới chuẩn (50 mm/s, 10 mm/mV) và chấm đỏ attention.
    - signal: tensor [leads, length] (mV)
    - attn_weights: attention [seq_len] chuẩn hóa 0-1
    - fs: tần số lấy mẫu (Hz)
    - paper_speed: mm/s (50mm/s chuẩn)
    """
    lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    n_leads, L = signal.shape
    time = np.arange(L) / fs  # giây

    # Upsample attention về đúng chiều
    if attn_weights is not None:
        attn_up = np.interp(np.arange(L), np.linspace(0, L-1, len(attn_weights)), attn_weights)
        high_attn_idx = np.where(attn_up > np.quantile(attn_up, 0.9))[0]

    # Lưới chuẩn ECG
    small_box_s = 0.02  # 1 mm = 0.02 s @ 50 mm/s
    big_box_s   = 0.1   # 5 mm = 0.1 s
    small_box_mV = 0.1  # 1 ô nhỏ = 0.1 mV
    big_box_mV   = 0.5  # 1 ô lớn = 0.5 mV

    fig, axes = plt.subplots(n_leads, 1, figsize=(15, 2*n_leads), sharex=True)
    if n_leads == 1:
        axes = [axes]

    for i in range(n_leads):
        lead = signal[i].cpu().numpy()
        ax = axes[i]

        ax.set_facecolor('white')
        ax.set_ylabel(lead_names[i] if i<len(lead_names) else f"L{i+1}", rotation=0, labelpad=30)

        y_min, y_max = lead.min()*1.2, lead.max()*1.2
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(time[0], time[-1])

        # --- Lưới dọc (thời gian) ---
        for t in np.arange(time[0], time[-1], small_box_s):
            ax.axvline(t, color='lightpink', linewidth=0.3, zorder=1)
        for t in np.arange(time[0], time[-1], big_box_s):
            ax.axvline(t, color='red', linewidth=0.8, zorder=1)

        # --- Lưới ngang (biên độ) ---
        for y in np.arange(y_min, y_max, small_box_mV):
            ax.axhline(y, color='lightpink', linewidth=0.3, zorder=1)
        for y in np.arange(y_min, y_max, big_box_mV):
            ax.axhline(y, color='red', linewidth=0.8, zorder=1)
            
        # --- Chấm attention ---
        ax.scatter(
            time[high_attn_idx],
            lead[high_attn_idx],
            color='red',       # đỏ pastel
            alpha=0.8,             # trong suốt nhẹ
            s=12,
            edgecolors='none',    
            zorder=2
        )

        # --- Sóng ECG ---
        ax.plot(time, lead, color='black', linewidth=0.9, zorder=2)

       

    axes[-1].set_xlabel("Time (s)")

    title = "ECG 12-lead with Attention (red dots) & 50 mm/s grid"
    if sample_idx is not None:
        title += f" | Sample {sample_idx}"
    if label_true is not None:
        title += f" | True: {label_true}"
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig