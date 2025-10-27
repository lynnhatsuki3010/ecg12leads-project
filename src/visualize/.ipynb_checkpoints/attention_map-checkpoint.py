import torch
import matplotlib.pyplot as plt
import numpy as np

def get_attention_weights(model, signal):
    model.eval()
    attention_weights = {}

    def hook_fn(module, input, output):
        # output: (context, weights)
        if isinstance(output, tuple) and len(output) > 1:
            attention_weights['value'] = output[1].detach().cpu()

    handle = model.attention.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(signal.unsqueeze(0).to(next(model.parameters()).device))
    handle.remove()

    weights = attention_weights.get('value', torch.zeros(signal.shape[1]))
    weights = weights.squeeze().numpy()
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
    return weights

def plot_attention_dots(signal, attn_weights, sample_idx=None, label_true=None, threshold=0.9):
    """
    Vẽ sóng ECG với các điểm đỏ biểu thị vùng mô hình chú ý.
    - signal: tensor [leads, length]
    - attn_weights: mảng attention (đã chuẩn hóa về [0,1])
    - threshold: ngưỡng attention (vd 0.9 nghĩa là chỉ hiển thị top 10% vùng chú ý)
    """
    lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    Ls = signal.shape[1]
    attn_up = np.interp(np.arange(Ls), np.linspace(0, Ls-1, len(attn_weights)), attn_weights)

    # Chọn các vị trí có attention cao
    high_attention_idx = np.where(attn_up > np.quantile(attn_up, threshold))[0]

    fig, axes = plt.subplots(signal.shape[0], 1, figsize=(12, 2 * signal.shape[0]), sharex=True)
    if signal.shape[0] == 1:
        axes = [axes]  # nếu chỉ có 1 lead

    for i in range(signal.shape[0]):
        lead = signal[i].cpu().numpy()
        axes[i].plot(lead, color='black', linewidth=0.8)
        axes[i].set_facecolor('white')  # nền trắng

        # Vẽ chấm đỏ tại các vị trí được chú ý
        axes[i].scatter(high_attention_idx, lead[high_attention_idx],
                        color='red', s=15, label='Attention')

        axes[i].set_ylabel(lead_names[i], rotation=0, labelpad=20)
        axes[i].grid(False)

    axes[-1].set_xlabel("Time steps")

    title = f"Attention Points (Red Dots)"
    if sample_idx is not None:
        title += f" | Sample {sample_idx}"
    if label_true is not None:
        title += f" | True: {label_true}"

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
