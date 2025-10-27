# Trực quan Grad-CAM

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# --- 1. Grad-CAM 1D raw ---
def gradcam_1d_raw(model, signal, target_class, target_layer_name="layer3.conv2"):
    model.eval()
    gradients = {}
    activations = {}

    def forward_hook(module, input, output):
        activations['value'] = output.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    target_layer = dict(model.named_modules())[target_layer_name]
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_backward_hook(backward_hook)

    x = signal.unsqueeze(0).to(next(model.parameters()).device)
    out = model(x)
    score = out[0, target_class]
    model.zero_grad()
    score.backward()

    grads = gradients['value']
    acts = activations['value']
    weights = torch.mean(grads, dim=2, keepdim=True)
    cam = torch.sum(weights * acts, dim=1).squeeze()
    cam = F.relu(cam)

    fh.remove()
    bh.remove()

    return cam.cpu().numpy()

# --- 2. Upsample bằng repeat ---
def upsample_repeat(cam_feat, signal_len):
    Lf = len(cam_feat)
    rep = int(np.ceil(signal_len / Lf))
    cam_rep = np.repeat(cam_feat, rep)[:signal_len]
    return cam_rep

# --- 3. Vẽ heatmap 12 lead ---
def plot_12lead_heatmap(signal, cam_feat, sample_idx=None, label_true=None, label_pred=None, cmap='hot'):
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                  'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    Ls = signal.shape[1]
    cam_up = upsample_repeat(cam_feat, Ls)
    cam_norm = (cam_up - cam_up.min()) / (cam_up.max() - cam_up.min() + 1e-9)

    num_leads = signal.shape[0]
    fig, axes = plt.subplots(num_leads, 1, figsize=(12, 2 * num_leads), sharex=True)

    for i in range(num_leads):
        lead = signal[i].cpu().numpy()
        axes[i].plot(lead, color='black', linewidth=0.7)
        axes[i].imshow(cam_norm[np.newaxis, :],
                       aspect='auto',
                       cmap=cmap,
                       extent=[0, len(lead), np.min(lead), np.max(lead)],
                       alpha=0.4)
        # 🩺 Dùng tên đạo trình thật
        axes[i].set_ylabel(lead_names[i], rotation=0, labelpad=20, fontsize=10)
        axes[i].grid(False)

    axes[-1].set_xlabel("Time steps")
    title = f"Grad-CAM (sample {sample_idx})"
    if label_true is not None:
        title += f" | True: {label_true}"
    if label_pred is not None:
        title += f" | Pred: {label_pred}"
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# --- 4. Hàm chính: Grad-CAM tự động theo nhãn dự đoán ---
def visualize_gradcam_multilabel(model, dataset, sample_idx=0, target_layer="layer3.conv2"):
    """
    - model: mô hình đã huấn luyện (multi-label)
    - dataset: tập test hoặc val (torch Dataset)
    - sample_idx: chỉ số mẫu muốn xem
    - target_layer: tên layer để lấy Grad-CAM
    """
    signal, true_label = dataset[sample_idx]  # (12, L), multi-label (8,)
    device = next(model.parameters()).device

    # Dự đoán xác suất cho tất cả lớp
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(signal.unsqueeze(0).to(device))).cpu().numpy()[0]

    # Xác định nhãn dự đoán mạnh nhất
    pred_idx = int(np.argmax(probs))
    print(f" Mẫu {sample_idx} | Dự đoán mạnh nhất: lớp {pred_idx} (p={probs[pred_idx]:.3f})")
    print(f"   Nhãn thật (multi-hot): {true_label.numpy()}")

    # Trích Grad-CAM cho lớp đó
    cam_feat = gradcam_1d_raw(model, signal, pred_idx, target_layer_name=target_layer)

    # Hiển thị heatmap
    plot_12lead_heatmap(signal, cam_feat,
                        sample_idx=sample_idx,
                        label_true=np.where(true_label.numpy() == 1)[0].tolist(),
                        label_pred=pred_idx)