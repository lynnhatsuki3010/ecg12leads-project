import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt, iirnotch
import torch
import torch.nn.functional as F
import io, base64, json, sys, os, traceback
sys.stderr.reconfigure(line_buffering=True)

# ensure stdout prints utf-8 (Windows compatibility)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ---------- import model ----------
from model import ResNet1D50Attention

# ---------- model loader ----------
def load_model(model_path="final_fine_tuned_modelres50.pth", device="cpu"):
    model = ResNet1D50Attention(num_classes=8)
    try:
        loaded = torch.load(model_path, map_location=torch.device(device), weights_only=True)
    except:
        loaded = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(loaded)
    model.to(device)
    model.eval()
    return model

try:
    model = load_model()
    print("[INFO] ✅ Model loaded successfully", file=sys.stderr, flush=True)
except Exception as e:
    print("[ERROR] ❌ Failed to load model:", str(e), file=sys.stderr, flush=True)
    model = None

# ---------- ECG processing ----------
def load_ecg_mat(mat_file):
    mat = scipy.io.loadmat(mat_file)
    if "val" in mat:
        arr = mat["val"]
    elif "ecg" in mat:
        arr = mat["ecg"]
    else:
        found = None
        for v in mat.values():
            if isinstance(v, np.ndarray) and v.ndim == 2:
                found = v
                break
        if found is None:
            raise KeyError("No 2D ECG array found in .mat")
        arr = found
    return arr.astype(np.float32)

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype="band")
    return b, a

def apply_bandpass(sig, fs, lowcut=0.5, highcut=40.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, sig)

def apply_notch(sig, fs, notch_freq=50.0, quality=30.0):
    b, a = iirnotch(notch_freq / (fs / 2), quality)
    return filtfilt(b, a, sig)

def denoise_all_leads(signals, fs=500):
    denoised = np.zeros_like(signals, dtype=np.float32)
    for i in range(signals.shape[0]):
        x = apply_notch(signals[i, :], fs)
        x = apply_bandpass(x, fs)
        denoised[i, :] = x
    return denoised

def preprocess_ecg(mat_file, fs=500):
    signals = load_ecg_mat(mat_file)
    if signals.shape[0] != 12 and signals.shape[1] == 12:
        signals = signals.T
    raw_mV = signals / 1000.0
    denoised = denoise_all_leads(raw_mV, fs)
    z = (denoised - denoised.mean(axis=1, keepdims=True)) / (denoised.std(axis=1, keepdims=True) + 1e-8)
    z = np.clip(z, -10, 10)
    return z

# ---------- plotting ----------
def fig_to_base64(fig, dpi=200):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def plot_ecg_12leads(signals, fs=500):
    leads = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
    t = np.arange(signals.shape[1]) / fs
    fig, axes = plt.subplots(6, 2, figsize=(12, 9), sharex=False)
    axes = axes.flatten()
    for i in range(12):
        axes[i].plot(t, signals[i, :], linewidth=1.0)
        axes[i].set_title(leads[i] if i < len(leads) else f"L{i+1}", fontsize=11)
        axes[i].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        axes[i].set_yticks([]); axes[i].set_xticks([])
    for j in range(12, len(axes)):
        axes[j].axis("off")
    plt.tight_layout(pad=2.0)
    return fig_to_base64(fig)

# ---------- attention ----------
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

    handle = model.attention.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(signal.unsqueeze(0).to(next(model.parameters()).device))
    handle.remove()

    # Lấy weights và chuẩn hóa về [0,1]
    weights = attention_weights.get('value', torch.zeros(signal.shape[1]))
    weights = weights.squeeze().numpy()
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
    return weights


def plot_attention_dots(signal, attn_weights=None, fs=500, paper_speed=50):
    """
    Vẽ sóng ECG 12-lead với lưới chuẩn (50 mm/s, 10 mm/mV) và chấm đỏ attention.
    - signal: tensor [leads, length] (mV)
    - attn_weights: attention [seq_len] chuẩn hóa 0-1
    - fs: tần số lấy mẫu (Hz)
    - paper_speed: mm/s (50 mm/s chuẩn)
    """
    if isinstance(signal, torch.Tensor):
        signal = signal.cpu().numpy()

    lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    n_leads, L = signal.shape
    time = np.arange(L) / fs

    # === Upsample attention ===
    if attn_weights is not None:
        attn_up = np.interp(np.arange(L), np.linspace(0, L-1, len(attn_weights)), attn_weights)
        high_attn_idx = np.where(attn_up > np.quantile(attn_up, 0.9))[0]
    else:
        high_attn_idx = np.array([], dtype=int)

    # === Grid parameters ===
    small_box_s = 0.02  # 1 mm = 0.02 s @ 50 mm/s
    big_box_s   = 0.1
    small_box_mV = 0.1
    big_box_mV   = 0.5

    # === Layout 2 cột × 6 hàng ===
    fig, axes = plt.subplots(6, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for i in range(12):
        ax = axes[i]
        lead = signal[i]

        ax.set_facecolor('white')
        ax.set_title(lead_names[i] if i < len(lead_names) else f"L{i+1}", fontsize=10)
        y_min, y_max = lead.min() * 1.2, lead.max() * 1.2
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(time[0], time[-1])

        # --- Lưới dọc ---
        for t in np.arange(time[0], time[-1], small_box_s):
            ax.axvline(t, color='lightpink', linewidth=0.3, zorder=1)
        for t in np.arange(time[0], time[-1], big_box_s):
            ax.axvline(t, color='red', linewidth=0.8, zorder=1)

        # --- Lưới ngang ---
        for y in np.arange(y_min, y_max, small_box_mV):
            ax.axhline(y, color='lightpink', linewidth=0.3, zorder=1)
        for y in np.arange(y_min, y_max, big_box_mV):
            ax.axhline(y, color='red', linewidth=0.8, zorder=1)
            
        # --- Chấm attention ---
        if high_attn_idx.size > 0:
            ax.scatter(
                time[high_attn_idx],
                lead[high_attn_idx],
                color='red',
                alpha=0.8,
                s=12,
                edgecolors='none',
                zorder=2
            )
        # --- Sóng ECG ---
        ax.plot(time, lead, color='black', linewidth=0.9, zorder=2)

        

        ax.set_yticks([])
        ax.set_xticks([])

    for j in range(12, len(axes)):
        axes[j].axis("off")

    plt.tight_layout(pad=2.0)
    fig.suptitle("ECG 12-lead with Attention (red dots) — 50 mm/s grid", fontsize=14)
    return fig_to_base64(fig)


# ---------- main detection ----------
def detect_ecg(mat_file):
    if model is None:
        return {"error": "Model not loaded on server"}

    mat_file = os.path.abspath(mat_file)
    print(f"[INFO] Detecting: {mat_file}", file=sys.stderr, flush=True)
    if not os.path.exists(mat_file):
        return {"error": f"File not found: {mat_file}"}

    x = preprocess_ecg(mat_file)
    x_input = np.expand_dims(x, axis=0)
    x_tensor = torch.tensor(x_input, dtype=torch.float32)

    try:
        with torch.no_grad():
            print("[DEBUG] Running model forward...", file=sys.stderr, flush=True)
            device = next(model.parameters()).device
            x_tensor = x_tensor.to(device)
            output = model(x_tensor)
            probs = torch.sigmoid(output).cpu().numpy().flatten()
    except Exception as e:
        print("[ERROR] Exception during model forward:\n", traceback.format_exc(), file=sys.stderr, flush=True)
        return {"error": f"Lỗi khi chạy model: {str(e)}"}

    # mapping
    best_thresholds = np.array([0.5, 0.75, 0.75, 0.65, 0.8, 0.6, 0.9, 0.9])
    adjusted_probs = np.clip((probs - best_thresholds + 0.5), 0, 1)

    labels = [
        "426177001", "426783006", "164890007", "427084000",
        "427393009", "164889003", "429622005", "39732003"
    ]
    snomed_to_name = {
        "426177001": "Nhịp chậm xoang",
        "426783006": "Nhịp xoang bình thường",
        "164890007": "Cuồng nhĩ",
        "427084000": "Chênh lên đoạn ST",
        "427393009": "Loạn nhịp xoang",
        "164889003": "Rung nhĩ",
        "429622005": "Chênh xuống đoạn ST",
        "39732003": "Trục điện tim lệch trái"
    }
    results = {snomed_to_name.get(labels[i], labels[i]): float(adjusted_probs[i]) for i in range(len(labels))}

    # === tạo thư mục static/results nếu chưa có ===
    os.makedirs("static/results", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(mat_file))[0]

    # === ảnh 1 ===
    img_base64 = plot_ecg_12leads(x)
    ecg_path = f"static/results/{base_name}_ecg.png"
    with open(ecg_path, "wb") as f:
        f.write(base64.b64decode(img_base64))

    # === ảnh 2 (attention) ===
    try:
        attn_weights = get_attention_weights(model, torch.tensor(x, dtype=torch.float32))
        attn_b64 = plot_attention_dots(torch.tensor(x, dtype=torch.float32), attn_weights)
        attn_path = f"static/results/{base_name}_attn.png"
        with open(attn_path, "wb") as f:
            f.write(base64.b64decode(attn_b64))
    except Exception as e:
        print("[WARN] Failed to produce attention image:", str(e), file=sys.stderr, flush=True)
        attn_path = ""

    # === trả về URL thay vì chuỗi base64 ===
    return {"results": results, "images": [f"/{ecg_path}", f"/{attn_path}"]}

# ---------- CLI entry ----------
if __name__ == "__main__":
    try:
        mat_file = " ".join(sys.argv[1:]).strip('"')
        out = detect_ecg(mat_file)
        print(json.dumps(out, ensure_ascii=False), flush=True)
    except Exception as e:
        tb = traceback.format_exc()
        print(json.dumps({"error": str(e), "traceback": tb}, ensure_ascii=False), flush=True)
