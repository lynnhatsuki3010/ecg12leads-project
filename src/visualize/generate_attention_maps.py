import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from visualize.attention_map import get_attention_weights, plot_attention_dots


def generate_attention_visualizations(
    model,
    train_files_path,
    y_train_path,
    ecg_npy_dir,
    output_root,
    label_map,
    device=None
):
    """
    Sinh ảnh attention cho toàn bộ tập train.
    Lưu về thư mục processed/attention_visualize/[Tên bệnh]/record_x.png
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Tạo thư mục đầu ra ===
    os.makedirs(output_root, exist_ok=True)
    for name in label_map.values():
        os.makedirs(os.path.join(output_root, name), exist_ok=True)

    # === Load danh sách file & nhãn ===
    train_files = np.load(train_files_path, allow_pickle=True)
    y_train = np.load(y_train_path, allow_pickle=True)

    print(f"Tổng số mẫu: {len(train_files)}")

    for i, (file_name, y_value) in enumerate(
        tqdm(zip(train_files, y_train), total=len(train_files), desc="Generating Attention")
    ):
        # --- Đảm bảo đường dẫn ---
        file_path = os.path.join(ecg_npy_dir, os.path.basename(file_name))
        if not os.path.exists(file_path):
            print(f"[Bỏ qua] Không tìm thấy file: {file_path}")
            continue

        try:
            # === Lấy tên nhãn ===
            if isinstance(y_value, (np.ndarray, list)) and len(y_value) == len(label_map):
                label_idx = int(np.argmax(y_value))
                label_code = list(label_map.keys())[label_idx]
            else:
                label_code = str(y_value)

            label_name = label_map.get(label_code, "Unknown")

            # === Load tín hiệu & lấy attention ===
            signal = torch.tensor(np.load(file_path), dtype=torch.float32)
            attn_weights = get_attention_weights(model, signal)

            # === Vẽ attention ===
            fig = plot_attention_dots(
                signal=signal,
                attn_weights=attn_weights,
                fs=500,
                paper_speed=50
            )

            # === Lưu ảnh ===
            output_label_dir = os.path.join(output_root, label_name)
            os.makedirs(output_label_dir, exist_ok=True)

            save_path = os.path.join(
                output_label_dir,
                f"{os.path.splitext(os.path.basename(file_name))[0]}.png"
            )
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        except Exception as e:
            print(f"[Warning] Lỗi khi xử lý {file_name}: {e}")
            continue

    print("Hoàn tất sinh ảnh attention cho toàn bộ tập train!")
