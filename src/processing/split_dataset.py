"""
processing/split_dataset.py
---------------------------
Tạo nhãn multi-hot theo SNOMED CT và chia dữ liệu train/val/test.
Lưu kết quả ra thư mục splits/ để main.py có thể load trực tiếp.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_multilabel_targets(df: pd.DataFrame, snomed_codes: list):
    """
    Tạo cột multi-hot nhãn từ danh sách SNOMED code.

    Args:
        df (pd.DataFrame): Bảng thông tin bệnh nhân, có cột 'diagnosis_codes'.
        snomed_codes (list[str]): Danh sách mã bệnh được giữ lại.

    Returns:
        tuple: (df mới có cột mã bệnh, y: ma trận nhãn (num_samples, num_codes))
    """
    df = df.copy()
    for code in snomed_codes:
        df[code] = df["diagnosis_codes"].apply(
            lambda x: 1 if isinstance(x, str) and code in x.split(",") else 0
        )
    y = df[snomed_codes].values
    return df, y


def split_dataset(
    save_dir: str,
    df_resolved: pd.DataFrame,
    snomed_codes: list,
    test_size=0.3,
    val_ratio=2 / 3,
    random_state=42,
):
    """
    Chia dữ liệu đã xử lý thành train/val/test và lưu ra thư mục splits/.

    Args:
        save_dir (str): Thư mục chứa các file .npy đã được xử lý (từ bước feature_extraction).
        df_resolved (pd.DataFrame): Bảng mapping giữa file path và chẩn đoán.
        snomed_codes (list[str]): Danh sách SNOMED code cần giữ lại.
        test_size (float): Tỷ lệ test.
        val_ratio (float): Tỷ lệ val trong phần test+val.
        random_state (int): Seed cố định.
    """
    # === 1️⃣ Tạo nhãn multi-hot ===
    df_resolved, y = create_multilabel_targets(df_resolved, snomed_codes)

    # === 2️⃣ Lấy danh sách file .npy ===
    all_files = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith(".npy")]
    all_files.sort()  # đảm bảo index nhất quán với df_resolved

    if len(all_files) == 0:
        raise FileNotFoundError(f"Không tìm thấy file .npy nào trong thư mục: {save_dir}")

    print(f"🔹 Tổng số file tín hiệu ECG: {len(all_files)}")

    # === 3️⃣ Chia train/val/test ===
    train_files, temp_files = train_test_split(
        all_files, test_size=test_size, random_state=random_state
    )
    val_files, test_files = train_test_split(
        temp_files, test_size=val_ratio, random_state=random_state
    )

    print(f"✅ Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    # === 4️⃣ Hàm lấy index từ tên file (record_123.npy -> 123) ===
    def get_index_from_filename(path):
        fname = os.path.basename(path)
        try:
            idx = int(fname.replace("record_", "").replace(".npy", ""))
            return idx
        except ValueError:
            raise ValueError(f"Tên file không hợp lệ: {fname}")

    # === 5️⃣ Ánh xạ nhãn tương ứng ===
    y_train = np.array([y[get_index_from_filename(f)] for f in train_files])
    y_val = np.array([y[get_index_from_filename(f)] for f in val_files])
    y_test = np.array([y[get_index_from_filename(f)] for f in test_files])

    print("y_train shape:", y_train.shape)
    print("y_val shape:", y_val.shape)
    print("y_test shape:", y_test.shape)

    # === 6️⃣ Lưu ra thư mục splits/ ===
    splits_dir = os.path.join(save_dir, "..", "splits")
    os.makedirs(splits_dir, exist_ok=True)

    np.save(os.path.join(splits_dir, "train_files.npy"), np.array(train_files))
    np.save(os.path.join(splits_dir, "val_files.npy"), np.array(val_files))
    np.save(os.path.join(splits_dir, "test_files.npy"), np.array(test_files))
    np.save(os.path.join(splits_dir, "y_train.npy"), y_train)
    np.save(os.path.join(splits_dir, "y_val.npy"), y_val)
    np.save(os.path.join(splits_dir, "y_test.npy"), y_test)

    print(f"📁 Đã lưu toàn bộ file chia tập vào: {splits_dir}")

    return {
        "train": {"X": train_files, "y": y_train},
        "val": {"X": val_files, "y": y_val},
        "test": {"X": test_files, "y": y_test},
    }


# === 7️⃣ Hàm wrapper để gọi từ main.py ===
def run_split_dataset():
    """
    Hàm tiện ích để gọi từ main.py — tự động đọc metadata và chạy chia tập.
    """
    print("\n===== Bước 3: Chia dữ liệu train/val/test =====")

    # Đường dẫn tương đối
    base_dir = os.path.dirname(os.path.dirname(__file__))
    processed_dir = os.path.join(base_dir, "data", "processed", "ecg_npy")
    metadata_path = os.path.join(base_dir, "data", "processed", "patient_metadata_clean.xlsx")

    # Load metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Không tìm thấy file metadata: {metadata_path}")

    df = pd.read_excel(metadata_path)

    # Danh sách mã SNOMED chính (ví dụ top 10)
    snomed_top10 = [
        "426177001","426783006","164890007","427084000",
        "427393009","164889003","429622005","39732003"
    ]

    # Chạy chia tập
    split_dataset(save_dir=processed_dir, df_resolved=df, snomed_codes=snomed_top10)

    print("✅ Hoàn tất chia dữ liệu!\n")


# === Ví dụ chạy độc lập ===
if __name__ == "__main__":
    run_split_dataset()
