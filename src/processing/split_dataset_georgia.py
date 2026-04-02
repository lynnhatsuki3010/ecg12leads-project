# processing/split_dataset_georgia.py
"""
Tạo tập test Georgia để đánh giá generalization của model Chapman.
Tương đương split_dataset.py của Chapman — nhưng chỉ tạo TEST SET.

Yêu cầu:
 - Phải có file valid_indices_georgia.npy sinh ra từ feature_extraction_georgia.py
 - Phải có file metadata Excel từ preprocess_georgia.py

Lý do chỉ tạo test set (không train/val):
 - Model đã huấn luyện trên Chapman
 - Georgia dùng để đánh giá cross-dataset generalization
 - Không có strat_fold như PTB-XL → dùng toàn bộ làm test
   (hoặc random split 80/20 nếu muốn giữ một phần để kiểm tra)
"""

import os
import numpy as np
import pandas as pd


# SNOMED codes — thứ tự PHẢI khớp với target_names lúc train Chapman
TARGET_COLS = ["426177001", "426783006", "164890007", "426761007", "427084000"]
# Tương ứng:   SB            SR            AF            SVT           ST

SHORT_MAP = {
    '426177001': 'SB',
    '426783006': 'SR',
    '164890007': 'AF',
    '426761007': 'SVT',
    '427084000': 'ST',
}


def create_multilabel_targets_georgia(df: pd.DataFrame, target_codes: list):
    """
    Tạo cột multi-hot nhãn từ danh sách SNOMED code.
    Tương đương create_multilabel_targets() trong split_dataset.py.

    Args:
        df (pd.DataFrame): Bảng metadata Georgia, có cột 'diagnosis_codes'
        target_codes (list): Danh sách SNOMED codes

    Returns:
        df (pd.DataFrame): Đã thêm cột nhãn
        y (np.ndarray): Ma trận nhãn shape (n_samples, n_classes)
    """
    df = df.copy()
    created_cols = []

    for code in target_codes:
        if code not in df.columns:
            if "diagnosis_codes" in df.columns:
                df[code] = df["diagnosis_codes"].apply(
                    lambda x: 1 if isinstance(x, str) and str(code) in x.split(",") else 0
                )
                created_cols.append(code)
            else:
                raise KeyError(f"Không tìm thấy cột '{code}' và không có cột 'diagnosis_codes'.")

    if created_cols:
        print(f"ℹ️ Đã tạo thêm {len(created_cols)} cột nhãn mới từ diagnosis_codes.")

    y = df[target_codes].values
    return df, y


def split_georgia_test(
    processed_dir: str,
    metadata_path: str,
    target_cols: list = None,
    output_dir_name: str = "splits_georgia_test"
):
    """
    Tạo tập test Georgia.
    Tương đương split_dataset() trong split_dataset.py.

    Georgia không có strat_fold → dùng toàn bộ làm test set.

    Args:
        processed_dir (str): Thư mục chứa .npy đã xử lý
        metadata_path (str): Đường dẫn file metadata Excel (.xlsx)
        target_cols (list): Danh sách SNOMED codes — mặc định dùng TARGET_COLS
        output_dir_name (str): Tên thư mục lưu kết quả
    """
    if target_cols is None:
        target_cols = TARGET_COLS

    print(f"\n🚀 Bắt đầu tạo test set Georgia từ: {processed_dir}")

    # === 1️⃣ Load Metadata & Valid Indices ===
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"❌ Không tìm thấy metadata: {metadata_path}")

    df = pd.read_excel(metadata_path)

    # Tìm file valid_indices_georgia.npy (nằm cùng cấp với thư mục processed_dir)
    valid_indices_path = os.path.join(os.path.dirname(processed_dir), "valid_indices_georgia.npy")

    if os.path.exists(valid_indices_path):
        inds = np.load(valid_indices_path)
        print(f"✅ Đã load valid_indices_georgia.npy: {len(inds)} mẫu hợp lệ.")
    else:
        raise FileNotFoundError(f"❌ Cần file {valid_indices_path} để đảm bảo đồng bộ nhãn!")

    # === 2️⃣ Lọc Dataframe theo valid indices ===
    df_valid = df.iloc[inds].copy()

    # === 3️⃣ Tạo X_paths (đường dẫn file .npy) ===
    # record_id dạng 'E00001' → processed_dir/E00001.npy
    def get_npy_path(record_id):
        return os.path.join(processed_dir, f"{record_id}.npy")

    X_paths = df_valid["record_id"].apply(get_npy_path).values

    # === 4️⃣ Tạo y (Nhãn) ===
    df_valid, y_all = create_multilabel_targets_georgia(df_valid, target_cols)

    print(f"📊 Shape dữ liệu: X={X_paths.shape}, y={y_all.shape}")

    # === 5️⃣ Dùng toàn bộ làm test (không có fold) ===
    X_test = X_paths
    y_test = y_all

    print(f"✅ Test set Georgia: {len(X_test)} mẫu")

    # In phân phối nhãn test
    print(f"\n📊 Phân phối nhãn test:")
    for i, code in enumerate(target_cols):
        n    = int(y_test[:, i].sum())
        flag = "✅" if n >= 30 else ("⚠️ " if n >= 5 else "❌")
        note = "  ← quá ít, chỉ tham khảo" if n < 30 else ""
        print(f"   {flag} {SHORT_MAP.get(code, code):5}: {n:5} mẫu{note}")

    # === 6️⃣ Lưu kết quả ===
    splits_dir = os.path.join(os.path.dirname(processed_dir), output_dir_name)
    os.makedirs(splits_dir, exist_ok=True)

    np.save(os.path.join(splits_dir, "test_files.npy"),    X_test)
    np.save(os.path.join(splits_dir, "y_test.npy"),        y_test)
    np.save(os.path.join(splits_dir, "target_names.npy"),
            np.array([SHORT_MAP[c] for c in target_cols]))

    print(f"\n📁 Kết quả đã lưu tại: {splits_dir}")
    print(f"   test_files.npy : {len(X_test)} đường dẫn")
    print(f"   y_test.npy     : shape {y_test.shape}")
    print(f"   target_names   : {[SHORT_MAP[c] for c in target_cols]}")


# === Hàm wrapper chính ===
def run_split_georgia():
    """
    Hàm gọi mặc định từ main.py.
    Tương đương run_split_dataset() trong split_dataset.py.
    """
    BASE_DATA_DIR  = r"E:\NCKH - 2026\ECG Project\data\processed_georgia"

    METADATA       = os.path.join(BASE_DATA_DIR, "georgia_metadata.xlsx")
    PROCESSED_DIR  = os.path.join(BASE_DATA_DIR, "ecg_leads12_georgia")
    OUTPUT_NAME    = "splits_georgia_test"

    try:
        split_georgia_test(
            processed_dir   = PROCESSED_DIR,
            metadata_path   = METADATA,
            target_cols     = TARGET_COLS,
            output_dir_name = OUTPUT_NAME
        )
    except Exception as e:
        print(f"\n❌ Lỗi xảy ra: {e}")


# === Main Block ===
if __name__ == "__main__":
    run_split_georgia()
