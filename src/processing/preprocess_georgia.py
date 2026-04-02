# processing/preprocess_georgia.py
"""
Tiền xử lý metadata Georgia → tương đương preprocess.py của Chapman.
Đọc từng file .hea và tạo bảng metadata chuẩn hóa.

Yêu cầu:
 - Thư mục Georgia chứa các cặp file .hea + .mat

Lưu ý:
 - Format file .hea của Georgia GIỐNG HỆT Chapman
   → Tái dùng logic parse từ create_metadata_from_wfdb()
 - Nhãn đã là SNOMED code → KHÔNG cần gọi API FHIR để resolve
 - Cột #Age có thể là NaN → xử lý riêng thay vì drop

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5 SNOMED CODES MỤC TIÊU (giống Chapman):
    426177001 → SB   (Sinus Bradycardia)
    426783006 → SR   (Sinus Rhythm)
    164890007 → AF   (Atrial Fibrillation)
    426761007 → SVT  (Supraventricular Tachycardia)
    427084000 → ST   (Sinus Tachycardia)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import pandas as pd


# ══════════════════════════════════════════════════════════════
# MAPPING NHÃN
# ══════════════════════════════════════════════════════════════

# SNOMED code → tên nhãn ngắn (giống Chapman)
SNOMED_TO_SHORT = {
    '426177001': 'SB',
    '426783006': 'SR',
    '164890007': 'AF',
    '426761007': 'SVT',
    '427084000': 'ST',
}

TARGET_SNOMED_CODES = set(SNOMED_TO_SHORT.keys())


# ══════════════════════════════════════════════════════════════
# HÀM CHÍNH
# ══════════════════════════════════════════════════════════════

def create_metadata_from_georgia(root_dir: str) -> pd.DataFrame:
    """
    Đọc toàn bộ file .hea trong thư mục Georgia và tạo bảng metadata.
    Tương đương create_metadata_from_wfdb() trong preprocess.py.

    Format file .hea Georgia (giống Chapman):
        #Age: NaN hoặc số
        #Sex: Male / Female
        #Dx:  SNOMED code, có thể nhiều code cách nhau dấu phẩy

    Output có các cột giống Chapman:
        record_id       : 'E00001'
        file_path       : đường dẫn tuyệt đối tới file .hea
        age             : tuổi (float, NaN nếu không có)
        sex             : giới tính (Male/Female)
        diagnosis_codes : SNOMED codes cách nhau dấu phẩy
        diagnosis_names : tên nhãn ngắn tương ứng

    Args:
        root_dir (str): Đường dẫn thư mục gốc Georgia

    Returns:
        df (pd.DataFrame): Bảng metadata đầy đủ (chưa lọc nhãn)
    """
    records = []

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(".hea"):
                continue

            file_path = os.path.join(subdir, file)
            record_id = os.path.splitext(file)[0]   # E00001

            age, sex, dx = None, None, None

            with open(file_path, "r") as f:
                for line in f:
                    if line.startswith("#Age:"):
                        age = line.split(":")[1].strip()
                    elif line.startswith("#Sex:"):
                        sex = line.split(":")[1].strip()
                    elif line.startswith("#Dx:"):
                        dx = line.split(":")[1].strip()

            records.append({
                "record_id"       : record_id,
                "file_path"       : file_path,
                "age"             : age,
                "sex"             : sex,
                "diagnosis_codes" : dx,
            })

    df = pd.DataFrame(records)
    print(f"Tạo metadata Georgia xong: {len(df)} bản ghi")
    return df


def map_snomed_codes_georgia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mapping SNOMED codes → tên nhãn ngắn.
    Tương đương map_snomed_codes() trong preprocess.py,
    nhưng không cần file CSV vì Georgia dùng thẳng SNOMED code.

    Args:
        df (pd.DataFrame): Bảng metadata từ create_metadata_from_georgia()

    Returns:
        df (pd.DataFrame): Đã thêm cột diagnosis_names
    """
    def map_codes(codes_str):
        if pd.isna(codes_str) or codes_str == "":
            return None
        names = []
        for code in codes_str.split(","):
            code = code.strip()
            names.append(SNOMED_TO_SHORT.get(code, "Unknown"))
        return ",".join(names)

    df = df.copy()
    df["diagnosis_names"] = df["diagnosis_codes"].apply(map_codes)
    print("Mapping SNOMED codes Georgia xong")
    return df


def clean_metadata_georgia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch metadata Georgia.
    Tương đương clean_metadata() trong preprocess.py.

    Khác Chapman ở chỗ:
     - Age có thể là 'NaN' (string) → chuyển về float, giữ lại thay vì drop
       (nhiều record Georgia không có tuổi nhưng tín hiệu vẫn hợp lệ)
     - Sex Unknown → drop (không thể suy luận)
     - Diagnosis codes rỗng/NaN → drop

    Args:
        df (pd.DataFrame): Bảng metadata đầy đủ

    Returns:
        df (pd.DataFrame): Đã làm sạch
    """
    df = df.copy()

    # Loại bỏ record không có nhãn
    df = df[df["diagnosis_codes"].notna()]
    df = df[df["diagnosis_codes"].str.strip() != ""]

    # Chuyển age về numeric, giữ NaN (không drop vì Georgia hay thiếu tuổi)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # Loại bỏ sex unknown
    df = df[df["sex"].astype(str).str.lower() != "unknown"]
    df = df[df["sex"].notna()]

    print(f"Sau làm sạch: {len(df)} bản ghi")
    return df


def filter_target_labels_georgia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lọc chỉ giữ các record có ÍT NHẤT 1 trong 5 nhãn mục tiêu.

    Lưu ý: Georgia có thể có multi-label (nhiều SNOMED codes trong #Dx).
    Ở đây lọc giống Chapman — chỉ giữ record có ĐÚNG 1 nhãn mục tiêu
    để tránh nhập nhằng khi so sánh.

    Args:
        df (pd.DataFrame): Bảng metadata đã làm sạch

    Returns:
        df_filtered (pd.DataFrame): Chỉ giữ record có đúng 1 nhãn mục tiêu
    """
    df = df.copy()

    def count_target_codes(codes_str):
        if pd.isna(codes_str):
            return 0
        codes = [c.strip() for c in codes_str.split(",")]
        return sum(1 for c in codes if c in TARGET_SNOMED_CODES)

    def get_primary_target(codes_str):
        """Lấy nhãn mục tiêu duy nhất nếu có đúng 1."""
        codes = [c.strip() for c in codes_str.split(",")]
        targets = [c for c in codes if c in TARGET_SNOMED_CODES]
        return targets[0] if len(targets) == 1 else None

    df["n_target"] = df["diagnosis_codes"].apply(count_target_codes)

    # Chỉ giữ record có đúng 1 nhãn mục tiêu
    df_filtered = df[df["n_target"] == 1].copy()
    df_filtered["primary_snomed"] = df_filtered["diagnosis_codes"].apply(get_primary_target)
    df_filtered["primary_label"]  = df_filtered["primary_snomed"].map(SNOMED_TO_SHORT)

    # Drop cột tạm
    df_filtered = df_filtered.drop(columns=["n_target"])

    print(f"Sau lọc 5 nhãn mục tiêu: {len(df_filtered)} bản ghi")

    # In phân phối nhãn
    print("\n📊 Phân phối nhãn Georgia:")
    for short in ['SB', 'SR', 'AF', 'SVT', 'ST']:
        n    = (df_filtered["primary_label"] == short).sum()
        flag = "✅" if n >= 50 else ("⚠️ " if n >= 10 else "❌")
        note = "  ← quá ít, chỉ dùng tham khảo" if n < 30 else ""
        print(f"   {flag} {short:5}: {n:6} mẫu{note}")

    return df_filtered


# ══════════════════════════════════════════════════════════════
# PIPELINE CHÍNH
# ══════════════════════════════════════════════════════════════

def run_preprocessing_georgia():
    """
    Hàm wrapper chính — tương đương run_preprocessing() trong preprocess.py.
    Gọi từ main.py hoặc chạy trực tiếp.
    """
    ROOT_DIR   = r"E:\NCKH - 2026\ECG Project\data\raw\Georgia"
    SAVE_PATH  = r"E:\NCKH - 2026\ECG Project\data\processed_georgia\georgia_metadata.xlsx"

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # Bước 1: Đọc tất cả file .hea
    df = create_metadata_from_georgia(ROOT_DIR)

    # Bước 2: Mapping SNOMED → tên nhãn
    df = map_snomed_codes_georgia(df)

    # Bước 3: Làm sạch
    df = clean_metadata_georgia(df)

    # Bước 4: Lọc 5 nhãn mục tiêu
    df = filter_target_labels_georgia(df)

    # Lưu
    df.to_excel(SAVE_PATH, index=False)
    print(f"\n✅ Đã lưu metadata Georgia: {SAVE_PATH}")


if __name__ == "__main__":
    run_preprocessing_georgia()
