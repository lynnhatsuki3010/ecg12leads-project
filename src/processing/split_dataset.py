# processing/split_dataset.py
"""
Chia dữ liệu train/val/test dựa trên kết quả tiền xử lý mới.
Yêu cầu:
 - Phải có file 'valid_indices.npy' sinh ra từ bước feature_extraction.
 - Map đúng tên file gốc (.mat -> .npy) với nhãn trong metadata.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_multilabel_targets(df: pd.DataFrame, target_codes: list):
    """
    Tạo cột multi-hot nhãn từ danh sách SNOMED code (nếu chưa có trong Excel).
    Nếu cột đã tồn tại trong df, hàm này sẽ bỏ qua việc tạo lại.
    """
    df = df.copy()
    created_cols = []
    
    for code in target_codes:
        # Nếu cột chưa có, tạo từ 'diagnosis_codes'
        if code not in df.columns:
            if "diagnosis_codes" in df.columns:
                df[code] = df["diagnosis_codes"].apply(
                    lambda x: 1 if isinstance(x, str) and str(code) in x.split(",") else 0
                )
                created_cols.append(code)
            else:
                raise KeyError(f"Không tìm thấy cột '{code}' và cũng không có cột 'diagnosis_codes' để sinh nhãn.")
    
    if created_cols:
        print(f"ℹ️ Đã tạo thêm {len(created_cols)} cột nhãn mới từ diagnosis_codes.")
        
    y = df[target_codes].values
    return df, y


def split_dataset(
    processed_dir: str,
    metadata_path: str,
    target_cols: list,
    test_size=0.2,
    val_size=0.1,  # 10% val, 10% test, 80% train
    random_state=42,
    output_dir_name="splits_leads12_130125"
):
    """
    Chia dữ liệu train/val/test đồng bộ với valid_indices.
    """
    print(f"\n🚀 Bắt đầu chia dữ liệu từ: {processed_dir}")
    
    # === 1️⃣ Load Metadata & Valid Indices ===
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"❌ Không tìm thấy metadata: {metadata_path}")
        
    df = pd.read_excel(metadata_path)
    
    # Tìm file valid_indices.npy (nằm cùng cấp với thư mục processed_dir)
    valid_indices_path = os.path.join(os.path.dirname(processed_dir), "valid_indices_v3.npy")
    
    if os.path.exists(valid_indices_path):
        inds = np.load(valid_indices_path)
        print(f"✅ Đã load valid_indices.npy: {len(inds)} mẫu hợp lệ.")
    else:
        # Fallback: Nếu không có file index, quét tất cả file trong folder
        print("⚠️ Không tìm thấy valid_indices.npy -> Quét file trong thư mục...")
        files = [f for f in os.listdir(processed_dir) if f.endswith('.npy')]
        # Cần logic để map file name ngược lại index trong excel (phức tạp và dễ sai)
        # Ở đây ta bắt buộc nên dùng valid_indices để an toàn
        raise FileNotFoundError(f"❌ Cần file {valid_indices_path} để đảm bảo đồng bộ nhãn!")

    # === 2️⃣ Lọc Dataframe theo valid indices ===
    # Chỉ lấy những dòng tương ứng với file đã xử lý thành công
    df_valid = df.iloc[inds].copy()

    # === 3️⃣ Tạo X_paths (Đường dẫn file .npy) ===
    # Giả định cột file_path trong excel dạng: ".../01/010/JS00001.mat"
    # Ta cần chuyển thành: "{processed_dir}/JS00001.npy"
    
    def get_npy_path(original_path):
        filename = os.path.splitext(os.path.basename(original_path))[0] + ".npy"
        return os.path.join(processed_dir, filename)

    X_paths = df_valid["file_path"].apply(get_npy_path).values

    # === 4️⃣ Tạo y (Nhãn) ===
    # Tự động tạo nhãn nếu chỉ đưa vào mã SNOMED, hoặc lấy cột có sẵn
    df_valid, y_all = create_multilabel_targets(df_valid, target_cols)

    print(f"📊 Shape dữ liệu: X={X_paths.shape}, y={y_all.shape}")

    # === 5️⃣ Chia Train / Val / Test ===
    # Logic: Chia Test trước, sau đó chia Train/Val từ phần còn lại
    # Ví dụ: test_size=0.1 (10%), val_size=0.1 (10%) -> Train = 80%
    
    # Chia ra Test trước
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        X_paths, y_all, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    # Tính tỷ lệ Val trên phần còn lại
    # Nếu muốn Val là 10% tổng, mà Test đã lấy 10%, còn lại 90%. 
    # Tỷ lệ split val phải là 0.1 / 0.9 = 1/9
    val_ratio_relative = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_temp, y_train_temp, test_size=val_ratio_relative, random_state=random_state, shuffle=True
    )

    print(f"✅ Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # === 6️⃣ Lưu kết quả ===
    # Lưu ra thư mục cùng cấp với processed_dir
    splits_dir = os.path.join(os.path.dirname(processed_dir), output_dir_name)
    os.makedirs(splits_dir, exist_ok=True)

    np.save(os.path.join(splits_dir, "train_files.npy"), X_train)
    np.save(os.path.join(splits_dir, "val_files.npy"), X_val)
    np.save(os.path.join(splits_dir, "test_files.npy"), X_test)
    
    np.save(os.path.join(splits_dir, "y_train.npy"), y_train)
    np.save(os.path.join(splits_dir, "y_val.npy"), y_val)
    np.save(os.path.join(splits_dir, "y_test.npy"), y_test)
    
    # Lưu tên các cột nhãn để visualize sau này
    np.save(os.path.join(splits_dir, "target_names.npy"), np.array(target_cols))

    print(f"📁 Kết quả đã lưu tại: {splits_dir}")


# === 7️⃣ Hàm wrapper chính ===
def run_split_dataset(processed_dir_name, output_dir_name): # adding parameters for flexibility
    """ 
    Hàm gọi mặc định từ main.py
    """
    # CẤU HÌNH ĐƯỜNG DẪN (Bạn sửa lại cho khớp với máy mình)
    BASE_DATA_DIR = r"E:\NCKH - 2026\ECG Project\data\processed"
    
    # Input
    METADATA = os.path.join(BASE_DATA_DIR, "patient_metadata_superclean.xlsx")
    #PROCESSED_DIR = os.path.join(BASE_DATA_DIR, "ecg_leads12_160125") # Folder chứa .npy mới
    
    # Output Folder Name
    #OUTPUT_NAME = "splits_leads12_160125"
    PROCESSED_DIR = os.path.join(BASE_DATA_DIR, processed_dir_name)
    OUTPUT_NAME = output_dir_name
    

    # DANH SÁCH NHÃN (Target Columns)
    # Nếu trong Excel đã có tên tiếng Việt/Anh thì điền vào đây.
    # Nếu chưa có, điền mã SNOMED code, code sẽ tự sinh cột từ 'diagnosis_codes'
    
    # Ví dụ dùng 7 nhãn lâm sàng (như bạn đã cấu hình trước đó):
    # TARGET_COLS = [
    #     "Nhịp chậm xoang - SB", 
    #     "Nhịp xoang bình thường - SR", 
    #     "Cuồng nhĩ - AF", 
    #     "Rung nhĩ - AFIB", 
    #     "Nhịp nhanh trên thất - SVT",
    #     "Nhịp xoang nhanh - ST", 
    #     "Phì đại thất trái - LVH"
    # ]
    
    # Ví dụ nếu muốn dùng SNOMED codes (như file cũ):
    TARGET_COLS = ["426177001","426783006","164890007", "426761007", "427084000"]

    try:
        split_dataset(
            processed_dir=PROCESSED_DIR,
            metadata_path=METADATA,
            target_cols=TARGET_COLS,
            test_size=0.1, # 10% Test
            val_size=0.1,  # 10% Val
            random_state=42,
            output_dir_name=OUTPUT_NAME
        )
    except Exception as e:
        print(f"\n❌ Lỗi xảy ra: {e}")

# === Main Block ===
if __name__ == "__main__":
    run_split_dataset()

# # processing/split_dataset.py
# """
# Chia dữ liệu train/val/test dựa trên kết quả tiền xử lý mới.
# Yêu cầu:
#  - Phải có file 'valid_indices.npy' sinh ra từ bước feature_extraction.
#  - Map đúng tên file gốc (.mat -> .npy) với nhãn trong metadata.
# """

# import os
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split


# def create_multilabel_targets(df: pd.DataFrame, target_codes: list):
#     """
#     Tạo cột multi-hot nhãn từ danh sách SNOMED code (nếu chưa có trong Excel).
#     Nếu cột đã tồn tại trong df, hàm này sẽ bỏ qua việc tạo lại.
#     """
#     df = df.copy()
#     created_cols = []
    
#     for code in target_codes:
#         # Nếu cột chưa có, tạo từ 'diagnosis_codes'
#         if code not in df.columns:
#             if "diagnosis_codes" in df.columns:
#                 df[code] = df["diagnosis_codes"].apply(
#                     lambda x: 1 if isinstance(x, str) and str(code) in x.split(",") else 0
#                 )
#                 created_cols.append(code)
#             else:
#                 raise KeyError(f"Không tìm thấy cột '{code}' và cũng không có cột 'diagnosis_codes' để sinh nhãn.")
    
#     if created_cols:
#         print(f"ℹ️ Đã tạo thêm {len(created_cols)} cột nhãn mới từ diagnosis_codes.")
        
#     y = df[target_codes].values
#     return df, y


# def split_dataset(
#     processed_dir: str,
#     metadata_path: str,
#     target_cols: list,
#     test_size=0.2,
#     val_size=0.1,  # 10% val, 10% test, 80% train
#     random_state=42,
#     output_dir_name="splits_fixed_v3"
# ):
#     """
#     Chia dữ liệu train/val/test đồng bộ với valid_indices.
#     """
#     print(f"\n🚀 Bắt đầu chia dữ liệu từ: {processed_dir}")
    
#     # === 1️⃣ Load Metadata & Valid Indices ===
#     if not os.path.exists(metadata_path):
#         raise FileNotFoundError(f"❌ Không tìm thấy metadata: {metadata_path}")
        
#     df = pd.read_excel(metadata_path)
    
#     # Tìm file valid_indices.npy (nằm cùng cấp với thư mục processed_dir)
#     valid_indices_path = os.path.join(os.path.dirname(processed_dir), "valid_indices.npy")
    
#     if os.path.exists(valid_indices_path):
#         inds = np.load(valid_indices_path)
#         print(f"✅ Đã load valid_indices.npy: {len(inds)} mẫu hợp lệ.")
#     else:
#         # Fallback: Nếu không có file index, quét tất cả file trong folder
#         print("⚠️ Không tìm thấy valid_indices.npy -> Quét file trong thư mục...")
#         files = [f for f in os.listdir(processed_dir) if f.endswith('.npy')]
#         # Cần logic để map file name ngược lại index trong excel (phức tạp và dễ sai)
#         # Ở đây ta bắt buộc nên dùng valid_indices để an toàn
#         raise FileNotFoundError(f"❌ Cần file {valid_indices_path} để đảm bảo đồng bộ nhãn!")

#     # === 2️⃣ Lọc Dataframe theo valid indices ===
#     # Chỉ lấy những dòng tương ứng với file đã xử lý thành công
#     df_valid = df.iloc[inds].copy()

#     # === 3️⃣ Tạo X_paths (Đường dẫn file .npy) ===
#     # Giả định cột file_path trong excel dạng: ".../01/010/JS00001.mat"
#     # Ta cần chuyển thành: "{processed_dir}/JS00001.npy"
    
#     def get_npy_path(original_path):
#         filename = os.path.splitext(os.path.basename(original_path))[0] + ".npy"
#         return os.path.join(processed_dir, filename)

#     X_paths = df_valid["file_path"].apply(get_npy_path).values

#     # === 4️⃣ Tạo y (Nhãn) ===
#     # Tự động tạo nhãn nếu chỉ đưa vào mã SNOMED, hoặc lấy cột có sẵn
#     df_valid, y_all = create_multilabel_targets(df_valid, target_cols)

#     print(f"📊 Shape dữ liệu: X={X_paths.shape}, y={y_all.shape}")

#     # === 5️⃣ Chia Train / Val / Test ===
#     # Logic: Chia Test trước, sau đó chia Train/Val từ phần còn lại
#     # Ví dụ: test_size=0.1 (10%), val_size=0.1 (10%) -> Train = 80%
    
#     # Chia ra Test trước
#     X_train_temp, X_test, y_train_temp, y_test = train_test_split(
#         X_paths, y_all, test_size=test_size, random_state=random_state, shuffle=True
#     )
    
#     # Tính tỷ lệ Val trên phần còn lại
#     # Nếu muốn Val là 10% tổng, mà Test đã lấy 10%, còn lại 90%. 
#     # Tỷ lệ split val phải là 0.1 / 0.9 = 1/9
#     val_ratio_relative = val_size / (1 - test_size)
    
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_train_temp, y_train_temp, test_size=val_ratio_relative, random_state=random_state, shuffle=True
#     )

#     print(f"✅ Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

#     # === 6️⃣ Lưu kết quả ===
#     # Lưu ra thư mục cùng cấp với processed_dir
#     splits_dir = os.path.join(os.path.dirname(processed_dir), output_dir_name)
#     os.makedirs(splits_dir, exist_ok=True)

#     np.save(os.path.join(splits_dir, "train_files.npy"), X_train)
#     np.save(os.path.join(splits_dir, "val_files.npy"), X_val)
#     np.save(os.path.join(splits_dir, "test_files.npy"), X_test)
    
#     np.save(os.path.join(splits_dir, "y_train.npy"), y_train)
#     np.save(os.path.join(splits_dir, "y_val.npy"), y_val)
#     np.save(os.path.join(splits_dir, "y_test.npy"), y_test)
    
#     # Lưu tên các cột nhãn để visualize sau này
#     np.save(os.path.join(splits_dir, "target_names.npy"), np.array(target_cols))

#     print(f"📁 Kết quả đã lưu tại: {splits_dir}")


# # === 7️⃣ Hàm wrapper chính ===
# def run_split_dataset():
#     """
#     Hàm gọi mặc định từ main.py
#     """
#     # CẤU HÌNH ĐƯỜNG DẪN (Bạn sửa lại cho khớp với máy mình)
#     BASE_DATA_DIR = r"E:\NCKH - 2026\ECG Project\data\processed"
    
#     # Input
#     METADATA = os.path.join(BASE_DATA_DIR, "patient_metadata_superclean.xlsx")
#     PROCESSED_DIR = os.path.join(BASE_DATA_DIR, "ecg_leads12_fixed_amplitude") # Folder chứa .npy mới
    
#     # Output Folder Name
#     OUTPUT_NAME = "splits_fixed_amplitude"

#     # DANH SÁCH NHÃN (Target Columns)
#     # Nếu trong Excel đã có tên tiếng Việt/Anh thì điền vào đây.
#     # Nếu chưa có, điền mã SNOMED code, code sẽ tự sinh cột từ 'diagnosis_codes'
    
#     # Ví dụ dùng 7 nhãn lâm sàng (như bạn đã cấu hình trước đó):
#     # TARGET_COLS = [
#     #     "Nhịp chậm xoang - SB", 
#     #     "Nhịp xoang bình thường - SR", 
#     #     "Cuồng nhĩ - AF", 
#     #     "Rung nhĩ - AFIB", 
#     #     "Nhịp nhanh trên thất - SVT",
#     #     "Nhịp xoang nhanh - ST", 
#     #     "Phì đại thất trái - LVH"
#     # ]
    
#     # Ví dụ nếu muốn dùng SNOMED codes (như file cũ):
#     TARGET_COLS = ["426177001","426783006","164890007","164889003", "426761007", "427084000","164873001"]

#     try:
#         split_dataset(
#             processed_dir=PROCESSED_DIR,
#             metadata_path=METADATA,
#             target_cols=TARGET_COLS,
#             test_size=0.1, # 10% Test
#             val_size=0.1,  # 10% Val
#             random_state=42,
#             output_dir_name=OUTPUT_NAME
#         )
#     except Exception as e:
#         print(f"\n❌ Lỗi xảy ra: {e}")

# # === Main Block ===
# if __name__ == "__main__":
#     run_split_dataset()





# """
# processing/split_dataset.py
# ---------------------------
# Tạo nhãn multi-hot theo SNOMED CT và chia dữ liệu train/val/test.
# Lưu kết quả ra thư mục splits/ để main.py có thể load trực tiếp.
# """

# import os
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split


# def create_multilabel_targets(df: pd.DataFrame, snomed_codes: list):
#     """
#     Tạo cột multi-hot nhãn từ danh sách SNOMED code.

#     Args:
#         df (pd.DataFrame): Bảng thông tin bệnh nhân, có cột 'diagnosis_codes'.
#         snomed_codes (list[str]): Danh sách mã bệnh được giữ lại.

#     Returns:
#         tuple: (df mới có cột mã bệnh, y: ma trận nhãn (num_samples, num_codes))
#     """
#     df = df.copy()
#     for code in snomed_codes:
#         df[code] = df["diagnosis_codes"].apply(
#             lambda x: 1 if isinstance(x, str) and code in x.split(",") else 0
#         )
#     y = df[snomed_codes].values
#     return df, y


# def split_dataset(
#     save_dir: str,
#     df_resolved: pd.DataFrame,
#     snomed_codes: list,
#     test_size=0.3,
#     val_ratio=2 / 3,
#     random_state=42,
# ):
#     """
#     Chia dữ liệu đã xử lý thành train/val/test và lưu ra thư mục splits/.

#     Args:
#         save_dir (str): Thư mục chứa các file .npy đã được xử lý (từ bước feature_extraction).
#         df_resolved (pd.DataFrame): Bảng mapping giữa file path và chẩn đoán.
#         snomed_codes (list[str]): Danh sách SNOMED code cần giữ lại.
#         test_size (float): Tỷ lệ test.
#         val_ratio (float): Tỷ lệ val trong phần test+val.
#         random_state (int): Seed cố định.
#     """
#     # === 1️⃣ Tạo nhãn multi-hot ===
#     df_resolved, y = create_multilabel_targets(df_resolved, snomed_codes)

#     # === 2️⃣ Lấy danh sách file .npy ===
#     all_files = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith(".npy")]
#     all_files.sort()  # đảm bảo index nhất quán với df_resolved

#     if len(all_files) == 0:
#         raise FileNotFoundError(f"Không tìm thấy file .npy nào trong thư mục: {save_dir}")

#     print(f"🔹 Tổng số file tín hiệu ECG: {len(all_files)}")

#     # === 3️⃣ Chia train/val/test ===
#     train_files, temp_files = train_test_split(
#         all_files, test_size=test_size, random_state=random_state
#     )
#     val_files, test_files = train_test_split(
#         temp_files, test_size=val_ratio, random_state=random_state
#     )

#     print(f"✅ Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

#     # === 4️⃣ Hàm lấy index từ tên file (record_123.npy -> 123) ===
#     def get_index_from_filename(path):
#         fname = os.path.basename(path)
#         try:
#             idx = int(fname.replace("record_", "").replace(".npy", ""))
#             return idx
#         except ValueError:
#             raise ValueError(f"Tên file không hợp lệ: {fname}")

#     # === 5️⃣ Ánh xạ nhãn tương ứng ===
#     y_train = np.array([y[get_index_from_filename(f)] for f in train_files])
#     y_val = np.array([y[get_index_from_filename(f)] for f in val_files])
#     y_test = np.array([y[get_index_from_filename(f)] for f in test_files])

#     print("y_train shape:", y_train.shape)
#     print("y_val shape:", y_val.shape)
#     print("y_test shape:", y_test.shape)

#     # === 6️⃣ Lưu ra thư mục splits/ ===
#     splits_dir = os.path.join(save_dir, "..", "splits_C7v12_v2") #thay tên thư mục 
#     os.makedirs(splits_dir, exist_ok=True)

#     np.save(os.path.join(splits_dir, "train_files.npy"), np.array(train_files))
#     np.save(os.path.join(splits_dir, "val_files.npy"), np.array(val_files))
#     np.save(os.path.join(splits_dir, "test_files.npy"), np.array(test_files))
#     np.save(os.path.join(splits_dir, "y_train.npy"), y_train)
#     np.save(os.path.join(splits_dir, "y_val.npy"), y_val)
#     np.save(os.path.join(splits_dir, "y_test.npy"), y_test)

#     print(f"📁 Đã lưu toàn bộ file chia tập vào: {splits_dir}")

#     return {
#         "train": {"X": train_files, "y": y_train},
#         "val": {"X": val_files, "y": y_val},
#         "test": {"X": test_files, "y": y_test},
#     }


# # === 7️⃣ Hàm wrapper để gọi từ main.py ===
# def run_split_dataset():
#     """
#     Hàm tiện ích để gọi từ main.py — tự động đọc metadata và chạy chia tập.
#     """
#     print("\n===== Bước 3: Chia dữ liệu train/val/test =====")

#     # Đường dẫn tương đối
#     base_dir = os.path.dirname(os.path.dirname(__file__))
#     metadata_path = r"E:\NCKH - 2026\ECG Project\data\processed\patient_metadata_superclean.xlsx"
#     processed_dir = r"E:\NCKH - 2026\ECG Project\data\processed\ecg_leads12_7c_v2"


#     # Load metadata
#     if not os.path.exists(metadata_path):
#         raise FileNotFoundError(f"Không tìm thấy file metadata: {metadata_path}")

#     df = pd.read_excel(metadata_path)

#     # Danh sách mã SNOMED chính (ví dụ top 10)
#     # snomed_top10 = [
#     #     "426177001","426783006","164890007","164889003","429622005",
#     #     "713422000","111975006","164930006","426761007", "164873001",
#     #     "427084000","164934002","59931005","39732003","164917005"
#     # ]
#     # snomed_top10 = [
#     #     "426177001","426783006","164890007","164889003", "164873001",
#     #     "427084000","164934002"
#     # ]
    
#     # snomed_top10 = [
#     #     "426177001","426783006","164890007","164889003", "426761007",
#     #     "427084000","713422000"
#     # ]
    
#     snomed_top10 = [
#         "426177001","426783006","164890007","164889003", "426761007",
#         "427084000","164873001"
#     ]

#     # Chạy chia tập
#     split_dataset(save_dir=processed_dir, df_resolved=df, snomed_codes=snomed_top10)

#     print("✅ Hoàn tất chia dữ liệu!\n")


# # === Ví dụ chạy độc lập ===
# if __name__ == "__main__":
#     run_split_dataset()
