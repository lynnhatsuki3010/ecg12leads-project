# processing/preprocess.py
import os
import time
import requests
import pandas as pd
import wfdb


# --- 1. Đọc WFDB metadata ---
def create_metadata_from_wfdb(root_dir: str) -> pd.DataFrame:
    records = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".hea"):
                file_path = os.path.join(subdir, file)
                record_id = os.path.splitext(file)[0]

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
                    "record_id": record_id,
                    "file_path": file_path,
                    "age": age,
                    "sex": sex,
                    "diagnosis_codes": dx
                })

    df = pd.DataFrame(records)
    print(f"✅ Tạo metadata xong: {len(df)} bản ghi")
    return df


# --- 2. Mapping ban đầu từ file SNOMED (có thể chứa Unknown) ---
def map_snomed_codes(df: pd.DataFrame, snomed_path: str) -> pd.DataFrame:
    cond_map = pd.read_csv(snomed_path)
    code2name = dict(zip(cond_map["Snomed_CT"].astype(str), cond_map["Acronym Name"]))

    def map_codes(codes):
        if pd.isna(codes):
            return None
        return ",".join([code2name.get(c.strip(), "Unknown") for c in codes.split(",")])

    df["diagnosis_names"] = df["diagnosis_codes"].apply(map_codes)
    print("🧩 Mapping SNOMED lần đầu (có thể còn Unknown)")
    return df


# --- 3. Hàm gọi API FHIR để resolve các mã Unknown ---
def parse_lookup_response(data):
    params = data.get("parameter", [])
    for p in params:
        if p.get("name") == "display" and "valueString" in p:
            return p["valueString"]
    for p in params:
        if p.get("name") == "designation" and "part" in p:
            for part in p["part"]:
                if part.get("name") == "value" and "valueString" in part:
                    return part["valueString"]
    return None


def lookup_snomed_name(code):
    url = (
        "https://snowstorm-training.snomedtools.org/snowstorm/snomed-ct/fhir/CodeSystem/$lookup"
        f"?system=http://snomed.info/sct&code={code}"
    )
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            return parse_lookup_response(r.json())
        else:
            return None
    except Exception:
        return None


def resolve_unknown_snomed(cond_path: str, save_path: str):
    df_unknown = pd.read_csv(cond_path)
    resolved_names = []

    for code, name in zip(df_unknown["Snomed_CT"], df_unknown["Acronym Name"]):
        if name == "Unknown":
            resolved = lookup_snomed_name(str(code))
            resolved_names.append(resolved if resolved else "Unknown")
        else:
            resolved_names.append(name)
        time.sleep(0.2)  # tránh spam API

    df_unknown["Resolved_Name"] = resolved_names
    df_unknown.to_excel(save_path, index=False)
    print(f"✅ Đã resolve SNOMED Unknown -> {save_path}")
    return df_unknown


# --- 4. Cập nhật tên bệnh vào bảng bệnh nhân ---
def update_diagnosis_names(patient_df: pd.DataFrame, resolved_file: str) -> pd.DataFrame:
    df_resolved = pd.read_excel(resolved_file)
    resolved_dict = dict(zip(df_resolved["Snomed_CT"].astype(str), df_resolved["Resolved_Name"]))
    print(f"🔍 Tải {len(resolved_dict)} mã SNOMED đã resolved")

    def update_row(row):
        codes = str(row["diagnosis_codes"]).split(",")
        names = str(row["diagnosis_names"]).split(",")
        updated = []
        for code, name in zip(codes, names):
            if name == "Unknown":
                updated.append(resolved_dict.get(code.strip(), "Unknown"))
            else:
                updated.append(name)
        return ",".join(updated)

    patient_df["diagnosis_names"] = patient_df.apply(update_row, axis=1)
    return patient_df


# --- 5. Làm sạch dữ liệu ---
def clean_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["diagnosis_codes"].notna()]
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df[df["age"] > 0]
    df = df[df["sex"].astype(str).str.lower() != "unknown"]
    return df


# --- 6. Pipeline chính ---
def run_preprocessing():
    root_dir = r"E:\NCKH - 2026\ECG Project\data\raw\WFDBRecords"
    snomed_path = r"E:\NCKH - 2026\ECG Project\data\raw\ConditionNames_SNOMED-CT.csv"
    resolved_path = r"E:\NCKH - 2026\ECG Project\data\processed\Unknown_SNOMED_Codes_Resolved_NEW.xlsx"
    save_path = r"E:\NCKH - 2026\ECG Project\data\processed\patient_metadata_clean.xlsx"

    df = create_metadata_from_wfdb(root_dir)
    df = map_snomed_codes(df, snomed_path)

    # nếu chưa có file resolved thì tự gọi API để tạo
    if not os.path.exists(resolved_path):
        resolve_unknown_snomed(snomed_path, resolved_path)

    df = update_diagnosis_names(df, resolved_path)
    df = clean_metadata(df)

    df.to_excel(save_path, index=False)
    print(f"💾 Đã lưu metadata sạch: {save_path}")


if __name__ == "__main__":
    run_preprocessing()
