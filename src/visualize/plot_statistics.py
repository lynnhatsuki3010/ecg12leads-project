# visualize/plot_statistics.py
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

def analyze_top_diseases(metadata_path: str, snomed_path: str, top_n: int = 15):
    df = pd.read_excel(metadata_path)
    snomed = pd.read_csv(snomed_path)
    diagnosis_dict = dict(zip(snomed["Snomed_CT"].astype(str), snomed["Acronym Name"]))

    all_codes = []
    for codes_str in df["diagnosis_codes"]:
        if isinstance(codes_str, str) and codes_str.strip() != "":
            all_codes.extend(codes_str.split(","))

    code_counts = Counter(all_codes)
    total = sum(code_counts.values())
    top = code_counts.most_common(top_n)

    df_top = pd.DataFrame(top, columns=["Code", "Count"])
    df_top["Disease_Name"] = df_top["Code"].map(lambda x: diagnosis_dict.get(x, "Unknown"))
    df_top["Percentage"] = df_top["Count"] / total * 100

    print(df_top)

    plt.figure(figsize=(10, 6))
    plt.barh(df_top["Disease_Name"], df_top["Count"], color="skyblue")
    plt.xlabel("Count")
    plt.ylabel("Disease")
    plt.title(f"Top {top_n} Diseases in ECG Dataset")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_top_diseases(
        r"E:\NCKH - 2026\ECG Project\data\processed\patient_metadata_clean.xlsx",
        r"E:\NCKH - 2026\ECG Project\data\raw\ConditionNames_SNOMED-CT.csv"
    )
