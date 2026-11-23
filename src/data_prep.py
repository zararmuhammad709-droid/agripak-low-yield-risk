"""Load and clean the AgriPak crop dataset."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "crop_data.csv"
CLEAN_PATH = ROOT / "data" / "crop_data_clean.csv"

TARGET = "low_yield_risk"
CAT_COLS = ["region", "soil_type", "crop", "weather_condition"]
BINARY_COLS = ["fertilizer_used", "irrigation_used", TARGET]


def load_data(path=RAW_PATH):
    return pd.read_csv(path)


def clean_data(df):
    df = df.copy()
    report = {"rows_before": len(df)}

    # 1. Tidy column names
    df.columns = df.columns.str.strip().str.lower()

    # 2. Remove exact duplicate rows
    report["duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # 3. Clean text columns (strip spaces, consistent capitalisation)
    for c in CAT_COLS:
        df[c] = df[c].astype("string").str.strip().str.title()

    # 4. Missing values: numeric -> median, categorical -> mode
    report["missing_before"] = int(df.isna().sum().sum())
    num_cols = df.select_dtypes("number").columns.drop(TARGET)
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for c in CAT_COLS:
        df[c] = df[c].fillna(df[c].mode()[0])
    df = df.dropna(subset=[TARGET])  # rows without a label are unusable

    # 5. Fix data types
    for c in BINARY_COLS:
        df[c] = df[c].astype(int)
    for c in CAT_COLS:
        df[c] = df[c].astype(str)

    # 6. Sanity checks on value ranges
    assert (df["rainfall_mm"] >= 0).all(), "negative rainfall found"
    assert df[BINARY_COLS].isin([0, 1]).all().all(), "binary column not 0/1"

    report["rows_after"] = len(df)
    return df.reset_index(drop=True), report


if __name__ == "__main__":
    raw = load_data()
    clean, report = clean_data(raw)
    clean.to_csv(CLEAN_PATH, index=False)
    print("Cleaning report:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"Saved cleaned data to {CLEAN_PATH}")