"""Leakage sanity check: correlation of every feature with the target."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = ROOT / "data" / "crop_data_clean.csv"
REPORT_PATH = ROOT / "data" / "leakage_report.csv"

TARGET = "low_yield_risk"
THRESHOLD = 0.9
CAT_COLS = ["region", "soil_type", "crop", "weather_condition"]


def correlation_with_target(df):
    # One-hot encode categoricals so every feature can be correlated
    encoded = pd.get_dummies(df, columns=CAT_COLS, dtype=int)
    corr = encoded.corr()[TARGET].drop(TARGET)
    out = corr.to_frame("correlation")
    out["abs_correlation"] = out["correlation"].abs()
    out["flagged"] = out["abs_correlation"] > THRESHOLD
    return out.sort_values("abs_correlation", ascending=False)


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH)
    report = correlation_with_target(df)
    report.to_csv(REPORT_PATH)

    print("Correlation of each feature with the target (top 15):")
    print(report.head(15).round(4).to_string())
    flagged = report[report["flagged"]]
    print(f"\nColumns above {THRESHOLD}: {len(flagged)}")
    if len(flagged):
        print(flagged.round(4).to_string())
    else:
        print("No column exceeds the threshold.")