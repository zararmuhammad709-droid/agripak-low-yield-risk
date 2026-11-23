"""Create a reproducible, stratified 80/20 train-test split."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = ROOT / "data" / "crop_data_clean.csv"
TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"

TARGET = "low_yield_risk"
SEED = 42
TEST_SIZE = 0.20


def make_split(df):
    return train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET],
        random_state=SEED,
    )


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH)
    train, test = make_split(df)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    print(f"Train rows: {len(train)}")
    print(f"Test rows:  {len(test)}")
    print(f"Train target rate: {train[TARGET].mean():.4f}")
    print(f"Test target rate:  {test[TARGET].mean():.4f}")
    print(f"Saved to {TRAIN_PATH} and {TEST_PATH}")