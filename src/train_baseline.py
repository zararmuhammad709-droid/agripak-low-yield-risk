"""Train a Logistic Regression baseline and report test-set metrics."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"

TARGET = "low_yield_risk"
NUM_COLS = ["rainfall_mm", "temperature_c", "days_to_harvest"]
CAT_COLS = ["region", "soil_type", "crop", "weather_condition"]
BINARY_COLS = ["fertilizer_used", "irrigation_used"]


def build_model():
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("bin", "passthrough", BINARY_COLS),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])


if __name__ == "__main__":
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]

    model = build_model()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    majority = max(y_test.mean(), 1 - y_test.mean())
    print("Baseline: Logistic Regression (test set)")
    print(f"  Accuracy:  {accuracy_score(y_test, pred):.4f}")
    print(f"  Precision: {precision_score(y_test, pred):.4f}")
    print(f"  Recall:    {recall_score(y_test, pred):.4f}")
    print(f"  F1:        {f1_score(y_test, pred):.4f}")
    print(f"  (Always-predict-majority accuracy: {majority:.4f})")