# AgriPak Farms: Low-Yield Risk Prediction

Machine learning project that flags fields at risk of a low yield before harvest, so AgriPak Farms can act in time. Week 1 covers repository setup, data cleaning, a reproducible train-test split, a baseline model, and a leakage sanity check.

## Repository structure

```
agripak-low-yield-risk/
├── data/
│   ├── crop_data.csv          # raw dataset (50,000 rows)
│   ├── crop_data_clean.csv    # cleaned dataset
│   ├── train.csv              # 80% training split
│   ├── test.csv               # 20% test split
│   └── leakage_report.csv     # correlation of each feature with the target
├── src/
│   ├── data_prep.py           # load and clean
│   ├── split_data.py          # stratified train-test split
│   ├── train_baseline.py      # Logistic Regression baseline
│   └── leakage_check.py       # leakage sanity check
├── notebooks/
├── requirements.txt
├── LICENSE
└── README.md
```

## How to run the pipeline

```bash
pip install -r requirements.txt
python src/data_prep.py
python src/split_data.py
python src/train_baseline.py
python src/leakage_check.py
```

Run the commands from the project root, in this order.

## Dataset

`crop_data.csv` has 50,000 rows and 10 columns. The target is `low_yield_risk` (1 = low yield, 0 = not). About 35% of rows are 1.

| Type | Columns |
|---|---|
| Categorical | region, soil_type, crop, weather_condition |
| Numeric | rainfall_mm, temperature_c, days_to_harvest |
| Binary | fertilizer_used, irrigation_used |

## Cleaning decisions

| Check | Finding | Decision |
|---|---|---|
| Column names | Already consistent | Strip spaces and lowercase anyway, for safety |
| Duplicate rows | 0 found | Script drops exact duplicates if any appear |
| Missing values | 0 found | Numeric: fill with median. Categorical: fill with mode. Rows missing the target are dropped. |
| Text columns | Clean | Strip whitespace and standardise capitalisation |
| Data types | Binary columns stored as integers | Cast binary columns and target to int, categoricals to string |
| Value ranges | Rainfall 100 to 1000 mm, temperature 15 to 40 C, days to harvest 60 to 149 | Script asserts rainfall is non-negative and binary columns are only 0/1 |

Result: 50,000 rows before cleaning and 50,000 after. The dataset was already clean, and the script is written to handle problems if they appear in future data.

## Train-test split

- 80% train (40,000 rows) and 20% test (10,000 rows)
- Stratified on `low_yield_risk`, with a fixed random seed of 42
- Target rate is 0.3500 in both the train and test sets
- Reproducible: running `src/split_data.py` always produces the same split

## Baseline model

Logistic Regression in a scikit-learn Pipeline. Numeric columns are scaled and categorical columns are one-hot encoded, with both fitted on the training data only.

| Metric | Test score |
|---|---|
| Accuracy | 0.9187 |
| Precision | 0.8886 |
| Recall | 0.8777 |
| F1 | 0.8831 |

For reference, always predicting "no low-yield risk" would give 0.6500 accuracy, so the baseline is well above the majority-class benchmark.

## Leakage findings

I computed the correlation of every feature (categoricals one-hot encoded) with `low_yield_risk` and flagged any column with an absolute correlation above 0.9.

**Flagged columns: none.** The highest correlations were:

| Feature | Correlation | Verdict | Reason |
|---|---|---|---|
| rainfall_mm | -0.6364 | Genuine predictor | Known before harvest. More rain means lower risk, which makes agronomic sense. |
| fertilizer_used | -0.3342 | Genuine predictor | A farm decision made before harvest. Fertilizer lowers risk. |
| irrigation_used | -0.2662 | Genuine predictor | A farm decision made before harvest. Irrigation lowers risk. |
| temperature_c | -0.0685 | Genuine predictor (weak) | Weather measurement available before harvest. |
| days_to_harvest | 0.0027 | Not leakage | Essentially uncorrelated with the target. |

All other features have an absolute correlation below 0.01.

**Caveat:** a pairwise correlation scan only catches strong single-feature leakage. It does not detect leakage that works through combinations of features, so this check will be revisited in later weeks.

## License

MIT# agripak-low-yield-risk