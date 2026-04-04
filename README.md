# AQI Fuzzy Logic Prediction

This project builds an Air Quality Index (AQI) prediction pipeline using:

- Wang-Mendel for fuzzy rule extraction
- Mamdani fuzzy inference for prediction
- CPCB-aligned trapezoidal and triangular membership functions

The current project baseline uses:

- cleaned dataset: `final.csv`
- inputs: `PM2.5, PM10, NO2, CO, O3, SO2`
- target: numeric `AQI`
- metadata retained in the dataset: `City, Date, AQI_Bucket`

## Dataset

Raw source:

- `city_day.csv`

Cleaned modeling dataset:

- `final.csv`

Final cleaned columns:

- `City`
- `Date`
- `PM2.5`
- `PM10`
- `NO2`
- `CO`
- `O3`
- `SO2`
- `AQI`
- `AQI_Bucket`

## Preprocessing

The cleaned dataset is created by [preprocess.py](/Users/parthsrivastava/Desktop/SC/preprocess.py).

Preprocessing steps:

1. Keep only the required columns
2. Drop rows where `AQI` is null
3. Drop rows where all 6 pollutant inputs are null
4. Drop rows where `PM10` is null
5. Impute remaining nulls in `PM2.5, NO2, CO, O3, SO2` using city-wise median
6. Drop rows still containing nulls
7. Cap outliers using CPCB upper bounds
8. Drop duplicate rows

Final cleaned dataset summary:

- rows: `17,612`
- columns: `10`
- nulls: `0`
- duplicates: `0`

## Fuzzy Logic Pipeline

The implementation is modular and lives in [fuzzy_aqi](/Users/parthsrivastava/Desktop/SC/fuzzy_aqi/__init__.py).

Main modules:

- [config.py](/Users/parthsrivastava/Desktop/SC/fuzzy_aqi/config.py): shared constants
- [data.py](/Users/parthsrivastava/Desktop/SC/fuzzy_aqi/data.py): dataset loading and train/test split
- [membership.py](/Users/parthsrivastava/Desktop/SC/fuzzy_aqi/membership.py): CPCB-style trap/triangle membership functions
- [wang_mendel.py](/Users/parthsrivastava/Desktop/SC/fuzzy_aqi/wang_mendel.py): rule extraction
- [mamdani.py](/Users/parthsrivastava/Desktop/SC/fuzzy_aqi/mamdani.py): inference and defuzzification
- [evaluate.py](/Users/parthsrivastava/Desktop/SC/fuzzy_aqi/evaluate.py): metrics and output export
- [run_fuzzy_evaluation.py](/Users/parthsrivastava/Desktop/SC/run_fuzzy_evaluation.py): run script

Pipeline:

1. Load `final.csv`
2. Split data into train/test with stratification by `AQI_Bucket`
3. Build fuzzy membership functions
4. Extract fuzzy rules using Wang-Mendel on training data
5. Run Mamdani inference on the test set
6. Predict numeric AQI
7. Convert predicted AQI to AQI buckets
8. Evaluate numeric and bucket-level accuracy

## Membership Functions

The project uses CPCB-style semantic memberships:

- leftmost set: trapezoidal, open left
- middle sets: triangular
- rightmost set: trapezoidal, open right

This is used for both:

- Wang-Mendel rule extraction
- Mamdani inference

Examples:

- lower pollution levels get full membership in clean categories
- extreme pollution levels stay fully active in severe categories

## Current Baseline Results

Latest verified baseline on `final.csv`:

- total rows: `17,612`
- train rows: `14,089`
- test rows: `3,523`
- learned rules: `241`

Accuracy:

- `RMSE = 33.79`
- `MAE = 23.73`
- `R² = 0.8768`
- exact bucket accuracy = `66.25%`
- within 1 bucket accuracy = `98.78%`

## Output Files

Generated after evaluation:

- [fuzzy_rule_base.json](/Users/parthsrivastava/Desktop/SC/fuzzy_rule_base.json): extracted fuzzy rules
- [fuzzy_test_predictions.csv](/Users/parthsrivastava/Desktop/SC/fuzzy_test_predictions.csv): test predictions with errors

## How To Run

Preprocess the raw dataset:

```bash
python3 preprocess.py
```

Run fuzzy evaluation:

```bash
python3 run_fuzzy_evaluation.py
```

## Notes

- This project currently keeps only the baseline trap/triangle fuzzy model.
- Gaussian membership experiments were removed.
- `Date` is retained as metadata only and is not used as a fuzzy input.
- No normalization or scaling is used because fuzzy membership functions handle the raw value ranges directly.
