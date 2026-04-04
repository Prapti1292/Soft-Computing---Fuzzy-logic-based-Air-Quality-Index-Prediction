from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fuzzy_aqi.config import INPUT_COLUMNS, TARGET_COLUMN
from fuzzy_aqi.data import load_dataset, split_dataset
from fuzzy_aqi.mamdani import predict
from fuzzy_aqi.membership import (
    bucket_distance,
    bucket_from_aqi,
    build_membership_definitions,
)
from fuzzy_aqi.wang_mendel import extract_rules


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mean_true = float(np.mean(y_true))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - mean_true) ** 2))
    return 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0


def run_evaluation(
    csv_path: str = "final.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    df = load_dataset(csv_path)
    train_df, test_df = split_dataset(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify_column="AQI_Bucket",
    )

    train_model = train_df[INPUT_COLUMNS + [TARGET_COLUMN]].copy()
    test_model = test_df[INPUT_COLUMNS + [TARGET_COLUMN]].copy()

    membership_defs = build_membership_definitions(train_model)
    rules = extract_rules(train_model, membership_defs)
    predictions = predict(test_model, rules, membership_defs)
    actuals = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    predicted_buckets = [bucket_from_aqi(value) for value in predictions]
    actual_buckets = test_df["AQI_Bucket"].tolist()
    bucket_accuracy = float(np.mean(np.array(predicted_buckets) == np.array(actual_buckets)))
    within_one_bucket = float(
        np.mean([bucket_distance(actual, predicted) <= 1 for actual, predicted in zip(actual_buckets, predicted_buckets)])
    )
    confusion_matrix = pd.crosstab(
        pd.Series(actual_buckets, name="Actual_Bucket"),
        pd.Series(predicted_buckets, name="Predicted_Bucket"),
    )

    metrics = {
        "dataset": csv_path,
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "input_columns": INPUT_COLUMNS,
        "rule_count": int(len(rules)),
        "rmse": rmse(actuals, predictions),
        "mae": mae(actuals, predictions),
        "r2": r2(actuals, predictions),
        "bucket_accuracy": bucket_accuracy,
        "within_1_bucket_accuracy": within_one_bucket,
        "mean_actual_aqi": float(np.mean(actuals)),
        "mean_predicted_aqi": float(np.mean(predictions)),
        "confusion_matrix": confusion_matrix.to_dict(),
    }

    results_df = test_df.copy()
    results_df["Predicted_AQI"] = predictions
    results_df["Predicted_Bucket"] = predicted_buckets
    results_df["Absolute_Error"] = (results_df[TARGET_COLUMN] - results_df["Predicted_AQI"]).abs()
    results_df["Squared_Error"] = (results_df[TARGET_COLUMN] - results_df["Predicted_AQI"]) ** 2
    results_df.to_csv("fuzzy_test_predictions.csv", index=False)

    rule_base_json = json.dumps(
        [
            {
                "if": dict(zip(INPUT_COLUMNS, rule.antecedent)),
                "then": rule.consequent,
                "strength": rule.strength,
                "support": rule.support,
                "confidence": rule.confidence,
            }
            for rule in rules
        ],
        indent=2,
    )
    Path("fuzzy_rule_base.json").write_text(rule_base_json)

    metrics["predictions_path"] = "fuzzy_test_predictions.csv"
    metrics["rule_base_path"] = "fuzzy_rule_base.json"
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="final.csv")
    args = parser.parse_args()

    metrics = run_evaluation(csv_path=args.csv)
    print("Fuzzy AQI Evaluation")
    for key, value in metrics.items():
        if key == "confusion_matrix":
            print("confusion_matrix:")
            print(pd.DataFrame(value).fillna(0).astype(int).to_string())
            continue
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
