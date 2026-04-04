from __future__ import annotations

import numpy as np
import pandas as pd

from fuzzy_aqi.config import INPUT_COLUMNS, TARGET_COLUMN
from fuzzy_aqi.data import load_dataset, split_dataset
from fuzzy_aqi.mamdani import infer_aqi
from fuzzy_aqi.membership import (
    build_membership_definitions,
    bucket_from_aqi,
    evaluate_membership,
    fuzzify_value,
    output_universe,
)
from fuzzy_aqi.wang_mendel import FuzzyRule, extract_rules


def build_model(csv_path: str = "final.csv") -> dict[str, object]:
    df = load_dataset(csv_path)
    train_df, _ = split_dataset(
        df,
        test_size=0.2,
        random_state=42,
        stratify_column="AQI_Bucket",
    )
    train_model = train_df[INPUT_COLUMNS + [TARGET_COLUMN]].copy()
    membership_defs = build_membership_definitions(train_model)
    rules = extract_rules(train_model, membership_defs)
    universe = output_universe()
    output_shapes = {
        label: np.array(
            [evaluate_membership(value, membership_defs[TARGET_COLUMN][label]) for value in universe]
        )
        for label in membership_defs[TARGET_COLUMN]
    }
    output_centers = {
        label: float(np.sum(universe * shape) / np.sum(shape))
        for label, shape in output_shapes.items()
    }
    rule_coverage = summarize_rule_coverage(rules)
    return {
        "membership_defs": membership_defs,
        "rules": rules,
        "universe": universe,
        "output_shapes": output_shapes,
        "output_centers": output_centers,
        "rule_coverage": rule_coverage,
    }


def predict_single(inputs: dict[str, float], model: dict[str, object]) -> dict[str, object]:
    row = pd.Series({column: float(inputs[column]) for column in INPUT_COLUMNS})
    prediction, inference_info = infer_aqi(
        row=row,
        rules=model["rules"],  # type: ignore[arg-type]
        membership_defs=model["membership_defs"],  # type: ignore[arg-type]
        universe=model["universe"],  # type: ignore[arg-type]
        output_shapes=model["output_shapes"],  # type: ignore[arg-type]
        output_centers=model["output_centers"],  # type: ignore[arg-type]
    )

    membership_defs = model["membership_defs"]  # type: ignore[assignment]
    input_memberships = {
        column: fuzzify_value(float(inputs[column]), membership_defs[column])
        for column in INPUT_COLUMNS
    }

    fired_rules = top_fired_rules(
        row=row,
        rules=model["rules"],  # type: ignore[arg-type]
        membership_defs=membership_defs,
        limit=5,
    )

    return {
        "predicted_aqi": prediction,
        "predicted_bucket": bucket_from_aqi(prediction),
        "input_memberships": input_memberships,
        "top_rules": fired_rules,
        "inference_info": inference_info,
        "rule_coverage": model["rule_coverage"],
    }


def summarize_rule_coverage(rules: list[FuzzyRule]) -> list[dict[str, object]]:
    coverage = {}
    for rule in rules:
        stats = coverage.setdefault(
            rule.consequent,
            {"rule_count": 0, "support_sum": 0, "mean_confidence_sum": 0.0},
        )
        stats["rule_count"] += 1
        stats["support_sum"] += int(rule.support)
        stats["mean_confidence_sum"] += float(rule.confidence)

    summary = []
    for consequent, stats in coverage.items():
        rule_count = int(stats["rule_count"])
        summary.append(
            {
                "consequent": consequent,
                "rule_count": rule_count,
                "support_sum": int(stats["support_sum"]),
                "mean_confidence": float(stats["mean_confidence_sum"] / rule_count),
            }
        )
    summary.sort(key=lambda item: item["consequent"])
    return summary


def top_fired_rules(
    row: pd.Series,
    rules: list[FuzzyRule],
    membership_defs: dict[str, dict[str, dict[str, object]]],
    limit: int = 5,
) -> list[dict[str, object]]:
    memberships = {
        column: fuzzify_value(float(row[column]), membership_defs[column])
        for column in INPUT_COLUMNS
    }
    scored_rules = []
    for rule in rules:
        antecedent_activation = min(
            memberships[column][label]
            for column, label in zip(INPUT_COLUMNS, rule.antecedent)
        )
        firing_strength = antecedent_activation * rule.strength
        if firing_strength <= 0.0:
            continue
        scored_rules.append(
            {
                "antecedent": dict(zip(INPUT_COLUMNS, rule.antecedent)),
                "consequent": rule.consequent,
                "firing_strength": float(firing_strength),
                "support": rule.support,
                "confidence": rule.confidence,
            }
        )
    scored_rules.sort(key=lambda item: item["firing_strength"], reverse=True)
    return scored_rules[:limit]
