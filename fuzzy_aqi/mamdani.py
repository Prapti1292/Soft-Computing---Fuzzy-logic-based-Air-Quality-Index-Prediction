from __future__ import annotations

import numpy as np
import pandas as pd

from fuzzy_aqi.config import INPUT_COLUMNS, TARGET_COLUMN
from fuzzy_aqi.membership import evaluate_membership, fuzzify_value, output_universe
from fuzzy_aqi.wang_mendel import FuzzyRule


def infer_aqi(
    row: pd.Series,
    rules: list[FuzzyRule],
    membership_defs: dict[str, dict[str, dict[str, object]]],
    universe: np.ndarray,
    output_shapes: dict[str, np.ndarray],
) -> float:
    aggregated = np.zeros_like(universe)

    input_memberships = {
        column: fuzzify_value(float(row[column]), membership_defs[column])
        for column in INPUT_COLUMNS
    }

    for rule in rules:
        antecedent_activation = min(
            input_memberships[column][label]
            for column, label in zip(INPUT_COLUMNS, rule.antecedent)
        )
        firing_strength = antecedent_activation * rule.strength

        if firing_strength <= 0.0:
            continue

        output_shape = output_shapes[rule.consequent]
        aggregated = np.maximum(aggregated, np.minimum(firing_strength, output_shape))

    if aggregated.sum() == 0.0:
        return float(np.mean(universe))

    centroid = float(np.sum(universe * aggregated) / np.sum(aggregated))
    return centroid


def predict(
    df: pd.DataFrame,
    rules: list[FuzzyRule],
    membership_defs: dict[str, dict[str, dict[str, object]]],
) -> np.ndarray:
    universe = output_universe()
    output_shapes = {
        label: np.array(
            [evaluate_membership(value, membership_defs[TARGET_COLUMN][label]) for value in universe]
        )
        for label in membership_defs[TARGET_COLUMN]
    }
    predictions = [
        infer_aqi(row, rules, membership_defs, universe, output_shapes)
        for _, row in df.iterrows()
    ]
    return np.array(predictions, dtype=float)
