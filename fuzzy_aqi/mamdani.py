from __future__ import annotations

import numpy as np
import pandas as pd

from fuzzy_aqi.config import INPUT_COLUMNS, TARGET_COLUMN
from fuzzy_aqi.membership import evaluate_membership, fuzzify_value, output_universe
from fuzzy_aqi.wang_mendel import FuzzyRule


def antecedent_geometric_mean(values) -> float:
    values = list(values)
    if not values or any(value <= 0.0 for value in values):
        return 0.0
    return float(np.prod(values) ** (1.0 / len(values)))


def output_label_order(membership_defs: dict[str, dict[str, dict[str, object]]]) -> dict[str, int]:
    return {
        label: index
        for index, label in enumerate(membership_defs[TARGET_COLUMN].keys())
    }


def dominant_input_floor(
    input_memberships: dict[str, dict[str, float]],
    label_order: dict[str, int],
) -> int:
    dominant_indices = []
    for memberships in input_memberships.values():
        label, degree = max(memberships.items(), key=lambda item: item[1])
        if degree > 0.0 and label in label_order:
            dominant_indices.append(label_order[label])
    return max(dominant_indices) if dominant_indices else 0


def fallback_aqi(
    input_memberships: dict[str, dict[str, float]],
    rules: list[FuzzyRule],
    output_centers: dict[str, float],
    label_order: dict[str, int],
    top_k: int = 5,
) -> tuple[float, dict[str, object]]:
    floor_index = dominant_input_floor(input_memberships, label_order)
    scored_rules = []
    for rule in rules:
        matching_values = [
            input_memberships[column][label]
            for column, label in zip(INPUT_COLUMNS, rule.antecedent)
        ]
        compatibility = antecedent_geometric_mean(matching_values)
        backup_score = compatibility * rule.strength
        if backup_score <= 0.0:
            continue
        consequent_index = label_order.get(rule.consequent, 0)
        scored_rules.append((backup_score, consequent_index, rule))

    if not scored_rules:
        severe_center = output_centers.get("Severe", 400.0)
        return severe_center, {
            "used_fallback": True,
            "fallback_reason": "no_matching_rules",
            "fallback_rule_count": 0,
            "fallback_floor_index": floor_index,
        }

    eligible_rules = [item for item in scored_rules if item[1] >= floor_index]
    if not eligible_rules and floor_index > 0:
        eligible_rules = [item for item in scored_rules if item[1] >= floor_index - 1]
    if not eligible_rules:
        eligible_rules = scored_rules

    eligible_rules.sort(key=lambda item: item[0], reverse=True)
    top_rules = eligible_rules[:top_k]
    weights = np.array([score for score, _, _ in top_rules], dtype=float)
    centers = np.array([output_centers[rule.consequent] for _, _, rule in top_rules], dtype=float)
    prediction = float(np.sum(weights * centers) / np.sum(weights))
    return prediction, {
        "used_fallback": True,
        "fallback_reason": "weak_or_zero_aggregation",
        "fallback_rule_count": len(top_rules),
        "fallback_top_consequents": [rule.consequent for _, _, rule in top_rules],
        "fallback_floor_index": floor_index,
    }


def infer_aqi(
    row: pd.Series,
    rules: list[FuzzyRule],
    membership_defs: dict[str, dict[str, dict[str, object]]],
    universe: np.ndarray,
    output_shapes: dict[str, np.ndarray],
    output_centers: dict[str, float],
) -> tuple[float, dict[str, object]]:
    aggregated = np.zeros_like(universe)
    label_order = output_label_order(membership_defs)

    input_memberships = {
        column: fuzzify_value(float(row[column]), membership_defs[column])
        for column in INPUT_COLUMNS
    }
    fired_rule_count = 0
    max_firing_strength = 0.0

    for rule in rules:
        antecedent_activation = antecedent_geometric_mean(
            input_memberships[column][label]
            for column, label in zip(INPUT_COLUMNS, rule.antecedent)
        )
        firing_strength = antecedent_activation * rule.strength

        if firing_strength <= 0.0:
            continue

        fired_rule_count += 1
        max_firing_strength = max(max_firing_strength, float(firing_strength))
        output_shape = output_shapes[rule.consequent]
        aggregated = np.maximum(aggregated, np.minimum(firing_strength, output_shape))

    if aggregated.sum() == 0.0:
        return fallback_aqi(input_memberships, rules, output_centers, label_order)

    centroid = float(np.sum(universe * aggregated) / np.sum(aggregated))
    return centroid, {
        "used_fallback": False,
        "fallback_reason": None,
        "fired_rule_count": fired_rule_count,
        "max_firing_strength": max_firing_strength,
        "aggregation_operator": "geometric_mean",
    }


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
    output_centers = {
        label: float(np.sum(universe * shape) / np.sum(shape))
        for label, shape in output_shapes.items()
    }
    predictions = [
        infer_aqi(row, rules, membership_defs, universe, output_shapes, output_centers)[0]
        for _, row in df.iterrows()
    ]
    return np.array(predictions, dtype=float)
