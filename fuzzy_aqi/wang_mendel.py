from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fuzzy_aqi.config import INPUT_COLUMNS, TARGET_COLUMN
from fuzzy_aqi.membership import best_label


@dataclass(frozen=True)
class FuzzyRule:
    antecedent: tuple[str, ...]
    consequent: str
    strength: float
    support: int
    confidence: float


def extract_rules(
    train_df: pd.DataFrame,
    membership_defs: dict[str, dict[str, dict[str, object]]],
    min_support: int = 2,
    min_confidence: float = 0.2,
) -> list[FuzzyRule]:
    antecedent_totals: dict[tuple[str, ...], float] = {}
    antecedent_supports: dict[tuple[str, ...], int] = {}
    consequent_stats: dict[tuple[str, ...], dict[str, dict[str, float]]] = {}

    for _, row in train_df.iterrows():
        antecedent_labels = []
        strength = 1.0

        for column in INPUT_COLUMNS:
            label, degree = best_label(float(row[column]), membership_defs[column])
            antecedent_labels.append(label)
            strength *= degree

        consequent_label, consequent_degree = best_label(
            float(row[TARGET_COLUMN]),
            membership_defs[TARGET_COLUMN],
        )
        strength *= consequent_degree

        antecedent = tuple(antecedent_labels)
        antecedent_totals[antecedent] = antecedent_totals.get(antecedent, 0.0) + strength
        antecedent_supports[antecedent] = antecedent_supports.get(antecedent, 0) + 1

        label_stats = consequent_stats.setdefault(antecedent, {}).setdefault(
            consequent_label,
            {"support": 0.0, "strength_sum": 0.0},
        )
        label_stats["support"] += 1
        label_stats["strength_sum"] += strength

    max_support = max(antecedent_supports.values()) if antecedent_supports else 1
    rules: list[FuzzyRule] = []

    for antecedent, label_map in consequent_stats.items():
        total_strength = antecedent_totals[antecedent]
        support = antecedent_supports[antecedent]
        consequent, stats = max(
            label_map.items(),
            key=lambda item: (item[1]["strength_sum"], item[1]["support"]),
        )
        confidence = float(stats["strength_sum"] / total_strength) if total_strength else 0.0
        avg_strength = float(stats["strength_sum"] / stats["support"]) if stats["support"] else 0.0
        support_factor = support / max_support
        weight = avg_strength * confidence * support_factor

        if support < min_support or confidence < min_confidence:
            continue

        rules.append(
            FuzzyRule(
                antecedent=antecedent,
                consequent=consequent,
                strength=weight,
                support=int(stats["support"]),
                confidence=confidence,
            )
        )

    if not rules:
        rules = [
            FuzzyRule(
                antecedent=antecedent,
                consequent=max(
                    label_map.items(),
                    key=lambda item: (item[1]["strength_sum"], item[1]["support"]),
                )[0],
                strength=1.0,
                support=antecedent_supports[antecedent],
                confidence=1.0,
            )
            for antecedent, label_map in consequent_stats.items()
        ]

    return sorted(rules, key=lambda rule: (rule.antecedent, rule.consequent))
