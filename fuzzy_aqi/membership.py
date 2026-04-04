from __future__ import annotations

import numpy as np
import pandas as pd

from fuzzy_aqi.config import (
    AQI_UNIVERSE_MAX,
    AQI_UNIVERSE_MIN,
    AQI_UNIVERSE_POINTS,
    OUTPUT_FUZZY_LABELS,
)


MembershipSpec = dict[str, object]


def trimf(x: float, abc: tuple[float, float, float]) -> float:
    a, b, c = abc
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b != a else 0.0
    return (c - x) / (c - b) if c != b else 0.0


def trapmf(x: float, abcd: tuple[float, float, float, float]) -> float:
    a, b, c, d = abcd
    if x <= a or x >= d:
        if (a == b and x <= b) or (c == d and x >= c):
            return 1.0
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a) if b != a else 1.0
    return (d - x) / (d - c) if d != c else 1.0


def evaluate_membership(x: float, spec: MembershipSpec) -> float:
    mf_type = str(spec["type"])
    params = tuple(spec["params"])
    if mf_type == "trimf":
        return trimf(x, params)  # type: ignore[arg-type]
    if mf_type == "trapmf":
        return trapmf(x, params)  # type: ignore[arg-type]
    raise ValueError(f"Unsupported membership type: {mf_type}")


def build_input_membership_definitions() -> dict[str, dict[str, MembershipSpec]]:
    return {
        "PM2.5": {
            "Good": {"type": "trapmf", "params": (0.0, 0.0, 30.0, 60.0)},
            "Satisfactory": {"type": "trimf", "params": (30.0, 60.0, 90.0)},
            "Moderate": {"type": "trimf", "params": (60.0, 90.0, 120.0)},
            "Poor": {"type": "trimf", "params": (90.0, 120.0, 250.0)},
            "Very Poor": {"type": "trimf", "params": (120.0, 250.0, 350.0)},
            "Severe": {"type": "trapmf", "params": (250.0, 350.0, 500.0, 500.0)},
        },
        "PM10": {
            "Good": {"type": "trapmf", "params": (0.0, 0.0, 50.0, 100.0)},
            "Satisfactory": {"type": "trimf", "params": (50.0, 100.0, 175.0)},
            "Moderate": {"type": "trimf", "params": (100.0, 175.0, 250.0)},
            "Poor": {"type": "trimf", "params": (175.0, 250.0, 350.0)},
            "Very Poor": {"type": "trimf", "params": (250.0, 350.0, 430.0)},
            "Severe": {"type": "trapmf", "params": (350.0, 430.0, 600.0, 600.0)},
        },
        "NO2": {
            "Good": {"type": "trapmf", "params": (0.0, 0.0, 40.0, 80.0)},
            "Satisfactory": {"type": "trimf", "params": (40.0, 80.0, 130.0)},
            "Moderate": {"type": "trimf", "params": (80.0, 130.0, 180.0)},
            "Poor": {"type": "trimf", "params": (130.0, 180.0, 280.0)},
            "Very Poor": {"type": "trimf", "params": (180.0, 280.0, 400.0)},
            "Severe": {"type": "trapmf", "params": (280.0, 400.0, 400.0, 400.0)},
        },
        "CO": {
            "Good": {"type": "trapmf", "params": (0.0, 0.0, 1.0, 2.0)},
            "Satisfactory": {"type": "trimf", "params": (1.0, 2.0, 6.0)},
            "Moderate": {"type": "trimf", "params": (2.0, 6.0, 10.0)},
            "Poor": {"type": "trimf", "params": (6.0, 10.0, 17.0)},
            "Very Poor": {"type": "trimf", "params": (10.0, 17.0, 34.0)},
            "Severe": {"type": "trapmf", "params": (17.0, 34.0, 50.0, 50.0)},
        },
        "O3": {
            "Good": {"type": "trapmf", "params": (0.0, 0.0, 50.0, 100.0)},
            "Satisfactory": {"type": "trimf", "params": (50.0, 100.0, 134.0)},
            "Moderate": {"type": "trimf", "params": (100.0, 134.0, 168.0)},
            "Poor": {"type": "trimf", "params": (134.0, 168.0, 208.0)},
            "Very Poor": {"type": "trimf", "params": (168.0, 208.0, 300.0)},
            "Severe": {"type": "trapmf", "params": (208.0, 300.0, 400.0, 400.0)},
        },
        "SO2": {
            "Good": {"type": "trapmf", "params": (0.0, 0.0, 40.0, 80.0)},
            "Satisfactory": {"type": "trimf", "params": (40.0, 80.0, 230.0)},
            "Moderate": {"type": "trimf", "params": (80.0, 230.0, 380.0)},
            "Poor": {"type": "trimf", "params": (230.0, 380.0, 800.0)},
            "Severe": {"type": "trapmf", "params": (380.0, 800.0, 800.0, 800.0)},
        },
    }


def build_aqi_output_params() -> dict[str, MembershipSpec]:
    return {
        "Good": {"type": "trapmf", "params": (0.0, 0.0, 50.0, 100.0)},
        "Satisfactory": {"type": "trimf", "params": (50.0, 100.0, 200.0)},
        "Moderate": {"type": "trimf", "params": (100.0, 200.0, 300.0)},
        "Poor": {"type": "trimf", "params": (200.0, 300.0, 400.0)},
        "Very Poor": {"type": "trimf", "params": (300.0, 400.0, 500.0)},
        "Severe": {"type": "trapmf", "params": (400.0, 500.0, 500.0, 500.0)},
    }


def build_membership_definitions(
    train_df: pd.DataFrame,
) -> dict[str, dict[str, MembershipSpec]]:
    del train_df
    membership_defs = build_input_membership_definitions()
    membership_defs["AQI"] = build_aqi_output_params()
    return membership_defs


def fuzzify_value(value: float, params: dict[str, MembershipSpec]) -> dict[str, float]:
    return {
        label: evaluate_membership(value, params[label])
        for label in params
    }


def best_label(value: float, params: dict[str, MembershipSpec]) -> tuple[str, float]:
    memberships = fuzzify_value(value, params)
    label, degree = max(memberships.items(), key=lambda item: item[1])
    return label, degree


def bucket_from_aqi(aqi: float) -> str:
    if aqi <= 50:
        return OUTPUT_FUZZY_LABELS[0]
    if aqi <= 100:
        return OUTPUT_FUZZY_LABELS[1]
    if aqi <= 200:
        return OUTPUT_FUZZY_LABELS[2]
    if aqi <= 300:
        return OUTPUT_FUZZY_LABELS[3]
    if aqi <= 400:
        return OUTPUT_FUZZY_LABELS[4]
    return OUTPUT_FUZZY_LABELS[5]


def bucket_distance(actual_bucket: str, predicted_bucket: str) -> int:
    bucket_index = {label: idx for idx, label in enumerate(OUTPUT_FUZZY_LABELS)}
    return abs(bucket_index[actual_bucket] - bucket_index[predicted_bucket])


def output_universe() -> np.ndarray:
    return np.linspace(AQI_UNIVERSE_MIN, AQI_UNIVERSE_MAX, AQI_UNIVERSE_POINTS)
