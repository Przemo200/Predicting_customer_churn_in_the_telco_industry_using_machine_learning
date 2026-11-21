"""
Shared data-cleaning rules used by training and scoring
"""

from __future__ import annotations

import pandas as pd

from config import POSITIVE_LABEL, TARGET, TOTAL_CHARGES

INTERNET_DEPENDENT = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]


def clean_feature_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = ["tenure", TOTAL_CHARGES, "SeniorCitizen"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    raw_tc = df[TOTAL_CHARGES]
    blank = raw_tc.isna() | raw_tc.astype(str).str.strip().eq("")
    tc = pd.to_numeric(raw_tc.where(~blank), errors="coerce")

    non_numeric = tc.isna() & ~blank
    if non_numeric.any():
        bad_values = sorted(raw_tc[non_numeric].astype(str).unique().tolist())
        raise ValueError(f"Non-numeric TotalCharges values: {bad_values}")

    if blank.any() and not df.loc[blank, "tenure"].eq(0).all():
        raise ValueError("Missing TotalCharges values occur for customers with tenure > 0")

    df[TOTAL_CHARGES] = tc.fillna(0.0)

    senior_mapping = {
        0: "No",
        1: "Yes",
        "0": "No",
        "1": "Yes",
        "No": "No",
        "Yes": "Yes"
    }
    senior = df["SeniorCitizen"].map(senior_mapping)

    if senior.isna().any():
        bad_values = sorted(df.loc[senior.isna(), "SeniorCitizen"].astype(str).unique().tolist())
        raise ValueError(f"Unexpected SeniorCitizen values: {bad_values}")

    df["SeniorCitizen"] = senior

    existing_internet_cols = [c for c in INTERNET_DEPENDENT if c in df.columns]
    if existing_internet_cols:
        df[existing_internet_cols] = df[existing_internet_cols].replace("No internet service", "No")

    if "MultipleLines" in df.columns:
        df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

    return df


def encode_target(series: pd.Series) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"Missing {TARGET} values: {int(series.isna().sum())}")

    if pd.api.types.is_numeric_dtype(series):
        unexpected = sorted(set(series.unique().tolist()) - {0, 1})
        if unexpected:
            raise ValueError(f"Unexpected numeric {TARGET} values: {unexpected}")
        return series.astype(int)

    values = series.astype(str)
    allowed = {"Yes", "No"}
    unexpected = sorted(set(values.unique()) - allowed)

    if unexpected:
        raise ValueError(f"Unexpected {TARGET} values: {unexpected}")

    return values.eq(POSITIVE_LABEL).astype(int)