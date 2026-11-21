"""
Shared preprocessing utilities

Standard:
- numerical variables -> StandardScaler
- categorical variables -> OneHotEncoder

SMOTENC:
- numerical variables are standardized before SMOTENC
- categorical variables remain unchanged during SMOTENC
- categorical variables are one-hot encoded after SMOTENC
"""

from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_lists(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    return numeric_features, categorical_features


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), categorical_features)
        ],
        verbose_feature_names_out=False
    )


def build_smotenc_scaler(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", "passthrough", categorical_features)
        ],
        verbose_feature_names_out=False
    )

    return transformer.set_output(transform="pandas")


def build_smotenc(categorical_features: list[str], random_state: int) -> SMOTENC:
    return SMOTENC(categorical_features=categorical_features, random_state=random_state)


def build_smotenc_postprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), categorical_features)
        ],
        verbose_feature_names_out=False
    )
