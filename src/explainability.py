"""
Helpers for model interpretation on fitted pipelines
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def transform_for_model(pipeline, X: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    if "scale_for_smotenc" in pipeline.named_steps:
        X = pipeline.named_steps["scale_for_smotenc"].transform(X)

    preprocessor = pipeline.named_steps["preprocess"]
    transformed = preprocessor.transform(X)
    transformed_df = pd.DataFrame(np.asarray(transformed), columns=preprocessor.get_feature_names_out().tolist(), index=X.index)

    return transformed_df, pipeline.named_steps["model"]


def original_feature_name(encoded_name: str, original_features: list[str]) -> str:
    if encoded_name in original_features:
        return encoded_name
    matches = [feature for feature in original_features if encoded_name.startswith(f"{feature}_")]
    if not matches:
        return encoded_name

    return max(matches, key=len)


def pretty_feature_name(encoded_name: str, original_features: list[str]) -> str:
    if encoded_name == "tenure_years":
        return "tenure [lata]"
    original = original_feature_name(encoded_name, original_features)
    if original == encoded_name:
        return encoded_name
    return f"{original}: {encoded_name[len(original) + 1:]}"


def aggregate_shap_by_original(feature_names: list[str], shap_values: np.ndarray, original_features: list[str]) -> pd.DataFrame:
    mapping = [original_feature_name(name, original_features) for name in feature_names]
    values = pd.DataFrame(np.asarray(shap_values, dtype=float), columns=mapping)
    per_sample = values.T.groupby(level=0).sum().T
    result = pd.DataFrame({"original_feature": per_sample.columns, "mean_abs_shap": per_sample.abs().mean(axis=0).to_numpy()})
    return result.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def aggregate_local_shap(shap_row: np.ndarray, encoded_names: list[str], original_row: pd.Series) -> tuple[np.ndarray, np.ndarray, list[str]]:
    mapping = [original_feature_name(name, original_row.index.tolist()) for name in encoded_names]
    summed = pd.Series(np.asarray(shap_row, dtype=float), index=mapping).groupby(level=0, sort=False).sum()
    data = np.array([original_row[feature] for feature in summed.index], dtype=object)
    return summed.to_numpy(), data, summed.index.tolist()


def display_frame(X_transformed: pd.DataFrame, X_original: pd.DataFrame) -> pd.DataFrame:
    shown = X_transformed.copy()
    for column in X_original.select_dtypes(include="number").columns:
        if column in shown.columns:
            shown[column] = X_original.loc[shown.index, column].to_numpy()
    return shown


def variance_inflation_factors(design: pd.DataFrame) -> pd.DataFrame:
    values = design.to_numpy(dtype=float)
    rows = []

    for j, column in enumerate(design.columns):
        target = values[:, j]
        others = np.column_stack([np.ones(len(values)), np.delete(values, j, axis=1)])
        coef = np.linalg.lstsq(others, target, rcond=None)[0]
        residual = target - others @ coef
        r_squared = 1.0 - residual.var() / target.var()
        rows.append({"feature": column, "vif": float(np.inf) if r_squared >= 1.0 - 1e-12 else float(1.0 / (1.0 - r_squared))})

    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)