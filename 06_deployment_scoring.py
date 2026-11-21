"""
Phase 6 CRISP-DM - batch scoring with input validation and drift monitoring of inputs and scores
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import (
    ID_COLUMN,
    MODELS_DIR,
    PROCESSED_DIR,
    PSI_ALERT,
    PSI_BINS,
    PSI_MIN_ROWS,
    PSI_WARNING,
    RAW_DATA_PATH,
    RESULTS_DIR,
    RETENTION_SCENARIOS,
    TARGET,
)
from src.data import INTERNET_DEPENDENT, clean_feature_values
from src.evaluation import assign_risk_group, benefit_kwargs, expected_contact_benefit
from src.manifest import library_versions, load_manifest, require_stages


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input must be .csv or .xlsx")


def prepare_scoring_features(df: pd.DataFrame, input_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    if ID_COLUMN in df.columns:
        missing_ids = df[ID_COLUMN].isna() | df[ID_COLUMN].astype(str).str.strip().eq("")
        if missing_ids.any():
            raise ValueError(f"Missing {ID_COLUMN} values in {int(missing_ids.sum())} row(s)")
        customer_ids = df[ID_COLUMN].astype(str)
        duplicated = customer_ids.duplicated(keep=False)
        if duplicated.any():
            raise ValueError(f"Duplicated {ID_COLUMN} values: {int(duplicated.sum())} rows share {int(customer_ids[duplicated].nunique())} ID(s); remove duplicates before scoring")
    else:
        customer_ids = pd.Series([f"row_{i}" for i in range(len(df))], name=ID_COLUMN)

    df = df.drop(columns=[c for c in (TARGET, ID_COLUMN) if c in df.columns])
    df = clean_feature_values(df)

    missing = [c for c in input_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing model input columns: {missing}")

    return df[input_columns].copy(), customer_ids


def validate_input(X: pd.DataFrame, reference: pd.DataFrame, allow_unseen: bool) -> tuple[pd.DataFrame, pd.Series]:
    X = X.copy()
    numeric = reference.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in reference.columns if c not in numeric]

    for column in numeric:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    missing = X.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        raise ValueError(f"Missing or non-numeric values in model inputs: {missing.to_dict()}")

    infinite = [c for c in numeric if not np.isfinite(X[c].to_numpy(dtype=float)).all()]
    if infinite:
        raise ValueError(f"Infinite values in numeric columns: {infinite}")

    negative = [c for c in numeric if (X[c] < 0).any()]
    if negative:
        raise ValueError(f"Negative values in numeric columns: {negative}")

    if "tenure" in X.columns and not X["tenure"].eq(X["tenure"].round()).all():
        raise ValueError("tenure must be a whole number of months")

    inconsistent = inconsistent_services(X)
    if inconsistent:
        raise ValueError(f"Inconsistent service columns: {inconsistent}")

    unseen = {c: sorted(set(X[c].unique()) - set(reference[c].unique())) for c in categorical}
    unseen = {c: values for c, values in unseen.items() if values}
    if unseen and not allow_unseen:
        raise ValueError(f"Categories not seen in training (use --allow-unseen to score anyway): {unseen}")

    flags = pd.Series("", index=X.index)

    for column in numeric:
        outside = (X[column] < reference[column].min()) | (X[column] > reference[column].max())
        flags = flags.where(~outside, flags + f"{column}_outside_training_range;")

    for column, values in unseen.items():
        flags = flags.where(~X[column].isin(values), flags + f"{column}_unseen_category;")

    flags = flags.str.rstrip(";").replace("", "ok")

    return X, flags


def inconsistent_services(X: pd.DataFrame) -> dict[str, int]:
    problems = {}

    if "InternetService" in X.columns:
        no_internet = X["InternetService"].eq("No")
        for column in INTERNET_DEPENDENT:
            if column in X.columns:
                count = int((no_internet & X[column].ne("No")).sum())
                if count:
                    problems[f"{column}_without_internet"] = count

    if {"PhoneService", "MultipleLines"} <= set(X.columns):
        count = int((X["PhoneService"].eq("No") & X["MultipleLines"].ne("No")).sum())
        if count:
            problems["MultipleLines_without_phone"] = count

    return problems


def warn_on_version_mismatch(expected: dict) -> None:
    current = library_versions()
    for name, expected_version in expected.items():
        if current.get(name) != expected_version:
            print(f"Warning: {name} is {current.get(name)}, model was trained with {expected_version}")


def distribution_shift(reference: np.ndarray, current: np.ndarray, categorical: bool) -> float:
    if categorical or np.unique(reference).size <= PSI_BINS:
        categories = sorted(set(reference.tolist()) | set(current.tolist()))
        ref = np.array([(reference == c).mean() for c in categories])
        cur = np.array([(current == c).mean() for c in categories])
    else:
        cuts = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, PSI_BINS + 1)[1:-1]))
        n_bins = len(cuts) + 1
        ref = np.bincount(np.digitize(reference, cuts, right=True), minlength=n_bins) / len(reference)
        cur = np.bincount(np.digitize(current, cuts, right=True), minlength=n_bins) / len(current)

    ref = np.clip(ref, 1e-4, None)
    cur = np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def monitoring_report(X_reference: pd.DataFrame, X_new: pd.DataFrame, score_reference: np.ndarray | None = None, scores: np.ndarray | None = None) -> pd.DataFrame:
    rows = []
    numeric = set(X_reference.select_dtypes(include="number").columns)
    large_enough = len(X_new) >= PSI_MIN_ROWS

    if score_reference is not None and scores is not None:
        psi = distribution_shift(np.asarray(score_reference, dtype=float), np.asarray(scores, dtype=float), False) if large_enough else np.nan
        status = "skipped_small_batch" if not large_enough else "alert" if psi >= PSI_ALERT else "warning" if psi >= PSI_WARNING else "ok"
        rows.append({"feature": "churn_score", "type": "model_score", "psi": psi, "unseen_categories": "", "status": status})

    for feature in X_reference.columns:
        is_categorical = feature not in numeric
        psi = distribution_shift(X_reference[feature].to_numpy(), X_new[feature].to_numpy(), is_categorical) if large_enough else np.nan
        unseen = sorted(set(X_new[feature].unique()) - set(X_reference[feature].unique())) if is_categorical else []

        if not large_enough:
            status = "skipped_small_batch"
        else:
            status = "alert" if psi >= PSI_ALERT else "warning" if psi >= PSI_WARNING else "ok"

        if unseen:
            status = "alert"

        rows.append({"feature": feature, "type": "categorical" if is_categorical else "numeric", "psi": psi, "unseen_categories": ";".join(map(str, unseen)), "status": status})

    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch churn scoring")
    parser.add_argument("input_file", type=Path, nargs="?", default=RAW_DATA_PATH, help="CSV or XLSX file with customer features; defaults to the project raw dataset for a scoring demonstration")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "scored_customers.csv", help="Output CSV path; the monitoring report is saved next to it")
    parser.add_argument("--allow-unseen", action="store_true", help="Score rows with categories not seen in training, they are flagged in the output")
    args = parser.parse_args()

    upstream = require_stages(["03_data_preparation", "05_evaluation"])
    print(f"Scoring with artefacts of runs {upstream}")

    pipeline = joblib.load(MODELS_DIR / "final_churn_pipeline.joblib")
    calibrator = joblib.load(MODELS_DIR / "final_calibrator.joblib")
    metadata = json.loads((RESULTS_DIR / "final_model_metadata.json").read_text(encoding="utf-8"))

    threshold = float(metadata["selected_threshold"])
    low_cut = float(metadata["risk_group_cutoffs"]["low_to_medium"])
    high_cut = float(metadata["risk_group_cutoffs"]["medium_to_high"])
    input_columns = list(metadata["input_columns"])
    warn_on_version_mismatch(load_manifest()["stages"]["05_evaluation"]["library_versions"])

    X_reference = pd.read_csv(PROCESSED_DIR / "X_train.csv")[input_columns]

    raw = read_table(args.input_file)
    X, customer_ids = prepare_scoring_features(raw, input_columns)
    X, input_flag = validate_input(X, X_reference, args.allow_unseen)

    scores = pipeline.predict_proba(X)[:, 1]
    probabilities = calibrator.predict(scores)
    predictions = (scores >= threshold).astype(int)
    base = RETENTION_SCENARIOS["base"]
    expected_benefit = expected_contact_benefit(probabilities, X["MonthlyCharges"].to_numpy(dtype=float), **benefit_kwargs(base))

    output = pd.DataFrame({
        ID_COLUMN: customer_ids.to_numpy(),
        "churn_score": scores,
        "churn_probability_calibrated": probabilities,
        "predicted_churn": np.where(predictions == 1, "Yes", "No"),
        "risk_group": assign_risk_group(scores, low_cut, high_cut),
        "expected_contact_benefit_base": expected_benefit,
        "contact_recommended_base": np.where(expected_benefit > 0, "Yes", "No"),
        "input_flag": input_flag.to_numpy()
    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    oof_reference = pd.read_csv(RESULTS_DIR / "training_oof_predictions.csv")["oof_score"].to_numpy(dtype=float)
    monitoring = monitoring_report(X_reference, X, oof_reference, scores)
    monitoring_path = args.output.with_name(f"{args.output.stem}_monitoring.csv")
    monitoring.to_csv(monitoring_path, index=False)

    risk_share = output["risk_group"].value_counts(normalize=True).reindex(["Low", "Medium", "High"]).fillna(0.0)
    print("Risk group shares - Medium and High are the predicted churners -")
    print(risk_share.round(3).to_dict())

    flagged_rows = int((output["input_flag"] != "ok").sum())
    print(f"Rows with input flags: {flagged_rows}")
    print(f"predicted_churn (raw score >= F1 threshold from OOF, a classification): {int(predictions.sum())} of {len(output)}")
    print(f"contact_recommended_base (expected simulated benefit > 0 in the base scenario, an illustrative economic rule): {int((expected_benefit > 0).sum())} of {len(output)}")
    print("The two columns answer different questions and need not agree")

    flagged = monitoring.loc[~monitoring["status"].isin(["ok", "skipped_small_batch"])]
    if len(X) < PSI_MIN_ROWS:
        print(f"Drift check skipped - batch has {len(X)} rows, PSI needs at least {PSI_MIN_ROWS}")
    else:
        print(f"Drift check (inputs and raw score vs training OOF scores) - {len(flagged)} item(s) with warning/alert (PSI warning >= {PSI_WARNING}, alert >= {PSI_ALERT})")
        print("PSI measures a change of distribution, not a loss of predictive performance")
    if not flagged.empty:
        print(flagged.to_string(index=False))

    print("\nBatch scoring completed")
    print(f"Input rows: {len(output)}")
    print(f"Decision threshold: {threshold:.4f}")
    print(f"Saved: {args.output}")
    print(f"Saved: {monitoring_path}")
    if args.input_file == RAW_DATA_PATH:
        print("Default raw-dataset scoring is a deployment demonstration only and is not used for model evaluation")
        print("About 80% of these rows are training customers scored in-sample, so the score PSI against OOF scores is not a real drift signal here")
    print()


if __name__ == "__main__":
    main()