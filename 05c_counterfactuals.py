"""
Counterfactual explanations for churn predictions
"""

from __future__ import annotations

import itertools
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from config import (
    COUNTERFACTUAL_MARGIN,
    COUNTERFACTUAL_MAX_CHANGES,
    COUNTERFACTUAL_TOP_SOLUTIONS,
    ID_COLUMN,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
)
from src.data import INTERNET_DEPENDENT
from src.manifest import record_stage, require_stages

CONTRACT_ORDER = ["Month-to-month", "One year", "Two year"]
AUTOMATIC_PAYMENTS = ["Bank transfer (automatic)", "Credit card (automatic)"]
ADD_ON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "MultipleLines"
]

PRICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

VARIANTS = {
    "all_levers": {"excluded": (), "reprice": True},
    "without_contract": {"excluded": ("Contract",), "reprice": True},
    "without_contract_free_services": {"excluded": ("Contract",), "reprice": False}
}


def load_inputs():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    predictions = pd.read_csv(RESULTS_DIR / "test_predictions.csv")
    metadata = json.loads((RESULTS_DIR / "final_model_metadata.json").read_text(encoding="utf-8"))
    pipeline = joblib.load(MODELS_DIR / "final_churn_pipeline.joblib")
    return X_train, X_test, predictions, metadata, pipeline


def build_price_model():
    return make_pipeline(OneHotEncoder(handle_unknown="ignore"), LinearRegression())


def fit_price_model(X_train: pd.DataFrame):
    model = build_price_model()
    model.fit(X_train[PRICE_COLUMNS], X_train["MonthlyCharges"])
    r_squared = float(model.score(X_train[PRICE_COLUMNS], X_train["MonthlyCharges"]))
    return model, r_squared


def price_model_cv_diagnostics(X_train: pd.DataFrame, folds: int = 5) -> dict:
    cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    result = cross_validate(build_price_model(), X_train[PRICE_COLUMNS], X_train["MonthlyCharges"], cv=cv, scoring={"mae": "neg_mean_absolute_error", "r2": "r2"})
    mae = -result["test_mae"]
    return {
        "folds": folds,
        "mae_usd_mean": float(mae.mean()),
        "mae_usd_std": float(mae.std(ddof=1)),
        "r2_mean": float(result["test_r2"].mean()),
        "r2_std": float(result["test_r2"].std(ddof=1))
    }


def updated_monthly_charges(price_model, row: pd.Series, candidates: pd.DataFrame, minimum_price: float) -> np.ndarray:
    base = float(price_model.predict(pd.DataFrame([row[PRICE_COLUMNS]]))[0])
    new = price_model.predict(candidates[PRICE_COLUMNS])
    return np.maximum(float(row["MonthlyCharges"]) + new - base, minimum_price)


def atomic_changes(row: pd.Series, excluded: tuple[str, ...] = ()) -> list[dict]:
    changes = []

    if "Contract" not in excluded:
        current = CONTRACT_ORDER.index(row["Contract"])
        for steps, value in enumerate(CONTRACT_ORDER[current + 1:], start=1):
            changes.append({"feature": "Contract", "old": row["Contract"], "new": value, "cost": steps})

    if "PaymentMethod" not in excluded and row["PaymentMethod"] not in AUTOMATIC_PAYMENTS:
        for value in AUTOMATIC_PAYMENTS:
            changes.append({"feature": "PaymentMethod", "old": row["PaymentMethod"], "new": value, "cost": 1})

    for feature in ADD_ON_SERVICES:
        if feature in excluded or row[feature] != "No":
            continue
        if feature in INTERNET_DEPENDENT and row["InternetService"] == "No":
            continue
        if feature == "MultipleLines" and row["PhoneService"] == "No":
            continue
        changes.append({"feature": feature, "old": "No", "new": "Yes", "cost": 1})

    return changes


def valid_combination(combo: tuple[dict, ...]) -> bool:
    features = [change["feature"] for change in combo]
    return len(features) == len(set(features))


def build_candidate(row: pd.Series, combo: tuple[dict, ...]) -> pd.Series:
    candidate = row.copy()
    for change in combo:
        candidate[change["feature"]] = change["new"]
    return candidate


def search_counterfactuals(pipeline, price_model, X_train: pd.DataFrame, row: pd.Series, threshold: float, max_changes: int, margin: float, top_solutions: int, excluded: tuple[str, ...] = (), reprice: bool = True) -> list[dict]:
    target = threshold - margin
    if target <= 0:
        raise ValueError(f"threshold - margin must be positive, got {target:.4f}")

    atoms = atomic_changes(row, excluded)
    minimum_price = float(X_train["MonthlyCharges"].min())

    solutions = []
    valid_sets_lower_levels = []
    used_feature_sets = set()

    for n_changes in range(1, max_changes + 1):
        combos = [combo for combo in itertools.combinations(atoms, n_changes) if valid_combination(combo)]

        if not combos:
            continue

        candidates = pd.DataFrame([build_candidate(row, combo) for combo in combos])[row.index]
        if reprice:
            candidates["MonthlyCharges"] = updated_monthly_charges(price_model, row, candidates, minimum_price)
        scores = pipeline.predict_proba(candidates)[:, 1]

        level = []
        for idx in np.flatnonzero(scores < target):
            combo = combos[int(idx)]
            change_set = frozenset((change["feature"], change["new"]) for change in combo)
            if any(previous < change_set for previous in valid_sets_lower_levels):
                continue

            cost = sum(change["cost"] for change in combo)
            level.append((cost, float(scores[idx]), int(idx), combo, change_set))

        level.sort(key=lambda item: (item[0], item[1]))
        valid_sets_lower_levels.extend(item[4] for item in level)

        for cost, score_after, idx, combo, _ in level:
            feature_set = frozenset(change["feature"] for change in combo)
            if feature_set in used_feature_sets:
                continue

            used_feature_sets.add(feature_set)
            solutions.append({
                "score_after": score_after,
                "n_changes": n_changes,
                "change_cost": cost,
                "changes": list(combo),
                "candidate": candidates.iloc[idx]
            })

            if len(solutions) >= top_solutions:
                return solutions

    return solutions


def numeric_ranges(X_train: pd.DataFrame) -> dict[str, float]:
    ranges = {}
    for feature in X_train.select_dtypes(include="number").columns:
        value_range = float(X_train[feature].max() - X_train[feature].min())
        ranges[feature] = value_range if value_range > 0 else 1.0
    return ranges


def gower_to_nearest_training(X_train: pd.DataFrame, point: pd.Series, ranges: dict[str, float]) -> float:
    total = np.zeros(len(X_train), dtype=float)

    for feature in X_train.columns:
        if feature in ranges:
            total += np.abs(X_train[feature].to_numpy(dtype=float) - float(point[feature])) / ranges[feature]
        else:
            total += (X_train[feature].to_numpy() != point[feature]).astype(float)

    return float((total / X_train.shape[1]).min())


def changes_to_text(changes: list[dict]) -> str:
    return "; ".join(f"{change['feature']}: {change['old']} -> {change['new']}" for change in changes)


def intervention_frequency(output: pd.DataFrame) -> pd.DataFrame:
    found = output.loc[output["counterfactual_found"]]
    exploded = found.assign(feature=found["changed_features"].str.split(";")).explode("feature")
    solutions = exploded.groupby(["variant", "feature"]).size().rename("n_solutions")
    customers = exploded.loc[exploded["rank"].eq(1)].groupby(["variant", "feature"])[ID_COLUMN].nunique().rename("n_customers_rank1")

    frequency = pd.concat([solutions, customers], axis=1).fillna(0).astype(int).reset_index()
    analysed = output.groupby("variant")[ID_COLUMN].nunique()
    with_cf = found.groupby("variant")[ID_COLUMN].nunique()
    frequency["share_of_analysed_customers"] = frequency["n_customers_rank1"] / frequency["variant"].map(analysed)
    frequency["share_of_customers_with_counterfactual"] = frequency["n_customers_rank1"] / frequency["variant"].map(with_cf)

    return frequency.sort_values(["variant", "n_customers_rank1", "n_solutions"], ascending=[True, False, False]).reset_index(drop=True)


CUSTOMER_GROUPS = (("all", None), ("TP", 1), ("FP", 0))


def summary_statistics(group: pd.DataFrame) -> dict:
    customers = int(group[ID_COLUMN].nunique())
    with_cf = int(group.loc[group["counterfactual_found"], ID_COLUMN].nunique())
    best = group.loc[group["rank"].eq(1)]
    per_customer = group.drop_duplicates(ID_COLUMN)

    return {
        "customers": customers,
        "customers_with_counterfactual": with_cf,
        "share_with_counterfactual": with_cf / customers if customers else np.nan,
        "median_score_before_all": float(per_customer["score_before"].median()) if customers else np.nan,
        "median_n_changes_rank1": float(best["n_changes"].median()) if len(best) else np.nan,
        "share_single_change_rank1": float(best["n_changes"].eq(1).mean()) if len(best) else np.nan,
        "median_score_before_rank1": float(best["score_before"].median()) if len(best) else np.nan,
        "median_score_after_rank1": float(best["score_after"].median()) if len(best) else np.nan,
        "median_score_reduction_rank1": float((best["score_before"] - best["score_after"]).median()) if len(best) else np.nan,
        "median_plausibility_before_rank1": float(best["plausibility_before"].median()) if len(best) else np.nan,
        "median_plausibility_after_rank1": float(best["plausibility_after"].median()) if len(best) else np.nan,
        "median_monthly_charges_change_rank1": float((best["monthly_charges_after"] - best["monthly_charges_before"]).median()) if len(best) else np.nan
    }


def summarize(output: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for variant, group in output.groupby("variant", sort=True):
        for label, truth in CUSTOMER_GROUPS:
            subset = group if truth is None else group.loc[group["y_true"].eq(truth)]
            rows.append({"variant": variant, "customer_group": label, **summary_statistics(subset)})

    return pd.DataFrame(rows)


def main() -> None:
    upstream = require_stages(["03_data_preparation", "05_evaluation"])
    X_train, X_test, predictions, metadata, pipeline = load_inputs()
    threshold = float(metadata["selected_threshold"])
    ranges = numeric_ranges(X_train)
    price_model, price_r_squared = fit_price_model(X_train)
    price_cv = price_model_cv_diagnostics(X_train)

    examples = predictions.loc[predictions["prediction_selected"].eq(1)]
    if examples.empty:
        raise ValueError("No positive predictions available for counterfactual analysis")
    unknown_contracts = set(X_test["Contract"]) - set(CONTRACT_ORDER)
    if unknown_contracts:
        raise ValueError(f"Unexpected Contract values: {unknown_contracts}")

    print("\nCounterfactual analysis\n")
    print(f"Decision threshold: {threshold:.4f}, margin: {COUNTERFACTUAL_MARGIN}, max changes: {COUNTERFACTUAL_MAX_CHANGES}")
    print(f"Price model R2 on training data: {price_r_squared:.4f}")
    print(f"Price model 5-fold CV: MAE {price_cv['mae_usd_mean']:.2f} USD, R2 {price_cv['r2_mean']:.4f}")
    print(f"Customers predicted as churn (analysed): {len(examples)}")

    output_rows = []

    for variant, settings in VARIANTS.items():
        print(f"\nVariant: {variant}")

        for done, (_, example) in enumerate(examples.iterrows(), start=1):
            row_index = int(example["row_index"])
            row = X_test.iloc[row_index].copy()

            solutions = search_counterfactuals(
                pipeline=pipeline,
                price_model=price_model,
                X_train=X_train,
                row=row,
                threshold=threshold,
                max_changes=COUNTERFACTUAL_MAX_CHANGES,
                margin=COUNTERFACTUAL_MARGIN,
                top_solutions=COUNTERFACTUAL_TOP_SOLUTIONS,
                excluded=settings["excluded"],
                reprice=settings["reprice"]
            )

            base = {
                "variant": variant,
                "row_index": row_index,
                ID_COLUMN: example[ID_COLUMN],
                "y_true": int(example["y_true"]),
                "score_before": float(example["churn_score"]),
                "threshold": threshold,
                "margin": COUNTERFACTUAL_MARGIN,
                "monthly_charges_before": float(row["MonthlyCharges"]),
                "plausibility_before": gower_to_nearest_training(X_train, row, ranges)
            }

            if not solutions:
                output_rows.append({**base, "rank": np.nan, "counterfactual_found": False, "score_after": np.nan, "n_changes": np.nan, "change_cost": np.nan, "monthly_charges_after": np.nan, "plausibility_after": np.nan, "changed_features": "", "changes": ""})
            else:
                for rank, solution in enumerate(solutions, start=1):
                    output_rows.append({
                        **base,
                        "rank": rank,
                        "counterfactual_found": True,
                        "score_after": solution["score_after"],
                        "n_changes": solution["n_changes"],
                        "change_cost": solution["change_cost"],
                        "monthly_charges_after": float(solution["candidate"]["MonthlyCharges"]),
                        "plausibility_after": gower_to_nearest_training(X_train, solution["candidate"], ranges),
                        "changed_features": ";".join(change["feature"] for change in solution["changes"]),
                        "changes": changes_to_text(solution["changes"])
                    })

            if done % 100 == 0 or done == len(examples):
                print(f"{done}/{len(examples)} customers processed")

    output = pd.DataFrame(output_rows)
    output.to_csv(RESULTS_DIR / "counterfactual_examples.csv", index=False)

    found = output.loc[output["counterfactual_found"]].copy()
    if not found.empty:
        frequency = intervention_frequency(output)
        frequency.to_csv(RESULTS_DIR / "counterfactual_feature_frequency.csv", index=False)

    summary = summarize(output)
    summary.to_csv(RESULTS_DIR / "counterfactual_summary.csv", index=False)
    print("\nSummary:\n")
    print(summary.round(3).to_string(index=False))

    cf_metadata = {
        "final_model": metadata["winner_combo_id"],
        "method": "level-wise exhaustive search (1, 2, ... max_changes interventions) over a discretised space of actionable retention actions, stopped after top_solutions solutions; minimal solutions only; one solution per set of changed features; ranked by number of interventions, then change cost, then score after the change",
        "validity_rule": "raw model churn score < threshold - margin (a change of the model prediction, not an avoided churn)",
        "minimality_rule": "a solution is rejected when a strict subset of its changes is already a valid counterfactual",
        "minimality_scope": "subset-minimal with respect to the customer's generated (feature, new value) actions, the validity rule and repricing used here, and at most max_changes actions; all candidates of a level are scored before the search may stop, so every reported solution is subset-minimal; it is not minimal over actions outside this space, over other values of the same feature (e.g. Contract -> Two year may be reported next to Contract -> One year combined with another action), in total cost, or in any real-world sense",
        "threshold": threshold,
        "margin": COUNTERFACTUAL_MARGIN,
        "max_changes": COUNTERFACTUAL_MAX_CHANGES,
        "top_solutions_per_customer": COUNTERFACTUAL_TOP_SOLUTIONS,
        "customers_analysed": len(examples),
        "variants": {name: {"excluded": list(v["excluded"]), "reprice": v["reprice"]} for name, v in VARIANTS.items()},
        "price_consistency": "reprice=True: MonthlyCharges is recomputed as current price + change in a linear price model fitted on the service columns; reprice=False: MonthlyCharges unchanged (services offered free); TotalCharges and tenure are history and stay fixed",
        "price_model_r_squared_train": price_r_squared,
        "price_model_cv": price_cv,
        "feature_frequency_columns": "n_solutions counts all found solutions (up to top_solutions per customer); n_customers_rank1 counts distinct customers whose best solution uses the feature; share_of_analysed_customers divides it by all analysed customers (predicted churners in the test set), share_of_customers_with_counterfactual by customers with at least one counterfactual",
        "plausibility_measure": "Gower distance (mean over features) to the nearest training customer, computed after the search for evaluation only; compare plausibility_after with plausibility_before",
        "n_changes_definition": "number of interventions; repriced MonthlyCharges is not counted as a separate change",
        "actionable_changes": {
            "Contract": "only to a longer contract (cost = number of steps)",
            "PaymentMethod": f"only to an automatic method: {AUTOMATIC_PAYMENTS}",
            "add_on_services": f"only No -> Yes for {ADD_ON_SERVICES}, if the customer has the required internet or phone service"
        },
        "immutable_or_not_searched": [
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
            "TotalCharges",
            "MonthlyCharges (derived from services)",
            "PhoneService",
            "InternetService",
            "PaperlessBilling (customer preference, not a retention action)"
        ],
        "summary_groups": "counterfactual_summary.csv has one row per variant and customer_group: all analysed customers, TP (predicted churners who churned) and FP (predicted churners who stayed); descriptive split on test labels, not used for any choice",
        "interpretation_note": "Counterfactuals describe how the model's prediction changes under hypothetical profile changes; they are not causal effects of business actions and do not show that churn would be avoided"
    }
    (RESULTS_DIR / "counterfactual_metadata.json").write_text(json.dumps(cf_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    inputs = [PROCESSED_DIR / "X_train.csv", PROCESSED_DIR / "X_test.csv", RESULTS_DIR / "test_predictions.csv", RESULTS_DIR / "final_model_metadata.json", MODELS_DIR / "final_churn_pipeline.joblib"]
    outputs = [RESULTS_DIR / name for name in ("counterfactual_examples.csv", "counterfactual_summary.csv", "counterfactual_metadata.json")]
    if not found.empty:
        outputs.append(RESULTS_DIR / "counterfactual_feature_frequency.csv")
    parameters = {"max_changes": COUNTERFACTUAL_MAX_CHANGES, "margin": COUNTERFACTUAL_MARGIN, "top_solutions": COUNTERFACTUAL_TOP_SOLUTIONS, "threshold": threshold, "variants": cf_metadata["variants"]}
    run_id = record_stage("05c_counterfactuals", inputs, outputs, parameters, upstream)
    print(f"\nCounterfactual outputs saved (run {run_id})")


if __name__ == "__main__":
    main()
