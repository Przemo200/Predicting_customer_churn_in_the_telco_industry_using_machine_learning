"""
Phase 4 CRISP-DM - modeling

4.1 Select modeling technique
4.2 Generate test design
4.3 Build model
4.4 Assess model

Hyperparameter tuning and model assessment are performed only on X_train/y_train using repeated stratified cross-validation

Models:
- Logistic Regression
- Random Forest
- XGBoost

Class-imbalance variants:
- no balancing
- class weights
- SMOTENC
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from scipy.stats import loguniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from xgboost import XGBClassifier

from config import (
    CV_FOLDS,
    CV_REPEATS,
    MODEL_RANKING_COLUMNS,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    SEARCH_ITER_BY_MODEL,
)
from src.manifest import record_stage, require_stages
from src.preprocessing import (
    build_preprocessor,
    build_smotenc,
    build_smotenc_postprocessor,
    build_smotenc_scaler,
    get_feature_lists,
)
from src.stats import corrected_resampled_ttest, holm_adjust_pvalues

MODEL_KEYS = ("lr", "rf", "xgb")
BALANCING_VARIANTS = ("none", "class_weight", "smotenc")

MODEL_LABELS = {
    "lr": "Logistic Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost"
}

BALANCING_LABELS = {
    "none": "No balancing",
    "class_weight": "Class weights",
    "smotenc": "SMOTENC"
}

SMOTENC_SPACE = {
    "smotenc__k_neighbors": [3, 5, 7, 9],
    "smotenc__sampling_strategy": [0.5, 0.75, 1.0]
}

SCORING = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "average_precision": "average_precision"
}


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").iloc[:, 0].astype(int)

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train have different numbers of rows")

    return X_train, y_train


def build_estimator(model_key: str, balancing: str, scale_pos_weight: float):
    if model_key == "lr":
        return LogisticRegression(
            class_weight="balanced" if balancing == "class_weight" else None,
            solver="lbfgs",
            max_iter=3000
        )

    if model_key == "rf":
        return RandomForestClassifier(
            class_weight="balanced" if balancing == "class_weight" else None,
            random_state=RANDOM_STATE,
            n_jobs=1
        )

    if model_key == "xgb":
        return XGBClassifier(
            scale_pos_weight=scale_pos_weight if balancing == "class_weight" else 1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0,
            enable_categorical=False
        )

    raise ValueError(f"Unknown model key: {model_key}")


def build_pipeline(model_key: str, balancing: str, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    numeric, categorical = get_feature_lists(X_train)
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    model = build_estimator(model_key, balancing, scale_pos_weight)

    if balancing == "smotenc":
        return Pipeline(
            steps=[
                ("scale_for_smotenc", build_smotenc_scaler(numeric, categorical)),
                ("smotenc", build_smotenc(categorical, RANDOM_STATE)),
                ("preprocess", build_smotenc_postprocessor(numeric, categorical)),
                ("model", model)
            ]
        )

    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric, categorical)),
            ("model", model)
        ]
    )


def parameter_space(model_key: str, balancing: str) -> dict:
    if model_key == "lr":
        params = {"model__C": loguniform(1e-4, 1e3)}

    elif model_key == "rf":
        params = {
            "model__n_estimators": [200, 300, 400, 500, 600],
            "model__criterion": ["gini", "entropy"],
            "model__max_depth": [None, 5, 8, 10, 12, 15, 20, 30],
            "model__min_samples_split": [2, 4, 6, 10, 15, 20],
            "model__min_samples_leaf": [1, 2, 3, 4, 6, 8, 12],
            "model__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0]
        }

    elif model_key == "xgb":
        params = {
            "model__n_estimators": [100, 150, 200, 300, 400, 500, 600, 800, 1000],
            "model__max_depth": [1, 2, 3, 4, 5, 6],
            "model__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20],
            "model__min_child_weight": [1, 2, 3, 5, 8, 10],
            "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "model__gamma": [0.0, 0.05, 0.1, 0.25, 0.5, 1.0],
            "model__reg_alpha": [0.0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0],
            "model__reg_lambda": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        }

    else:
        raise ValueError(f"Unknown model key: {model_key}")

    if balancing == "smotenc":
        params.update(SMOTENC_SPACE)

    return params


def run_search(model_key: str, balancing: str, X_train: pd.DataFrame, y_train: pd.Series, cv) -> tuple[RandomizedSearchCV, dict]:
    pipeline = build_pipeline(model_key, balancing, X_train, y_train)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_space(model_key, balancing),
        n_iter=SEARCH_ITER_BY_MODEL[model_key],
        scoring=SCORING,
        refit="average_precision",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        return_train_score=False,
        error_score="raise",
        verbose=1
    )

    start = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - start

    combo_id = f"{model_key}_{balancing}"
    best_idx = search.best_index_

    best_params = {}
    for key, value in search.best_params_.items():
        if isinstance(value, np.generic):
            value = value.item()
        best_params[key] = value

    row = {
        "combo_id": combo_id,
        "model": MODEL_LABELS[model_key],
        "balancing": BALANCING_LABELS[balancing],
        "search_method": "RandomizedSearchCV",
        "n_candidates": len(search.cv_results_["params"]),
        "best_params": json.dumps(best_params, ensure_ascii=False, sort_keys=True),
        "cv_accuracy_at_0_5": float(search.cv_results_["mean_test_accuracy"][best_idx]),
        "cv_precision_at_0_5": float(search.cv_results_["mean_test_precision"][best_idx]),
        "cv_recall_at_0_5": float(search.cv_results_["mean_test_recall"][best_idx]),
        "cv_f1_at_0_5": float(search.cv_results_["mean_test_f1"][best_idx]),
        "cv_roc_auc": float(search.cv_results_["mean_test_roc_auc"][best_idx]),
        "cv_average_precision": float(search.cv_results_["mean_test_average_precision"][best_idx]),
        "cv_average_precision_std": float(search.cv_results_["std_test_average_precision"][best_idx]),
        "fit_time_seconds": float(elapsed)
    }

    return search, row


def save_search_outputs(search: RandomizedSearchCV, model_key: str, balancing: str) -> None:
    combo_id = f"{model_key}_{balancing}"
    joblib.dump(search.best_estimator_, MODELS_DIR / f"{combo_id}.joblib")
    pd.DataFrame(search.cv_results_).to_csv(RESULTS_DIR / f"cv_results_{combo_id}.csv", index=False)


def best_candidate_fold_scores(combo_id: str) -> np.ndarray:
    cv_results = pd.read_csv(RESULTS_DIR / f"cv_results_{combo_id}.csv")
    best_idx = cv_results["rank_test_average_precision"].idxmin()
    columns = [c for c in cv_results.columns if c.startswith("split") and c.endswith("_test_average_precision")]
    columns.sort(key=lambda c: int(c[len("split"):c.index("_")]))

    return cv_results.loc[best_idx, columns].to_numpy(dtype=float)


def paired_comparison_vs_winner(summary: pd.DataFrame, n_train: int) -> pd.DataFrame:
    winner_id = summary.iloc[0]["combo_id"]
    winner_scores = best_candidate_fold_scores(winner_id)
    n_test = n_train / CV_FOLDS

    rows = []
    for combo_id in summary["combo_id"].iloc[1:]:
        diff = best_candidate_fold_scores(combo_id) - winner_scores
        t_stat, p_value = corrected_resampled_ttest(diff, n_train - n_test, n_test)
        rows.append({
            "combo_id": combo_id,
            "winner_combo_id": winner_id,
            "mean_ap_difference_vs_winner": float(diff.mean()),
            "t_statistic": t_stat,
            "p_value_nadeau_bengio": p_value
        })

    result = pd.DataFrame(rows)
    result["p_value_holm"] = holm_adjust_pvalues(result["p_value_nadeau_bengio"].to_numpy())
    result["exploratory_significant_vs_winner"] = result["p_value_holm"] < 0.05

    return result


def main() -> None:
    upstream = require_stages(["03_data_preparation"])
    X_train, y_train = load_training_data()

    print("\nPhase 4 CRISP-DM - modeling\n")
    print(f"Training rows: {len(X_train)}")
    print(f"Positive-class share: {y_train.mean():.4f}")
    print(f"CV: {CV_FOLDS} folds * {CV_REPEATS} repeats")
    print("Search method - RandomizedSearchCV")
    print(f"Candidates per combination: {SEARCH_ITER_BY_MODEL}")
    print("Held-out test set remains untouched in this phase\n")

    cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)

    summary_rows = []

    for model_key in MODEL_KEYS:
        for balancing in BALANCING_VARIANTS:
            print(f"{MODEL_LABELS[model_key]} | {BALANCING_LABELS[balancing]}")

            search, row = run_search(model_key=model_key, balancing=balancing, X_train=X_train, y_train=y_train, cv=cv)

            save_search_outputs(search, model_key, balancing)
            summary_rows.append(row)

            print(f"Candidates: {row['n_candidates']}")
            print(f"Best CV Average Precision: {row['cv_average_precision']:.4f}")
            print(f"Recall @0.5: {row['cv_recall_at_0_5']:.4f}")
            print(f"F1-score @0.5: {row['cv_f1_at_0_5']:.4f}")
            print(f"Best params: {row['best_params']}\n")

    summary = (pd.DataFrame(summary_rows).sort_values(MODEL_RANKING_COLUMNS, ascending=False).reset_index(drop=True))
    summary.insert(0, "cv_rank", np.arange(1, len(summary) + 1))
    summary.to_csv(RESULTS_DIR / "modeling_cv_summary.csv", index=False)

    paired = paired_comparison_vs_winner(summary, len(y_train))
    paired.to_csv(RESULTS_DIR / "model_comparison_paired.csv", index=False)
    winner = summary.iloc[0]

    metadata = {
        "phase": "CRISP-DM 4 - modeling",
        "selection_metric": "Average Precision",
        "winner_combo_id": winner["combo_id"],
        "winner_model": winner["model"],
        "winner_balancing": winner["balancing"],
        "winner_cv_average_precision": float(winner["cv_average_precision"]),
        "cv": {
            "type": "RepeatedStratifiedKFold",
            "folds": CV_FOLDS,
            "repeats": CV_REPEATS,
            "random_state": RANDOM_STATE
        },
        "search": {
            "type": "RandomizedSearchCV",
            "n_iter_per_combination": SEARCH_ITER_BY_MODEL,
            "random_state": RANDOM_STATE
        },
        "models": [MODEL_LABELS[k] for k in MODEL_KEYS],
        "balancing_variants": [BALANCING_LABELS[k] for k in BALANCING_VARIANTS],
        "smotenc_tuning": {**SMOTENC_SPACE, "numeric_scaling_before_smotenc": True},
        "xgb_scale_pos_weight_class_weights": float((y_train == 0).sum() / (y_train == 1).sum())
    }

    (RESULTS_DIR / "modeling_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    combo_ids = [f"{m}_{b}" for m in MODEL_KEYS for b in BALANCING_VARIANTS]
    outputs = [MODELS_DIR / f"{c}.joblib" for c in combo_ids] + [RESULTS_DIR / f"cv_results_{c}.csv" for c in combo_ids]
    outputs += [RESULTS_DIR / name for name in ("modeling_cv_summary.csv", "model_comparison_paired.csv", "modeling_metadata.json")]
    parameters = {"cv_folds": CV_FOLDS, "cv_repeats": CV_REPEATS, "random_state": RANDOM_STATE, "search_iter_by_model": SEARCH_ITER_BY_MODEL, "ranking_columns": MODEL_RANKING_COLUMNS, "winner_combo_id": winner["combo_id"]}
    run_id = record_stage("04_modeling", [PROCESSED_DIR / "X_train.csv", PROCESSED_DIR / "y_train.csv"], outputs, parameters, upstream)

    print("CV ranking:\n")
    print(summary[["cv_rank", "model", "balancing", "n_candidates", "cv_average_precision", "cv_recall_at_0_5", "cv_f1_at_0_5", "cv_roc_auc"]].to_string(index=False))
    print("\nExploratory paired comparison with the winner selected on the same folds (Nadeau-Bengio corrected t-test, Holm-adjusted):\n")
    print(paired.to_string(index=False))
    print(f"\nSelected winner - {winner['combo_id']} (run {run_id})")
    print("\nSaved:")
    print("- tuned pipelines:", MODELS_DIR)
    print("- CV results:", RESULTS_DIR)
    print()


if __name__ == "__main__":
    main()
