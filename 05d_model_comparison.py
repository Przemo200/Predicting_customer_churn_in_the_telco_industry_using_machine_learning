"""
Phase 5 extension - held-out test comparison of all tuned model variants
"""

from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from config import (
    BOOTSTRAP_CI,
    BOOTSTRAP_ITERATIONS,
    MODEL_RANKING_COLUMNS,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
)
from src.evaluation import (
    oof_probabilities,
    select_threshold,
    threshold_metrics,
)
from src.manifest import record_stage, require_stages
from src.plotting import (
    ACCENT_COLOR,
    MAIN_COLOR,
    REFERENCE_COLOR,
    finish_axes,
    save_figure,
    set_style,
)

MODEL_NAMES_PL = {"Logistic Regression": "Regresja logistyczna", "Random Forest": "Las losowy", "XGBoost": "XGBoost"}
BALANCING_NAMES_PL = {"No balancing": "bez balansowania", "Class weights": "wagi klas", "SMOTENC": "SMOTENC"}


def load_data():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").iloc[:, 0].astype(int)
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").iloc[:, 0].astype(int)
    summary = pd.read_csv(RESULTS_DIR / "modeling_cv_summary.csv").sort_values(MODEL_RANKING_COLUMNS, ascending=False).reset_index(drop=True)
    return X_train, X_test, y_train, y_test, summary


def point_row(label: str, model: str, balancing: str, y_test: pd.Series, scores: np.ndarray, threshold: float) -> dict:
    metrics = threshold_metrics(y_test, scores, threshold, "oof_f1")
    metrics_default = threshold_metrics(y_test, scores, 0.5, "default_0.5")
    return {
        "combo_id": label,
        "model": model,
        "balancing": balancing,
        "threshold": threshold,
        "average_precision": float(average_precision_score(y_test, scores)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy_at_0_5": metrics_default["accuracy"],
        "recall_at_0_5": metrics_default["recall"]
    }


def paired_bootstrap(y: np.ndarray, scores: dict[str, np.ndarray], thresholds: dict[str, float], winner: str) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y)
    records = []

    for _ in range(BOOTSTRAP_ITERATIONS):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue

        record = {}
        for name, s in scores.items():
            sb = s[idx]
            record[(name, "average_precision")] = average_precision_score(yb, sb)
            record[(name, "roc_auc")] = roc_auc_score(yb, sb)
            record[(name, "f1")] = f1_score(yb, (sb >= thresholds[name]).astype(int), zero_division=0)
        records.append(record)

    frame = pd.DataFrame(records)
    alpha = (1.0 - BOOTSTRAP_CI) / 2.0
    rows = []

    for name in scores:
        row = {"combo_id": name}
        for metric in ("average_precision", "roc_auc", "f1"):
            row[f"{metric}_ci_low"] = float(frame[(name, metric)].quantile(alpha))
            row[f"{metric}_ci_high"] = float(frame[(name, metric)].quantile(1.0 - alpha))

        for metric in ("average_precision", "roc_auc"):
            diff = frame[(name, metric)] - frame[(winner, metric)]
            row[f"{metric}_diff_vs_winner"] = float(diff.mean())
            row[f"{metric}_diff_ci_low"] = float(diff.quantile(alpha))
            row[f"{metric}_diff_ci_high"] = float(diff.quantile(1.0 - alpha))

        rows.append(row)

    return pd.DataFrame(rows)


def plot_model_comparison(results: pd.DataFrame, winner: str) -> None:
    models = results.loc[results["combo_id"].str.startswith(("lr_", "rf_", "xgb_"))].sort_values("average_precision")
    baselines = results.set_index("combo_id")["average_precision"]
    labels = [f"{MODEL_NAMES_PL[m]} ({BALANCING_NAMES_PL[b]})" for m, b in zip(models["model"], models["balancing"], strict=True)]
    y_pos = np.arange(len(models))
    is_winner = models["combo_id"].eq(winner).to_numpy()

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.errorbar(models["average_precision"], y_pos, xerr=[models["average_precision"] - models["average_precision_ci_low"], models["average_precision_ci_high"] - models["average_precision"]], fmt="none", ecolor=REFERENCE_COLOR, elinewidth=1, capsize=2)
    ax.scatter(models["average_precision"][~is_winner], y_pos[~is_winner], color=MAIN_COLOR, zorder=3, s=22, label="Zbiór testowy (95% CI)")
    ax.scatter(models["average_precision"][is_winner], y_pos[is_winner], color=ACCENT_COLOR, zorder=3, s=22, label="Model końcowy, zbiór testowy (95% CI)")
    ax.scatter(models["cv_average_precision"], y_pos, facecolors="none", edgecolors=REFERENCE_COLOR, zorder=3, s=22, label="Walidacja krzyżowa")
    ax.axvline(baselines["baseline_month_to_month_rule"], color=REFERENCE_COLOR, linestyle="--", linewidth=1, label="Reguła (umowa miesięczna)")
    ax.axvline(baselines["baseline_no_skill"], color=REFERENCE_COLOR, linestyle=":", linewidth=1, label="Kontakt ze wszystkimi")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Average Precision")
    values = np.r_[models["average_precision_ci_low"], models["average_precision_ci_high"], models["cv_average_precision"], baselines[["baseline_month_to_month_rule", "baseline_no_skill"]]]
    ax.set_xlim(np.floor(values.min() * 20) / 20 - 0.05, np.ceil(values.max() * 20) / 20 + 0.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.22, -0.13), ncol=2, fontsize=8.5)
    finish_axes(ax)
    save_figure(fig, "fig_model_comparison.pdf")


def main() -> None:
    set_style()
    upstream = require_stages(["03_data_preparation", "04_modeling"])
    X_train, X_test, y_train, y_test, summary = load_data()
    winner = str(summary.iloc[0]["combo_id"])
    y = y_test.to_numpy()

    print("\nPhase 5 extension - test comparison of all model variants\n")
    print(f"Phase 4 winner: {winner}\n")

    scores, thresholds, rows = {}, {}, []
    oof_by_model, oof_folds = {}, None

    for _, entry in summary.iterrows():
        combo_id = str(entry["combo_id"])
        print(f"{combo_id}: OOF threshold and test scores...")

        pipeline = joblib.load(MODELS_DIR / f"{combo_id}.joblib")
        oof_scores, oof_folds = oof_probabilities(pipeline, X_train, y_train, verbose=False)
        threshold, _ = select_threshold(y_train, oof_scores)
        test_scores = pipeline.predict_proba(X_test)[:, 1]

        scores[combo_id] = test_scores
        thresholds[combo_id] = threshold
        oof_by_model[combo_id] = oof_scores

        row = point_row(combo_id, str(entry["model"]), str(entry["balancing"]), y_test, test_scores, threshold)
        row["brier"] = float(brier_score_loss(y_test, test_scores))
        row["cv_average_precision"] = float(entry["cv_average_precision"])
        rows.append(row)

    results = pd.DataFrame(rows).merge(paired_bootstrap(y, scores, thresholds, winner), on="combo_id")
    results = results.sort_values("average_precision", ascending=False).reset_index(drop=True)

    baselines = [
        point_row("baseline_no_skill", "Predict everyone as churn", "-", y_test, np.ones(len(y)), 1.0),
        point_row("baseline_month_to_month_rule", "Churn if Contract is Month-to-month", "-", y_test, (X_test["Contract"] == "Month-to-month").to_numpy(dtype=float), 1.0)
    ]
    results = pd.concat([results, pd.DataFrame(baselines)], ignore_index=True)
    results.to_csv(RESULTS_DIR / "test_comparison_all_models.csv", index=False)
    plot_model_comparison(results, winner)

    pd.DataFrame({"y_true": y_train.to_numpy(), "oof_fold": oof_folds, **oof_by_model}).to_csv(RESULTS_DIR / "oof_scores_all_models.csv", index=False)
    pd.DataFrame({"row_index": np.arange(len(y)), "y_true": y, **scores}).to_csv(RESULTS_DIR / "test_scores_all_models.csv", index=False)

    columns = [
        "combo_id", "threshold", "average_precision", "average_precision_ci_low", "average_precision_ci_high",
        "roc_auc", "accuracy", "precision", "recall", "f1", "accuracy_at_0_5", "recall_at_0_5", "brier",
        "average_precision_diff_vs_winner", "average_precision_diff_ci_low", "average_precision_diff_ci_high"
    ]
    print("\nHeld-out test results (threshold = max F1 on single 5-fold OOF predictions of each model; *_at_0_5 = default threshold 0.5; brier = raw, uncalibrated scores):\n")
    print(results[columns].round(3).to_string(index=False))
    inputs = [PROCESSED_DIR / name for name in ("X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv")]
    inputs += [RESULTS_DIR / "modeling_cv_summary.csv"] + [MODELS_DIR / f"{combo_id}.joblib" for combo_id in summary["combo_id"]]
    parameters = {"bootstrap_iterations": BOOTSTRAP_ITERATIONS, "bootstrap_ci": BOOTSTRAP_CI, "random_state": RANDOM_STATE, "winner_combo_id": winner}
    outputs = [RESULTS_DIR / name for name in ("test_comparison_all_models.csv", "oof_scores_all_models.csv", "test_scores_all_models.csv")]
    run_id = record_stage("05d_model_comparison", inputs, outputs, parameters, upstream)
    print(f"\nSaved - test_comparison_all_models.csv, oof_scores_all_models.csv, test_scores_all_models.csv (run {run_id})\n")


if __name__ == "__main__":
    main()