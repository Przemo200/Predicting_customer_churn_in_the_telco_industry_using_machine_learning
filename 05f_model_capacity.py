"""
Phase 4 extension - model capacity, how hyperparameters control model complexity and generalisation
"""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, validation_curve

from config import (
    CV_FOLDS,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
)
from src.manifest import record_stage, require_stages
from src.plotting import (
    REFERENCE_COLOR,
    finish_axes,
    number,
    save_figure,
    set_style,
)

TRAIN_COLOR = "#C95F5F"
VALID_COLOR = "#2F6DB5"

CAPACITY_EXPERIMENTS = [
    ("lr_none", "model__C", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0, 1000.0], {}, "Regresja logistyczna"),
    ("rf_none", "model__max_depth", [2, 3, 4, 6, 8, 10, 12, 15, 20, None], {}, "Las losowy"),
    ("rf_none", "model__min_samples_leaf", [1, 2, 3, 5, 10, 20, 50, 100, 200], {"model__max_depth": None}, "Las losowy (bez limitu głębokości)"),
    ("xgb_none", "model__max_depth", [1, 2, 3, 4, 5, 6, 8, 10], {}, "XGBoost")
]
C_TICKS = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]

BOOSTING_LEARNING_RATES = (0.02, 0.1, 0.3)
BOOSTING_DEPTHS = (1, 4)
BEST_LABEL_OFFSETS = {1: ((4, -6), "right", "top"), 4: ((0, 7), "center", "bottom")}
BOOSTING_MAX_ROUNDS = 1500
BOOSTING_ROUNDS = (10, 25, 50, 100, 200, 300, 400, 600, 800, 1000, 1250, 1500)


def load_data():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").iloc[:, 0].astype(int)
    summary = pd.read_csv(RESULTS_DIR / "modeling_cv_summary.csv")
    return X_train, y_train, summary


def cv_splitter() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def search_plateau(summary: pd.DataFrame, results_dir=RESULTS_DIR) -> pd.DataFrame:
    rows = []
    for combo_id in summary["combo_id"]:
        results = pd.read_csv(results_dir / f"cv_results_{combo_id}.csv")
        best = results.loc[results["rank_test_average_precision"].idxmin()]
        within = results["mean_test_average_precision"] >= best["mean_test_average_precision"] - best["std_test_average_precision"]
        top = results["mean_test_average_precision"].sort_values(ascending=False)
        rows.append({
            "combo_id": combo_id,
            "n_candidates": len(results),
            "best_cv_ap": float(best["mean_test_average_precision"]),
            "best_cv_ap_fold_std": float(best["std_test_average_precision"]),
            "n_within_one_fold_std_of_best": int(within.sum()),
            "gap_best_to_5th": float(top.iloc[0] - top.iloc[min(4, len(top) - 1)]),
            "gap_best_to_median": float(top.iloc[0] - top.median())
        })
    return pd.DataFrame(rows)


def run_validation_curves(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    rows = []
    for combo_id, param, values, fixed, _ in CAPACITY_EXPERIMENTS:
        tuned = joblib.load(MODELS_DIR / f"{combo_id}.joblib")
        pipeline = clone(tuned).set_params(**fixed)
        tuned_value = tuned.get_params()[param]
        print(f"{combo_id}: {param} over {values} (fixed: {fixed or 'tuned values'})")
        train_scores, valid_scores = validation_curve(pipeline, X_train, y_train, param_name=param, param_range=values, cv=cv_splitter(), scoring="average_precision", n_jobs=-1)

        for value, train, valid in zip(values, train_scores, valid_scores, strict=True):
            rows.append({
                "combo_id": combo_id,
                "hyperparameter": param.removeprefix("model__"),
                "value": "None" if value is None else value,
                "fixed_overrides": json.dumps({k.removeprefix("model__"): v for k, v in fixed.items()}),
                "tuned_value": "None" if tuned_value is None else tuned_value,
                "train_ap_mean": float(train.mean()),
                "train_ap_std": float(train.std(ddof=1)),
                "valid_ap_mean": float(valid.mean()),
                "valid_ap_std": float(valid.std(ddof=1)),
                "train_minus_valid_ap": float(train.mean() - valid.mean())
            })
    return pd.DataFrame(rows)


def boosting_rounds_curves(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    base = joblib.load(MODELS_DIR / "xgb_none.joblib")
    records = []

    for depth in BOOSTING_DEPTHS:
        for learning_rate in BOOSTING_LEARNING_RATES:
            print(f"xgb_none: max_depth={depth}, learning_rate={learning_rate}, up to {BOOSTING_MAX_ROUNDS} rounds")
            for fold, (train_idx, valid_idx) in enumerate(cv_splitter().split(X_train, y_train), start=1):
                pipeline = clone(base).set_params(model__max_depth=depth, model__learning_rate=learning_rate, model__n_estimators=BOOSTING_MAX_ROUNDS)
                pipeline.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                transform, model = pipeline[:-1], pipeline.named_steps["model"]
                X_fit, X_valid = transform.transform(X_train.iloc[train_idx]), transform.transform(X_train.iloc[valid_idx])

                for rounds in BOOSTING_ROUNDS:
                    records.append({
                        "max_depth": depth,
                        "learning_rate": learning_rate,
                        "rounds": rounds,
                        "fold": fold,
                        "train_ap": float(average_precision_score(y_train.iloc[train_idx], model.predict_proba(X_fit, iteration_range=(0, rounds))[:, 1])),
                        "valid_ap": float(average_precision_score(y_train.iloc[valid_idx], model.predict_proba(X_valid, iteration_range=(0, rounds))[:, 1]))
                    })

    frame = pd.DataFrame(records)
    return frame.groupby(["max_depth", "learning_rate", "rounds"], as_index=False).agg(train_ap_mean=("train_ap", "mean"), valid_ap_mean=("valid_ap", "mean"), valid_ap_std=("valid_ap", "std"))


def axis_values(values: pd.Series, param: str) -> tuple[np.ndarray, list[str], bool]:
    labels = [str(v) for v in values]
    if param == "C":
        return values.astype(float).to_numpy(), labels, True
    numeric = [float(v) for v in values if v != "None"]
    if "None" in labels:
        numeric.append(max(numeric) + 5)
        labels = [label if label != "None" else "brak" for label in labels]
    return np.array(numeric), labels, param == "min_samples_leaf"


def plot_validation_curves(curves: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.4))
    for ax, (combo_id, param, _, fixed, title) in zip(axes.ravel(), CAPACITY_EXPERIMENTS, strict=True):
        name = param.removeprefix("model__")
        data = curves.loc[curves["combo_id"].eq(combo_id) & curves["hyperparameter"].eq(name)]
        x, labels, log_scale = axis_values(data["value"], name)
        ax.plot(x, data["train_ap_mean"], color=TRAIN_COLOR, marker="o", markersize=3.5, linewidth=1.4, linestyle="--", label="Zbiór treningowy")
        ax.plot(x, data["valid_ap_mean"], color=VALID_COLOR, marker="o", markersize=3.5, linewidth=1.4, label="Walidacja")
        ax.fill_between(x, data["valid_ap_mean"] - data["valid_ap_std"], data["valid_ap_mean"] + data["valid_ap_std"], color=VALID_COLOR, alpha=0.15, linewidth=0, label="Walidacja ± 1 odchylenie standardowe między foldami")
        tuned = str(data["tuned_value"].iloc[0])
        if not fixed:
            position = x[[str(v) for v in data["value"]].index(tuned)] if tuned == "None" else float(tuned)
            ax.axvline(position, color=REFERENCE_COLOR, linestyle=":", linewidth=1, label="Wartość z tuningu")
        if log_scale:
            ax.set_xscale("log")
        if name == "C":
            ax.set_xticks(C_TICKS)
            ax.set_xticklabels([f"{v:g}".replace(".", ",") if i % 2 == 0 else "" for i, v in enumerate(C_TICKS)], fontsize=8)
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([label.replace(".", ",") for label in labels], fontsize=8)
        ax.minorticks_off()
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_xlabel(name)
        ax.set_ylabel("AP")
        finish_axes(ax)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=8, frameon=False)
    save_figure(fig, "fig_capacity_validation_curves.pdf")


def plot_boosting_rounds(curves: pd.DataFrame) -> None:
    colors = dict(zip(BOOSTING_LEARNING_RATES, ["#7B5BA6", "#2F6DB5", "#C98F22"], strict=True))
    fig, axes = plt.subplots(1, len(BOOSTING_DEPTHS), figsize=(6.3, 3.4), sharey=True)
    for ax, depth in zip(np.atleast_1d(axes), BOOSTING_DEPTHS, strict=True):
        panel = curves.loc[curves["max_depth"].eq(depth)]
        for learning_rate in BOOSTING_LEARNING_RATES:
            data = panel.loc[panel["learning_rate"].eq(learning_rate)]
            ax.plot(data["rounds"], data["valid_ap_mean"], color=colors[learning_rate], linewidth=1.4, marker="o", markersize=3)
            ax.plot(data["rounds"], data["train_ap_mean"], color=colors[learning_rate], linewidth=1.0, linestyle="--")
        best = panel.loc[panel["valid_ap_mean"].idxmax()]
        ax.scatter(best["rounds"], best["valid_ap_mean"], s=90, facecolors="none", edgecolors="#1A1A1A", linewidths=1.0, zorder=4)
        offset, align, valign = BEST_LABEL_OFFSETS.get(depth, ((0, 9), "center", "bottom"))
        ax.annotate(number(best["valid_ap_mean"]), (best["rounds"], best["valid_ap_mean"]), xytext=offset, textcoords="offset points", ha=align, va=valign, fontsize=7.5, color="#1A1A1A", zorder=5)
        ax.set_xscale("log")
        ticks = [10, 25, 50, 100, 200, 400, 1000]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("Liczba rund boostingu")
        ax.set_title(f"max_depth = {depth}", fontsize=9.5, loc="left")
        ax.set_ylabel("AP")
        ax.tick_params(labelleft=True)
        finish_axes(ax)
    handles = [Line2D([], [], color=colors[rate], linewidth=1.4, label=f"learning_rate = {number(rate, 2)}") for rate in BOOSTING_LEARNING_RATES]
    handles += [Line2D([], [], color=REFERENCE_COLOR, linewidth=1.4, marker="o", markersize=3, label="Walidacja"), Line2D([], [], color=REFERENCE_COLOR, linewidth=1.0, linestyle="--", label="Zbiór treningowy")]
    handles.append(Line2D([], [], linestyle="none", marker="o", markersize=8, markerfacecolor="none", markeredgecolor="#1A1A1A", label="Najlepsza walidacja"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=8, frameon=False)
    save_figure(fig, "fig_capacity_xgb_boosting_rounds.pdf")


def main() -> None:
    set_style()
    upstream = require_stages(["03_data_preparation", "04_modeling"])
    X_train, y_train, summary = load_data()

    print("\nPhase 4 extension - model capacity (training set only)\n")
    plateau = search_plateau(summary)
    curves = run_validation_curves(X_train, y_train)
    boosting = boosting_rounds_curves(X_train, y_train)
    plot_validation_curves(curves)
    plot_boosting_rounds(boosting)

    plateau.to_csv(RESULTS_DIR / "capacity_search_plateau.csv", index=False)
    curves.to_csv(RESULTS_DIR / "capacity_validation_curves.csv", index=False)
    boosting.to_csv(RESULTS_DIR / "capacity_xgb_boosting_rounds.csv", index=False)

    metadata = {
        "phase": "CRISP-DM 4 extension - model capacity",
        "data": "training set only; the test set is not used",
        "validation_curves": f"sklearn validation_curve, StratifiedKFold({CV_FOLDS}, shuffle, random_state={RANDOM_STATE}); one hyperparameter varied, the others fixed at the tuned no-balancing pipeline from phase 4 plus the listed overrides; train AP is in-sample on the fitting folds",
        "boosting_rounds": f"XGBoost no-balancing pipeline with the tuned values except max_depth in {list(BOOSTING_DEPTHS)}, learning_rate in {list(BOOSTING_LEARNING_RATES)}; AP after k rounds from iteration_range of one {BOOSTING_MAX_ROUNDS}-round ensemble per fold",
        "search_plateau": "from cv_results_*.csv of phase 4: n_within_one_fold_std_of_best = candidates whose mean CV AP is within one fold standard deviation of the best candidate",
        "experiments": [{"combo_id": c, "hyperparameter": p.removeprefix("model__"), "values": [str(v) for v in values], "fixed_overrides": {k.removeprefix("model__"): v for k, v in f.items()}} for c, p, values, f, _ in CAPACITY_EXPERIMENTS],
        "note": "the selected model and all phase 5 results are unchanged; these curves explain capacity and are not used for any selection"
    }
    (RESULTS_DIR / "capacity_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("\nSearch plateau of phase 4:\n")
    print(plateau.round(4).to_string(index=False))
    print("\nValidation curves (train vs validation AP):\n")
    print(curves.drop(columns=["fixed_overrides"]).round(4).to_string(index=False))
    print("\nXGBoost learning rate vs boosting rounds:\n")
    print(boosting.round(4).to_string(index=False))

    combo_ids = sorted({c for c, *_ in CAPACITY_EXPERIMENTS})
    inputs = [PROCESSED_DIR / "X_train.csv", PROCESSED_DIR / "y_train.csv", RESULTS_DIR / "modeling_cv_summary.csv"]
    inputs += [MODELS_DIR / f"{c}.joblib" for c in combo_ids] + [RESULTS_DIR / f"cv_results_{c}.csv" for c in summary["combo_id"]]
    outputs = [RESULTS_DIR / name for name in ("capacity_search_plateau.csv", "capacity_validation_curves.csv", "capacity_xgb_boosting_rounds.csv", "capacity_metadata.json")]
    parameters = {"cv_folds": CV_FOLDS, "random_state": RANDOM_STATE, "boosting_depths": list(BOOSTING_DEPTHS), "boosting_learning_rates": list(BOOSTING_LEARNING_RATES), "boosting_max_rounds": BOOSTING_MAX_ROUNDS}
    run_id = record_stage("05f_model_capacity", inputs, outputs, parameters, upstream)
    print(f"\nSaved capacity tables to results/ and figures to figures/ (run {run_id})\n")


if __name__ == "__main__":
    main()
