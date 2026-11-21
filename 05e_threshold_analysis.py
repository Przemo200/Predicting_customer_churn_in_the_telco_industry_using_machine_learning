"""
Phase 5 extension - choosing the decision threshold for FN/FP requirements
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from config import (
    BOOTSTRAP_CI,
    BOOTSTRAP_ITERATIONS,
    CONTACT_CAPACITY,
    COST_RATIOS_FN_TO_FP,
    MODEL_RANKING_COLUMNS,
    PROCESSED_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    RETENTION_SCENARIOS,
    TARGET_RECALL,
)
from src.evaluation import (
    benefit_kwargs,
    contact_net_benefit,
    decision_summary,
    operating_points,
    policy_outcome,
    select_min_cost,
    select_min_fpr_at_recall,
    select_threshold,
    top_fraction_mask,
)
from src.manifest import record_stage, require_stages
from src.plotting import (
    REFERENCE_COLOR,
    finish_axes,
    number,
    percent,
    percent_formatter,
    save_figure,
    set_style,
)

STRATEGIES = ("default_0.5", "max_f1", "min_fpr_at_target_recall", "min_cost_base_scenario", "top_capacity")
MAIN_STRATEGIES = STRATEGIES[:4]
STRATEGY_LABELS = {
    "default_0.5": "Domyślny",
    "max_f1": "Maks. F1",
    "min_fpr_at_target_recall": f"Czułość ≥ {percent(TARGET_RECALL, 0)}",
    "min_cost_base_scenario": "Min. kosztu, scenariusz bazowy",
    "top_capacity": f"Top {percent(CONTACT_CAPACITY, 0)} klientów"
}

STRATEGY_COLORS = dict(zip(STRATEGIES, ["#2F6DB5", "#C95F5F", "#7B5BA6", "#C98F22", "#3F8A4E"], strict=True))
STRATEGY_MARKERS = dict(zip(STRATEGIES, ["o", "s", "D", "^", "v"], strict=True))
THRESHOLD_LINE_STYLES = dict(zip(MAIN_STRATEGIES, [":", "--", "-.", (0, (5, 1.5, 1, 1.5, 1, 1.5))], strict=True))
FAMILY_STYLES = {"xgb": ("#2F6DB5", "-"), "rf": ("#3F8A4E", "--"), "lr": ("#C95F5F", "-.")}
ZOOM_FPR, ZOOM_TPR = (0.34, 0.48), (0.86, 0.94)
FPR_LABEL = "Odsetek fałszywych alarmów (FPR)"
TPR_LABEL = "Czułość (TPR)"

MODEL_NAMES_PL = {"Logistic Regression": "Regresja logistyczna", "Random Forest": "Las losowy", "XGBoost": "XGBoost"}
BALANCING_NAMES_PL = {"No balancing": "bez balansowania", "Class weights": "wagi klas", "SMOTENC": "SMOTENC"}
SCENARIO_LABELS = {"pessimistic": "Pesymistyczny", "base": "Bazowy", "optimistic": "Optymistyczny"}


def load_inputs() -> dict:
    return {
        "X_train": pd.read_csv(PROCESSED_DIR / "X_train.csv"),
        "X_test": pd.read_csv(PROCESSED_DIR / "X_test.csv"),
        "y_train": pd.read_csv(PROCESSED_DIR / "y_train.csv").iloc[:, 0].astype(int),
        "y_test": pd.read_csv(PROCESSED_DIR / "y_test.csv").iloc[:, 0].astype(int),
        "oof": pd.read_csv(RESULTS_DIR / "training_oof_predictions.csv"),
        "test": pd.read_csv(RESULTS_DIR / "test_predictions.csv"),
        "metadata": json.loads((RESULTS_DIR / "final_model_metadata.json").read_text(encoding="utf-8")),
        "summary": pd.read_csv(RESULTS_DIR / "modeling_cv_summary.csv").sort_values(MODEL_RANKING_COLUMNS, ascending=False).reset_index(drop=True),
        "retention": pd.read_csv(RESULTS_DIR / "retention_scenarios.csv"),
        "oof_all": pd.read_csv(RESULTS_DIR / "oof_scores_all_models.csv"),
        "test_all": pd.read_csv(RESULTS_DIR / "test_scores_all_models.csv")
    }


def check_consistency(data: dict, winner: str) -> dict:
    y_train, y_test = data["y_train"].to_numpy(), data["y_test"].to_numpy()
    checks = {
        "oof_labels_match_y_train": bool(np.array_equal(data["oof"]["y_true"].to_numpy(), y_train) and np.array_equal(data["oof_all"]["y_true"].to_numpy(), y_train)),
        "test_labels_match_y_test": bool(np.array_equal(data["test"]["y_true"].to_numpy(), y_test) and np.array_equal(data["test_all"]["y_true"].to_numpy(), y_test)),
        "oof_folds_match_05": bool(np.array_equal(data["oof"]["oof_fold"].to_numpy(), data["oof_all"]["oof_fold"].to_numpy())),
        "final_model_oof_scores_match_05d": bool(np.allclose(data["oof"]["oof_score"], data["oof_all"][winner], rtol=0, atol=1e-12)),
        "final_model_test_scores_match_05d": bool(np.allclose(data["test"]["churn_score"], data["test_all"][winner], rtol=0, atol=1e-12)),
        "max_f1_threshold_reproduced": bool(np.isclose(select_threshold(data["y_train"], data["oof"]["oof_score"].to_numpy())[0], data["metadata"]["selected_threshold"], rtol=0, atol=1e-12)),
        "final_model_is_phase_4_winner": data["metadata"]["winner_combo_id"] == winner
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"Inconsistent upstream artefacts: {failed}")
    return checks


def scenario_error_costs(params: dict, monthly_charges: float) -> tuple[float, float]:
    gain_tp, loss_fp = contact_net_benefit(np.array([1, 0]), np.array([monthly_charges, monthly_charges]), **benefit_kwargs(params))
    return max(float(gain_tp), 0.0), float(-loss_fp)


def select_strategies(y_train: pd.Series, oof_scores: np.ndarray, f1_threshold: float, cost_fn: float, cost_fp: float) -> pd.DataFrame:
    points = operating_points(y_train, oof_scores)
    recall_row = select_min_fpr_at_recall(points, TARGET_RECALL)
    cost_row = select_min_cost(points, cost_fn, cost_fp)

    return pd.DataFrame([
        {"strategy": "default_0.5", "threshold": 0.5, "rule": "fixed default threshold, not chosen on data"},
        {"strategy": "max_f1", "threshold": float(f1_threshold), "rule": "max F1 on training OOF scores (phase 5); F1 ties to the lowest threshold"},
        {"strategy": "min_fpr_at_target_recall", "threshold": float(recall_row["threshold"]), "rule": f"min FPR subject to recall >= {TARGET_RECALL} on training OOF scores; ties to higher recall, then higher threshold"},
        {"strategy": "min_cost_base_scenario", "threshold": float(cost_row["threshold"]), "rule": f"min FN * {cost_fn:.2f} + FP * {cost_fp:.2f} USD (base retention scenario at mean training MonthlyCharges) on training OOF scores; ties to fewer contacts"},
        {"strategy": "top_capacity", "threshold": np.nan, "rule": f"rank-based: top {CONTACT_CAPACITY:.0%} of the scored customers (stable order for ties)"}
    ])


def strategy_mask(strategy: str, threshold: float, scores: np.ndarray) -> np.ndarray:
    if strategy == "top_capacity":
        return top_fraction_mask(scores, CONTACT_CAPACITY)
    return np.asarray(scores) >= threshold


def evaluate_strategies(strategies: pd.DataFrame, splits: dict) -> tuple[pd.DataFrame, dict]:
    rows, masks = [], {}
    for split, (y, scores) in splits.items():
        for _, strategy in strategies.iterrows():
            contact = strategy_mask(strategy["strategy"], strategy["threshold"], scores)
            masks[(split, strategy["strategy"])] = contact
            summary = decision_summary(y, contact)
            rows.append({
                "split": split,
                "strategy": strategy["strategy"],
                "threshold": strategy["threshold"],
                **summary,
                "meets_target_recall": summary["recall"] >= TARGET_RECALL
            })
    return pd.DataFrame(rows), masks


def bootstrap_frozen_decisions(y: np.ndarray, masks: dict[str, np.ndarray]) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    y = np.asarray(y).astype(int)
    n = len(y)
    records = {name: [] for name in masks}

    for _ in range(BOOTSTRAP_ITERATIONS):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        for name, mask in masks.items():
            summary = decision_summary(yb, mask[idx])
            records[name].append((summary["recall"], summary["fpr"], summary["precision"]))

    alpha = (1.0 - BOOTSTRAP_CI) / 2.0
    rows = []
    for name, values in records.items():
        frame = pd.DataFrame(values, columns=["recall", "fpr", "precision"])
        row = {"strategy": name, "bootstrap_iterations": len(frame), "ci_level": BOOTSTRAP_CI}
        for metric in frame.columns:
            row[f"{metric}_ci_low"] = float(frame[metric].quantile(alpha))
            row[f"{metric}_ci_high"] = float(frame[metric].quantile(1.0 - alpha))
        rows.append(row)
    return pd.DataFrame(rows)


def capacity_check(points: pd.DataFrame, y_test: np.ndarray, test_scores: np.ndarray) -> dict:
    recall_row = select_min_fpr_at_recall(points, TARGET_RECALL)
    within = points.loc[points["contact_rate"] <= CONTACT_CAPACITY]
    first = points.iloc[0]
    n, positives = int(first[["tp", "fp", "fn", "tn"]].sum()), int(first["fn"])
    test_top = decision_summary(y_test, top_fraction_mask(test_scores, CONTACT_CAPACITY))
    return {
        "target_recall": TARGET_RECALL,
        "contact_capacity": CONTACT_CAPACITY,
        "oof_min_contact_rate_for_target_recall": float(recall_row["contact_rate"]),
        "oof_max_recall_within_capacity": float(within["recall"].max()),
        "perfect_ranking_max_recall_within_capacity": float(min(1.0, CONTACT_CAPACITY * n / positives)),
        "test_recall_of_top_capacity": test_top["recall"],
        "jointly_feasible_on_oof": bool(recall_row["contact_rate"] <= CONTACT_CAPACITY)
    }


def cost_ratio_analysis(y_train: pd.Series, oof_scores: np.ndarray, y_test: pd.Series, test_scores: np.ndarray, scenario_costs: dict) -> pd.DataFrame:
    points = operating_points(y_train, oof_scores)
    entries = [("assumed_ratio", float(r), np.nan, np.nan) for r in COST_RATIOS_FN_TO_FP]
    entries += [(f"retention_scenario_{name}", fn / fp, fn, fp) for name, (fn, fp) in scenario_costs.items()]
    y_test_arr = np.asarray(y_test)
    positives_test, negatives_test = int(y_test_arr.sum()), int(len(y_test_arr) - y_test_arr.sum())
    rows = []

    for source, ratio, cost_fn_usd, cost_fp_usd in entries:
        selected = select_min_cost(points, ratio, 1.0)
        test = decision_summary(y_test, np.asarray(test_scores) >= selected["threshold"])
        rows.append({
            "source": source,
            "cost_ratio_fn_to_fp": ratio,
            "cost_fn_usd_at_mean_charges": cost_fn_usd,
            "cost_fp_usd_at_mean_charges": cost_fp_usd,
            "oof_threshold": float(selected["threshold"]),
            "oof_contact_rate": float(selected["contact_rate"]),
            "oof_recall": float(selected["recall"]),
            "oof_fpr": float(selected["fpr"]),
            "test_contacted": test["contacted"],
            "test_tp": test["tp"],
            "test_fp": test["fp"],
            "test_fn": test["fn"],
            "test_recall": test["recall"],
            "test_fpr": test["fpr"],
            "test_cost": test["fn"] * ratio + test["fp"],
            "test_cost_contact_nobody": positives_test * ratio,
            "test_cost_contact_all": float(negatives_test)
        })

    return pd.DataFrame(rows).sort_values(["cost_ratio_fn_to_fp", "source"]).reset_index(drop=True)


def retention_outcomes(strategies: pd.DataFrame, masks: dict, retention: pd.DataFrame, y_test: np.ndarray, test_scores: np.ndarray, test_charges: np.ndarray) -> pd.DataFrame:
    rows = []
    for scenario, params in RETENTION_SCENARIOS.items():
        policies = [(name, masks[("test", name)]) for name in strategies["strategy"]]
        fraction = float(retention.set_index("scenario").loc[scenario, "best_fraction_selected_on_oof"])
        policies.append(("retention_optimal_fraction_phase_5", top_fraction_mask(test_scores, fraction)))
        for name, contact in policies:
            rows.append({"split": "test", "scenario": scenario, "strategy": name, **policy_outcome(y_test, test_charges, contact, params)})
    return pd.DataFrame(rows)


def fpr_at_recall(y: np.ndarray, scores: np.ndarray) -> float:
    return float(select_min_fpr_at_recall(operating_points(y, scores), TARGET_RECALL)["fpr"])


def compare_models(summary: pd.DataFrame, oof_all: pd.DataFrame, test_all: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    rows = []
    for _, entry in summary.iterrows():
        combo_id = str(entry["combo_id"])
        oof_scores, test_scores = oof_all[combo_id].to_numpy(), test_all[combo_id].to_numpy()
        selected = select_min_fpr_at_recall(operating_points(y_train, oof_scores), TARGET_RECALL)
        test = decision_summary(y_test, test_scores >= selected["threshold"])

        rows.append({
            "combo_id": combo_id,
            "model": entry["model"],
            "balancing": entry["balancing"],
            "cv_rank_by_average_precision": int(entry["cv_rank"]),
            "cv_average_precision": float(entry["cv_average_precision"]),
            "cv_roc_auc": float(entry["cv_roc_auc"]),
            "oof_average_precision": float(average_precision_score(y_train, oof_scores)),
            "oof_roc_auc": float(roc_auc_score(y_train, oof_scores)),
            "oof_threshold_at_target_recall": float(selected["threshold"]),
            "oof_recall": float(selected["recall"]),
            "oof_fpr": float(selected["fpr"]),
            "oof_precision": float(selected["precision"]),
            "oof_tp": int(selected["tp"]),
            "oof_fp": int(selected["fp"]),
            "oof_fn": int(selected["fn"]),
            "oof_contact_rate": float(selected["contact_rate"]),
            "test_recall_frozen_threshold": test["recall"],
            "test_fpr_frozen_threshold": test["fpr"],
            "test_precision_frozen_threshold": test["precision"],
            "test_tp_frozen_threshold": test["tp"],
            "test_fp_frozen_threshold": test["fp"],
            "test_fn_frozen_threshold": test["fn"],
            "test_contact_rate_frozen_threshold": test["contact_rate"],
            "test_meets_target_recall": test["recall"] >= TARGET_RECALL
        })

    frame = pd.DataFrame(rows)
    frame["oof_rank_by_fpr_at_target_recall"] = frame["oof_fpr"].rank(method="min").astype(int)
    return frame


def paired_bootstrap_fpr(y: np.ndarray, scores: dict[str, np.ndarray], reference: str) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    y = np.asarray(y).astype(int)
    n = len(y)
    records = []

    for _ in range(BOOTSTRAP_ITERATIONS):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        records.append({name: fpr_at_recall(yb, s[idx]) for name, s in scores.items()})

    frame = pd.DataFrame(records)
    alpha = (1.0 - BOOTSTRAP_CI) / 2.0
    rows = []
    for name in scores:
        diff = frame[name] - frame[reference]
        rows.append({
            "combo_id": name,
            "oof_fpr_ci_low": float(frame[name].quantile(alpha)),
            "oof_fpr_ci_high": float(frame[name].quantile(1.0 - alpha)),
            "oof_fpr_diff_vs_ap_winner_bootstrap_mean": float(diff.mean()),
            "oof_fpr_diff_ci_low": float(diff.quantile(alpha)),
            "oof_fpr_diff_ci_high": float(diff.quantile(1.0 - alpha)),
            "bootstrap_iterations": len(frame)
        })
    return pd.DataFrame(rows)


def model_label(model: str, balancing: str) -> str:
    return MODEL_NAMES_PL[model] if balancing == "No balancing" else f"{MODEL_NAMES_PL[model]} ({BALANCING_NAMES_PL[balancing]})"


def threshold_label(strategy: str, threshold: float) -> str:
    return f"{STRATEGY_LABELS[strategy]} (próg {number(threshold)})"


def plot_roc_models(y_train: pd.Series, oof_all: pd.DataFrame, comparison: pd.DataFrame) -> None:
    family_best = comparison.sort_values("cv_rank_by_average_precision").groupby(comparison["combo_id"].str.split("_").str[0], sort=False).head(1)

    fig, (ax, zoom) = plt.subplots(1, 2, figsize=(6.3, 3.6), gridspec_kw={"width_ratios": [1, 1.1]})
    for _, row in family_best.sort_values("oof_roc_auc", ascending=False, kind="stable").iterrows():
        family = row["combo_id"].split("_")[0]
        color, style = FAMILY_STYLES[family]
        fpr, tpr, _ = roc_curve(y_train, oof_all[row["combo_id"]])
        ax.plot(fpr, tpr, color=color, linestyle=style, linewidth=1.3, label=f"{model_label(row['model'], row['balancing'])} (AUC {number(row['oof_roc_auc'])} oraz FPR {number(row['oof_fpr'])} przy czułości {percent(TARGET_RECALL, 0)})")
        zoom.plot(fpr, tpr, color=color, linestyle=style, linewidth=1.3)
        zoom.scatter(row["oof_fpr"], row["oof_recall"], color=color, s=60, zorder=4, edgecolors="white", linewidths=1.0)

    ax.plot([0, 1], [0, 1], color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
    ax.add_patch(plt.Rectangle((ZOOM_FPR[0], ZOOM_TPR[0]), ZOOM_FPR[1] - ZOOM_FPR[0], ZOOM_TPR[1] - ZOOM_TPR[0], fill=False, edgecolor=REFERENCE_COLOR, linewidth=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(FPR_LABEL)
    ax.set_ylabel(TPR_LABEL)
    ax.set_title("Krzywe ROC (OOF)", fontsize=9, loc="left")

    zoom.axhline(TARGET_RECALL, color=REFERENCE_COLOR, linestyle=":", linewidth=1)
    zoom.text(ZOOM_FPR[1], TARGET_RECALL, f"czułość {percent(TARGET_RECALL, 0)} ", ha="right", va="bottom", fontsize=8, color=REFERENCE_COLOR)
    zoom.set_xlim(*ZOOM_FPR)
    zoom.set_xticks([0.35, 0.40, 0.45])
    zoom.set_ylim(*ZOOM_TPR)
    zoom.set_xlabel(FPR_LABEL)
    zoom.set_ylabel(TPR_LABEL)
    zoom.set_title(f"Przybliżenie (czułość co najmniej {percent(TARGET_RECALL, 0)})", fontsize=9, loc="left")
    for target in (ax, zoom):
        finish_axes(target)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=1, fontsize=8, frameon=False)
    save_figure(fig, "fig_roc_models_target_recall.pdf")


def plot_roc_thresholds(y_test: pd.Series, test_scores: np.ndarray, results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    fpr, tpr, _ = roc_curve(y_test, test_scores)
    ax.plot(fpr, tpr, color=REFERENCE_COLOR, linewidth=1.2, label=f"Model końcowy (AUC {number(roc_auc_score(y_test, test_scores))})")
    ax.plot([0, 1], [0, 1], color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)

    test = results.loc[results["split"].eq("test")].set_index("strategy")
    for strategy in MAIN_STRATEGIES:
        row = test.loc[strategy]
        color = STRATEGY_COLORS[strategy]
        ax.plot([0, row["fpr"]], [row["recall"], row["recall"]], color=color, linestyle=":", linewidth=1.0, zorder=2)
        ax.text(0.008, row["recall"] + 0.008, number(row["recall"]), color=color, fontsize=8, va="bottom")
        ax.scatter(row["fpr"], row["recall"], color=color, marker=STRATEGY_MARKERS[strategy], s=85, zorder=4, edgecolors="white", linewidths=1.0, label=threshold_label(strategy, row["threshold"]))

    ax.set_xlabel(FPR_LABEL)
    ax.set_ylabel(TPR_LABEL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8, title="Zbiór testowy", title_fontsize=8, frameon=False)
    finish_axes(ax)
    save_figure(fig, "fig_roc_thresholds.pdf")


def plot_fn_fp_threshold(points: pd.DataFrame, strategies: pd.DataFrame) -> None:
    data = points.loc[np.isfinite(points["threshold"])]
    n = int(points.iloc[0][["tp", "fp", "fn", "tn"]].sum())
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(data["threshold"], data["contacted"], color="#C98F22", linewidth=1.4, label="Kontaktowani (TP + FP)")
    ax.plot(data["threshold"], data["fp"], color="#2F6DB5", linewidth=1.4, label="FP (niepotrzebne kontakty)")
    ax.plot(data["threshold"], data["fn"], color="#C95F5F", linewidth=1.4, label="FN (pominięci rezygnujący)")

    for strategy, row in strategies.set_index("strategy").loc[list(MAIN_STRATEGIES)].iterrows():
        ax.axvline(row["threshold"], color=REFERENCE_COLOR, linestyle=THRESHOLD_LINE_STYLES[strategy], linewidth=1.0, label=threshold_label(strategy, row["threshold"]))

    ax.set_xlabel("Próg decyzyjny")
    ax.set_ylabel("Liczba klientów (OOF)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n * 1.02)
    ax.legend(loc="upper right", bbox_to_anchor=(1.08, 1.0), fontsize=8, frameon=True, facecolor="white", edgecolor="none", framealpha=1, handlelength=3.2)
    finish_axes(ax)
    save_figure(fig, "fig_fn_fp_threshold.pdf")


def plot_confusion_matrices(results: pd.DataFrame) -> None:
    test = results.loc[results["split"].eq("test")].set_index("strategy")
    cmap = LinearSegmentedColormap.from_list("blues", ["#F4F7FB", "#2F6DB5"])
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.6))

    for ax, strategy in zip(axes.ravel(), MAIN_STRATEGIES, strict=True):
        row = test.loc[strategy]
        counts = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=int)
        shares = counts / counts.sum(axis=1, keepdims=True)
        names = np.array([["TN", "FP"], ["FN", "TP"]])
        ax.imshow(shares, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                color = "white" if shares[i, j] > 0.6 else "#1A1A1A"
                ax.text(j, i - 0.1, str(counts[i, j]), ha="center", va="center", fontsize=13, color=color)
                ax.text(j, i + 0.17, names[i, j], ha="center", va="center", fontsize=8, color=color, alpha=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Nie", "Tak"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Nie", "Tak"])
        ax.set_title(threshold_label(strategy, row["threshold"]), fontsize=9.5, pad=9)
        ax.set_xlabel("Przewidziany churn", labelpad=13)
        ax.set_ylabel("Rzeczywisty churn", labelpad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
    save_figure(fig, "fig_confusion_matrices_thresholds.pdf")


SCENARIO_STYLES = {"pessimistic": ("#C95F5F", "v"), "base": (STRATEGY_COLORS["min_cost_base_scenario"], "D"), "optimistic": ("#3F8A4E", "^")}
SCENARIO_LABEL_OFFSETS = {
    "oof_threshold": {"pessimistic": (7, -3), "base": (-7, -11), "optimistic": (6, 4)},
    "oof_contact_rate": {"pessimistic": (7, 2), "base": (6, -11), "optimistic": (6, -11)}
}


def plot_cost_ratios(costs: pd.DataFrame, recall_threshold: float) -> None:
    grid = costs.loc[costs["source"].eq("assumed_ratio")]
    scenarios = costs.loc[costs["source"].ne("assumed_ratio")]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.4, 5.4), sharex=True)
    panels = ((top, "oof_threshold", "Optymalny próg", number), (bottom, "oof_contact_rate", "Kontaktowani klienci", percent))

    for ax, column, ylabel, fmt in panels:
        ax.scatter(grid["cost_ratio_fn_to_fp"], grid[column].where(np.isfinite(grid[column])), marker="o", s=30, facecolors="white", edgecolors="#2F6DB5", linewidths=1.3, zorder=3, label="Inne założone relacje FN / FP")
        for _, row in scenarios.iterrows():
            name = row["source"].removeprefix("retention_scenario_")
            color, marker = SCENARIO_STYLES[name]
            ratio = row["cost_ratio_fn_to_fp"]
            ax.axvline(ratio, color=color, linestyle=":", linewidth=0.8, zorder=1)
            ax.scatter(ratio, row[column], marker=marker, s=50, color=color, edgecolors="white", linewidths=0.8, zorder=4, label=f"Scenariusz {SCENARIO_LABELS[name].lower()} (FN / FP = {number(ratio, 2)})")
            offset = SCENARIO_LABEL_OFFSETS[column][name]
            ax.annotate(fmt(row[column]), (ratio, row[column]), xytext=offset, textcoords="offset points", ha="left" if offset[0] > 0 else "right", fontsize=8, color="#1A1A1A")
        ax.set_ylim(0, 0.85)
        ax.set_ylabel(ylabel)
        finish_axes(ax)

    top.axhline(recall_threshold, color=STRATEGY_COLORS["min_fpr_at_target_recall"], linestyle="--", linewidth=1.0, zorder=1, label=threshold_label("min_fpr_at_target_recall", recall_threshold))
    bottom.yaxis.set_major_formatter(percent_formatter())
    bottom.set_xscale("log")
    bottom.set_xticks(grid["cost_ratio_fn_to_fp"])
    bottom.set_xticklabels([f"{v:g}".replace(".", ",") for v in grid["cost_ratio_fn_to_fp"]])
    bottom.minorticks_off()
    bottom.set_xlabel("Relacja kosztów FN / FP")
    bottom.legend(*top.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=8.5, frameon=False)
    save_figure(fig, "fig_cost_ratio_threshold.pdf")


def main() -> None:
    set_style()
    upstream = require_stages(["03_data_preparation", "04_modeling", "05_evaluation", "05d_model_comparison"])
    data = load_inputs()
    summary = data["summary"]
    winner = str(summary.iloc[0]["combo_id"])
    y_train, y_test = data["y_train"], data["y_test"]
    oof_scores, test_scores = data["oof"]["oof_score"].to_numpy(), data["test"]["churn_score"].to_numpy()

    print("\nPhase 5 extension - business operating points\n")
    print(f"Final model (phase 4 AP winner): {winner}")
    checks = check_consistency(data, winner)
    print(f"Consistency checks of 05 and 05d artefacts passed: {list(checks)}")

    mean_charges = float(data["X_train"]["MonthlyCharges"].mean())
    scenario_costs = {name: scenario_error_costs(params, mean_charges) for name, params in RETENTION_SCENARIOS.items()}
    base_fn, base_fp = scenario_costs["base"]

    points = operating_points(y_train, oof_scores)
    strategies = select_strategies(y_train, oof_scores, float(data["metadata"]["selected_threshold"]), base_fn, base_fp)
    splits = {"oof": (y_train.to_numpy(), oof_scores), "test": (y_test.to_numpy(), test_scores)}
    results, masks = evaluate_strategies(strategies, splits)
    intervals = bootstrap_frozen_decisions(y_test.to_numpy(), {name: masks[("test", name)] for name in strategies["strategy"]})
    results = results.merge(intervals.assign(split="test"), on=["split", "strategy"], how="left")
    capacity = capacity_check(points, y_test.to_numpy(), test_scores)

    costs = cost_ratio_analysis(y_train, oof_scores, y_test, test_scores, scenario_costs)
    retention = retention_outcomes(strategies, masks, data["retention"], y_test.to_numpy(), test_scores, data["X_test"]["MonthlyCharges"].to_numpy(dtype=float))

    comparison = compare_models(summary, data["oof_all"], data["test_all"], y_train, y_test)
    scores_by_model = {c: data["oof_all"][c].to_numpy() for c in summary["combo_id"]}
    print(f"Paired bootstrap of FPR at recall >= {TARGET_RECALL} on OOF predictions ({BOOTSTRAP_ITERATIONS} resamples)...")
    comparison = comparison.merge(paired_bootstrap_fpr(y_train.to_numpy(), scores_by_model, winner), on="combo_id").sort_values(["oof_fpr", "cv_rank_by_average_precision"]).reset_index(drop=True)

    points.to_csv(RESULTS_DIR / "operating_points_oof.csv", index=False)
    results.to_csv(RESULTS_DIR / "threshold_strategies.csv", index=False)
    strategies.to_csv(RESULTS_DIR / "threshold_strategy_rules.csv", index=False)
    costs.to_csv(RESULTS_DIR / "threshold_cost_ratios.csv", index=False)
    retention.to_csv(RESULTS_DIR / "threshold_strategies_retention.csv", index=False)
    comparison.to_csv(RESULTS_DIR / "model_comparison_target_recall.csv", index=False)

    recall_threshold = float(strategies.set_index("strategy").loc["min_fpr_at_target_recall", "threshold"])
    plot_roc_models(y_train, data["oof_all"], comparison)
    plot_roc_thresholds(y_test, test_scores, results)
    plot_fn_fp_threshold(points, strategies)
    plot_confusion_matrices(results)
    plot_cost_ratios(costs, recall_threshold)

    metadata = {
        "phase": "CRISP-DM 5 extension - business operating points",
        "final_model": winner,
        "score": "raw model score (uncalibrated), as for the phase 5 threshold and risk groups; decision = churn if score >= threshold",
        "threshold_selection_data": "training OOF predictions (single stratified 5-fold CV of phase 5); the held-out test set only evaluates the frozen thresholds",
        "oof_limitation": "hyperparameters were tuned earlier on the whole training set and thresholds are chosen on the same OOF predictions, so OOF-based numbers may be optimistic; the held-out test set gives the final evaluation of the frozen decisions",
        "target_recall": TARGET_RECALL,
        "target_recall_definition": "recall = TP / (TP + FN) >= target, i.e. FNR = FN / (TP + FN) <= 1 - target: share of the actual churners that are missed, not a share of all customers",
        "contact_capacity": CONTACT_CAPACITY,
        "candidate_thresholds": "every distinct OOF score plus 'contact nobody' (threshold = inf); tied scores switch together",
        "tie_rules": {
            "min_fpr_at_target_recall": "lowest FPR, then higher recall, then higher threshold",
            "min_cost": "lowest cost, then fewer contacted customers",
            "max_f1": "highest F1, then lowest threshold (phase 5)",
            "top_capacity": "stable sort of scores, first ceil(capacity * n) customers"
        },
        "cost_model": {
            "definition": "cost = FN * cost_FN + FP * cost_FP; cost ratios are assumptions, not operator data",
            "scenario_costs_usd_at_mean_training_monthly_charges": {name: {"cost_fn": fn, "cost_fp": fp, "ratio": fn / fp} for name, (fn, fp) in scenario_costs.items()},
            "link_to_phase_5": "cost_FN = simulated benefit of contacting a churner, cost_FP = incentive + contact cost for a non-churner (contact_net_benefit); with constant MonthlyCharges minimising the cost maximises the simulated net benefit; threshold_strategies_retention.csv evaluates every strategy with the per-customer simulation of phase 5",
            "interpretation": "hypothetical revenue-based scenario values; the model estimates churn propensity, not uplift; a prevented churn is an assumption of the scenario, not an observed effect"
        },
        "capacity_check": capacity,
        "consistency_checks": checks,
        "model_comparison": {
            "data": "OOF predictions of every tuned variant from 05d (same 5-fold split, hyperparameters tuned on the same training data, so possibly optimistic for all variants)",
            "best_at_target_recall_oof": str(comparison.iloc[0]["combo_id"]),
            "ap_winner": winner,
            "bootstrap": f"{BOOTSTRAP_ITERATIONS} paired resamples of training customers; FPR at the target recall re-selected in every resample; descriptive, conditional on the OOF scores, no multiple-testing correction; an interval containing 0 means no difference was detected, not that the models are equivalent",
            "test_columns": "*_frozen_threshold: the OOF threshold of each variant applied to the test set; descriptive only, the test set is not used to choose a model or a threshold"
        }
    }
    (RESULTS_DIR / "threshold_analysis_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    pd.set_option("display.width", 200)
    columns = ["split", "strategy", "threshold", "tp", "fp", "fn", "tn", "recall", "fnr", "fpr", "tnr", "precision", "f1", "contacted", "contact_rate", "meets_target_recall"]
    print("\nThreshold strategies - chosen on OOF, evaluated on OOF and test:\n")
    print(results[columns].round(4).to_string(index=False))
    print("\nTest bootstrap intervals of frozen strategies:\n")
    print(intervals.round(3).to_string(index=False))
    print(f"\nContact capacity check: {capacity}")
    print("\nCost ratios (cost_FP = 1):\n")
    print(costs.round(3).to_string(index=False))
    print("\nRetention simulation of the strategies, test set:\n")
    print(retention[["scenario", "strategy", "customers_contacted", "net_benefit", "random_net_benefit"]].round(0).to_string(index=False))
    print(f"\nModel comparison at recall >= {TARGET_RECALL} (OOF selection; test with frozen thresholds):\n")
    print(comparison[["combo_id", "cv_rank_by_average_precision", "oof_average_precision", "oof_roc_auc", "oof_threshold_at_target_recall", "oof_recall", "oof_fpr", "oof_fp", "oof_fn", "oof_contact_rate", "oof_fpr_diff_ci_low", "oof_fpr_diff_ci_high", "test_recall_frozen_threshold", "test_fpr_frozen_threshold"]].round(4).to_string(index=False))

    inputs = [PROCESSED_DIR / name for name in ("X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv")]
    inputs += [RESULTS_DIR / name for name in ("training_oof_predictions.csv", "test_predictions.csv", "final_model_metadata.json", "modeling_cv_summary.csv", "retention_scenarios.csv", "oof_scores_all_models.csv", "test_scores_all_models.csv")]
    outputs = [RESULTS_DIR / name for name in ("operating_points_oof.csv", "threshold_strategies.csv", "threshold_strategy_rules.csv", "threshold_cost_ratios.csv", "threshold_strategies_retention.csv", "model_comparison_target_recall.csv", "threshold_analysis_metadata.json")]
    parameters = {"target_recall": TARGET_RECALL, "contact_capacity": CONTACT_CAPACITY, "cost_ratios_fn_to_fp": list(COST_RATIOS_FN_TO_FP), "bootstrap_iterations": BOOTSTRAP_ITERATIONS, "random_state": RANDOM_STATE, "final_model": winner}
    run_id = record_stage("05e_threshold_analysis", inputs, outputs, parameters, upstream)
    print(f"\nSaved operating-point tables to results/ and figures to figures/ (run {run_id})\n")


if __name__ == "__main__":
    main()
