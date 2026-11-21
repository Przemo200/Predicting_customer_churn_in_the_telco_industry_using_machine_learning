"""
Phase 5 CRISP-DM - evaluation
"""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import (
    BOOTSTRAP_CI,
    BOOTSTRAP_ITERATIONS,
    CV_FOLDS,
    ID_COLUMN,
    MODEL_RANKING_COLUMNS,
    MODELS_DIR,
    OOF_RANDOM_STATE,
    PROCESSED_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    RETENTION_SCENARIOS,
    RETENTION_VALUE_MONTHS,
    RISK_HIGH_QUANTILE,
    SENSITIVITY_MARGINS,
    SENSITIVITY_REFERENCE_SCENARIO,
    SENSITIVITY_SUCCESS_RATES,
    TOP_K_FRACTIONS,
)
from src.evaluation import (
    assign_risk_group,
    benefit_kwargs,
    break_even_probability,
    break_even_success_rate,
    expected_contact_benefit,
    expected_value_policy,
    oof_probabilities,
    policy_outcome,
    retention_profit_curve,
    select_contact_fraction,
    select_threshold,
    threshold_metrics,
    top_fraction_mask,
)
from src.manifest import record_stage, require_stages
from src.plotting import (
    ACCENT_COLOR,
    MAIN_COLOR,
    REFERENCE_COLOR,
    SERIES_COLORS,
    finish_axes,
    number,
    percent_formatter,
    save_figure,
    set_style,
    thousands_formatter,
)

CONTACT_FRACTIONS = np.r_[0.0, np.linspace(0.01, 1.0, 100)]


def load_data():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").iloc[:, 0].astype(int)
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").iloc[:, 0].astype(int)
    id_test = pd.read_csv(PROCESSED_DIR / "customer_id_test.csv")[ID_COLUMN].astype(str)

    if not (len(X_test) == len(y_test) == len(id_test)):
        raise ValueError("Test features, target and IDs are not aligned")

    return X_train, X_test, y_train, y_test, id_test


def load_winner():
    summary = pd.read_csv(RESULTS_DIR / "modeling_cv_summary.csv")
    if summary.empty:
        raise ValueError("modeling_cv_summary.csv is empty")

    winner = summary.sort_values(MODEL_RANKING_COLUMNS, ascending=False).iloc[0]
    combo_id = str(winner["combo_id"])
    pipeline = joblib.load(MODELS_DIR / f"{combo_id}.joblib")

    return winner, combo_id, pipeline


def top_k_analysis(y_true: pd.Series, scores: np.ndarray) -> pd.DataFrame:
    order = np.argsort(-scores, kind="stable")
    frame = pd.DataFrame({"y_true": y_true.to_numpy()[order], "score": scores[order]})

    total_positives = int(frame["y_true"].sum())
    overall_rate = float(frame["y_true"].mean())
    rows = []

    for fraction in TOP_K_FRACTIONS:
        n = max(1, int(np.ceil(len(frame) * fraction)))
        subset = frame.head(n)
        positives = int(subset["y_true"].sum())
        segment_rate = float(subset["y_true"].mean())

        rows.append({
            "top_fraction": float(fraction),
            "customers": int(n),
            "positives_captured": positives,
            "capture_rate": float(positives / total_positives) if total_positives else np.nan,
            "segment_churn_rate": segment_rate,
            "overall_churn_rate": overall_rate,
            "lift": float(segment_rate / overall_rate) if overall_rate else np.nan
        })

    return pd.DataFrame(rows)


def bootstrap_confidence_intervals(y_true: pd.Series, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    y = y_true.to_numpy()
    n = len(y)
    top_n = int(np.ceil(n * 0.10))

    def compute(yb, sb):
        pred = (sb >= threshold).astype(int)
        top = np.argsort(-sb, kind="stable")[:top_n]
        return {
            "average_precision": average_precision_score(yb, sb),
            "roc_auc": roc_auc_score(yb, sb),
            "precision": precision_score(yb, pred, zero_division=0),
            "recall": recall_score(yb, pred, zero_division=0),
            "f1": f1_score(yb, pred, zero_division=0),
            "lift_top_10": yb[top].mean() / yb.mean()
        }

    point = compute(y, scores)
    records = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        records.append(compute(y[idx], scores[idx]))

    frame = pd.DataFrame(records)
    alpha = (1.0 - BOOTSTRAP_CI) / 2.0

    return pd.DataFrame({
        "metric": list(point.keys()),
        "estimate": list(point.values()),
        "ci_low": frame.quantile(alpha).to_numpy(),
        "ci_high": frame.quantile(1.0 - alpha).to_numpy(),
        "bootstrap_iterations": len(frame),
        "ci_level": BOOTSTRAP_CI
    })


def gains_curve(y_true: pd.Series, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="stable")
    y = y_true.to_numpy()[order]
    x = np.arange(1, len(y) + 1) / len(y)
    cumulative = np.cumsum(y) / y.sum()
    return np.r_[0.0, x], np.r_[0.0, cumulative]


def retention_analysis(y_train: pd.Series, oof_scores: np.ndarray, X_train: pd.DataFrame, y_test: pd.Series, test_scores: np.ndarray, test_probabilities: np.ndarray, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fractions = CONTACT_FRACTIONS
    summary_rows = []
    curve_frames = []
    mean_charges = float(X_train["MonthlyCharges"].mean())

    for name, params in RETENTION_SCENARIOS.items():
        train_curve = retention_profit_curve(y_train, oof_scores, X_train["MonthlyCharges"], params, fractions)
        best_idx = select_contact_fraction(train_curve)

        test_curve = retention_profit_curve(y_test, test_scores, X_test["MonthlyCharges"], params, fractions)
        test_curve["scenario"] = name
        curve_frames.append(test_curve)

        ev_customers, ev_net_benefit = expected_value_policy(y_test, test_probabilities, X_test["MonthlyCharges"], params)

        summary_rows.append({
            "scenario": name,
            "contact_cost": params["cost"],
            "retention_success_rate": params["success_rate"],
            "incentive_months": params["incentive_months"],
            "value_months": RETENTION_VALUE_MONTHS,
            "break_even_probability_at_mean_charges": break_even_probability(mean_charges, params),
            "best_fraction_selected_on_oof": float(fractions[best_idx]),
            "oof_net_benefit_at_best_fraction": float(train_curve.loc[best_idx, "net_benefit"]),
            "test_customers_contacted_at_best_fraction": int(test_curve.loc[best_idx, "customers"]),
            "test_net_benefit_at_best_fraction": float(test_curve.loc[best_idx, "net_benefit"]),
            "test_net_benefit_random_same_fraction": float(test_curve.loc[best_idx, "random_net_benefit"]),
            "test_net_benefit_contact_all": float(test_curve.loc[len(fractions) - 1, "net_benefit"]),
            "ev_policy_customers_contacted": ev_customers,
            "test_net_benefit_ev_policy": ev_net_benefit
        })

    return pd.DataFrame(summary_rows), pd.concat(curve_frames, ignore_index=True)


def retention_sensitivity(y_train: pd.Series, oof_scores: np.ndarray, train_charges: pd.Series, y_test: pd.Series, test_scores: np.ndarray, test_probabilities: np.ndarray, test_charges: pd.Series, fixed_fraction: float) -> pd.DataFrame:
    reference = RETENTION_SCENARIOS[SENSITIVITY_REFERENCE_SCENARIO]
    rows = []

    for margin in SENSITIVITY_MARGINS:
        for rate in SENSITIVITY_SUCCESS_RATES:
            params = {**reference, "success_rate": rate, "margin": margin}
            oof_curve = retention_profit_curve(y_train, oof_scores, train_charges, params, CONTACT_FRACTIONS)
            optimal_fraction = float(CONTACT_FRACTIONS[select_contact_fraction(oof_curve)])

            policies = [
                ("oof", "fixed_top_fraction", fixed_fraction, y_train, top_fraction_mask(oof_scores, fixed_fraction), train_charges),
                ("oof", "oof_optimal_top_fraction", optimal_fraction, y_train, top_fraction_mask(oof_scores, optimal_fraction), train_charges),
                ("test", "fixed_top_fraction", fixed_fraction, y_test, top_fraction_mask(test_scores, fixed_fraction), test_charges),
                ("test", "oof_optimal_top_fraction", optimal_fraction, y_test, top_fraction_mask(test_scores, optimal_fraction), test_charges),
                ("test", "expected_value", np.nan, y_test, expected_contact_benefit(np.asarray(test_probabilities), test_charges.to_numpy(dtype=float), **benefit_kwargs(params)) > 0, test_charges)
            ]
            for split, policy, fraction, y, contact, charges in policies:
                rows.append({
                    "split": split,
                    "policy": policy,
                    "margin": margin,
                    "success_rate": rate,
                    "fraction_selected_on_oof": fraction,
                    **policy_outcome(y, charges, contact, params)
                })

    return pd.DataFrame(rows)


def retention_break_even(y_train: pd.Series, oof_scores: np.ndarray, train_charges: pd.Series, y_test: pd.Series, test_scores: np.ndarray, test_charges: pd.Series, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    splits = (("oof", y_train, oof_scores, train_charges), ("test", y_test, test_scores, test_charges))

    for _, scenario in summary.iterrows():
        params = RETENTION_SCENARIOS[scenario["scenario"]]
        fraction = float(scenario["best_fraction_selected_on_oof"])

        for margin in SENSITIVITY_MARGINS:
            for split, y, scores, charges in splits:
                y_arr, charges_arr = np.asarray(y), np.asarray(charges, dtype=float)
                masks = (("top_fraction_selected_on_oof", fraction, top_fraction_mask(scores, fraction)), ("contact_all_or_random_selection", 1.0, np.ones(len(y_arr), dtype=bool)))

                for policy, policy_fraction, contact in masks:
                    rate, status = break_even_success_rate(y_arr[contact], charges_arr[contact], params["cost"], params["incentive_months"], margin)
                    outcome = policy_outcome(y_arr, charges_arr, contact, {**params, "margin": margin})
                    rows.append({
                        "scenario": scenario["scenario"],
                        "margin": margin,
                        "split": split,
                        "policy": policy,
                        "fraction": policy_fraction,
                        "customers_contacted": outcome["customers_contacted"],
                        "assumed_success_rate": params["success_rate"],
                        "net_benefit_at_assumed_rate": outcome["net_benefit"],
                        "break_even_success_rate": rate,
                        "break_even_status": status
                    })

    return pd.DataFrame(rows)


SCENARIO_LABELS = {"pessimistic": "Pesymistyczny", "base": "Bazowy", "optimistic": "Optymistyczny"}


def plot_retention_profit(curves: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for color, (name, frame) in zip(SERIES_COLORS, curves.groupby("scenario", sort=False), strict=False):
        ax.plot(frame["fraction"], frame["net_benefit"], color=color, label=SCENARIO_LABELS[name])
        best = summary.loc[summary["scenario"].eq(name)].iloc[0]
        ax.scatter(best["best_fraction_selected_on_oof"], best["test_net_benefit_at_best_fraction"], color=color, zorder=3, s=20)
    ax.axhline(0, linewidth=0.8, color=REFERENCE_COLOR)
    ax.scatter([], [], color=REFERENCE_COLOR, s=20, label="Odsetek wybrany na OOF")
    ax.set_xlabel("Odsetek kontaktowanych klientów")
    ax.set_ylabel("Symulowana korzyść netto [tys. USD]")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(percent_formatter())
    ax.yaxis.set_major_formatter(thousands_formatter())
    ax.legend()
    finish_axes(ax)
    save_figure(fig, "fig_retention_profit.pdf")


def plot_retention_sensitivity(sensitivity: pd.DataFrame) -> None:
    data = sensitivity.loc[sensitivity["split"].eq("test") & sensitivity["policy"].eq("fixed_top_fraction")]
    fig, ax = plt.subplots(figsize=(5.8, 4.0))

    for color, (margin, frame) in zip(SERIES_COLORS, data.groupby("margin", sort=False), strict=False):
        label = "przychód (marża 100%)" if margin == 1.0 else f"marża {100 * margin:g}%"
        ax.plot(frame["success_rate"], frame["net_benefit"], color=color, marker="o", markersize=3, label=f"Model: {label}")
        ax.plot(frame["success_rate"], frame["random_net_benefit"], color=color, linestyle="--", linewidth=1)

    ax.plot([], [], color=REFERENCE_COLOR, linestyle="--", linewidth=1, label="Wybór losowy tej samej liczby klientów")
    ax.axhline(0, linewidth=0.8, color=REFERENCE_COLOR)
    reference_rate = RETENTION_SCENARIOS[SENSITIVITY_REFERENCE_SCENARIO]["success_rate"]
    ax.axvline(reference_rate, color=REFERENCE_COLOR, linestyle=":", linewidth=1, label=f"Założenie scenariusza bazowego ({100 * reference_rate:g}%)")
    ax.set_xlabel("Założona skuteczność działań retencyjnych")
    ax.set_ylabel("Symulowana korzyść netto [tys. USD]")
    ax.xaxis.set_major_formatter(percent_formatter())
    ax.yaxis.set_major_formatter(thousands_formatter())
    ax.legend(loc="upper left")
    finish_axes(ax)
    save_figure(fig, "fig_retention_sensitivity.pdf")


def plot_pr_curve(y_true: pd.Series, scores: np.ndarray) -> None:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.plot(recall, precision, color=MAIN_COLOR, drawstyle="steps-post", label=f"Model (AP = {number(ap)})")
    ax.axhline(y_true.mean(), color=REFERENCE_COLOR, linestyle="--", linewidth=1, label=f"Klasyfikator losowy ({number(y_true.mean())})")
    ax.set_xlabel("Czułość")
    ax.set_ylabel("Precyzja")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right")
    finish_axes(ax)
    save_figure(fig, "fig_test_pr_curve.pdf")


def plot_roc_curve(y_true: pd.Series, scores: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.plot(fpr, tpr, color=MAIN_COLOR, label=f"Model (AUC = {number(auc)})")
    ax.plot([0, 1], [0, 1], color=REFERENCE_COLOR, linestyle="--", linewidth=1, label="Klasyfikator losowy")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    finish_axes(ax)
    save_figure(fig, "fig_test_roc_curve.pdf")


def plot_calibration(y_true: pd.Series, raw_scores: np.ndarray, calibrated: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.0))

    for values, label, color in ((raw_scores, "Przed kalibracją", MAIN_COLOR), (calibrated, "Po kalibracji izotonicznej", ACCENT_COLOR)):
        observed, predicted = calibration_curve(y_true, values, n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, marker="o", markersize=3.5, color=color, label=label)

    ax.plot([0, 1], [0, 1], color=REFERENCE_COLOR, linestyle="--", linewidth=1, label="Idealna kalibracja")
    ax.set_xlabel("Przewidywane prawdopodobieństwo")
    ax.set_ylabel("Zaobserwowany odsetek rezygnacji")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    finish_axes(ax)
    save_figure(fig, "fig_test_calibration.pdf")


def plot_threshold_analysis(threshold_table: pd.DataFrame, selected_threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for column, label, color in (("precision", "Precyzja", SERIES_COLORS[0]), ("recall", "Czułość", SERIES_COLORS[1]), ("f1", "F1", SERIES_COLORS[2])):
        ax.plot(threshold_table["threshold"], threshold_table[column], color=color, label=label)
    ax.axvline(selected_threshold, color=REFERENCE_COLOR, linestyle="--", linewidth=1, label=f"Wybrany próg ({number(selected_threshold)})")
    ax.set_xlabel("Próg decyzyjny")
    ax.set_ylabel("Wartość miary")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="center right", bbox_to_anchor=(1.03, 0.58))
    finish_axes(ax)
    save_figure(fig, "fig_threshold_analysis.pdf")


def plot_confusion(y_true: pd.Series, pred: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        pred,
        labels=[0, 1],
        display_labels=["Nie", "Tak"],
        values_format="d",
        cmap="Blues",
        ax=ax,
        colorbar=False
    )
    ax.set_xlabel("Przewidziany churn")
    ax.set_ylabel("Rzeczywisty churn")
    save_figure(fig, "fig_test_confusion_matrix.pdf")


def plot_gains(y_true: pd.Series, scores: np.ndarray, top_k: pd.DataFrame) -> None:
    x, cumulative = gains_curve(y_true, scores)

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.plot(x, cumulative, color=MAIN_COLOR, label="Model")
    ax.scatter(top_k["top_fraction"], top_k["capture_rate"], color=MAIN_COLOR, s=16, zorder=3)
    ax.plot([0, 1], [0, 1], color=REFERENCE_COLOR, linestyle="--", linewidth=1, label="Wybór losowy")
    ax.set_xlabel("Odsetek kontaktowanych klientów")
    ax.set_ylabel("Odsetek wychwyconych rezygnacji")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(percent_formatter())
    ax.yaxis.set_major_formatter(percent_formatter())
    ax.legend(loc="lower right")
    finish_axes(ax)
    save_figure(fig, "fig_cumulative_gains.pdf")


def main() -> None:
    set_style()
    upstream = require_stages(["03_data_preparation", "04_modeling"])
    X_train, X_test, y_train, y_test, id_test = load_data()
    winner, combo_id, pipeline = load_winner()

    print("\nPhase 5 CRISP-DM - evaluation\n")
    print(f"Upstream runs {upstream}")
    print(f"Winner from phase 4: {combo_id}")
    print("Generating out-of-fold probabilities on training data (single stratified 5-fold CV)...")

    oof_scores, oof_folds = oof_probabilities(pipeline, X_train, y_train)
    selected_threshold, threshold_table = select_threshold(y_train, oof_scores)

    risk_low_cut = float(selected_threshold)
    risk_high_cut = float(max(np.quantile(oof_scores, RISK_HIGH_QUANTILE), selected_threshold))

    calibrator = IsotonicRegression(out_of_bounds="clip").fit(oof_scores, y_train)
    joblib.dump(calibrator, MODELS_DIR / "final_calibrator.joblib")

    print(f"Selected threshold (max F1 on OOF): {selected_threshold:.4f}")
    print(f"Risk cutoffs from OOF scores: low/medium={risk_low_cut:.4f}, medium/high={risk_high_cut:.4f}")
    print("Evaluating the frozen pipeline on the held-out test set...\n")

    test_scores = pipeline.predict_proba(X_test)[:, 1]
    test_probabilities = calibrator.predict(test_scores)
    calibration = {
        "brier_raw": float(brier_score_loss(y_test, test_scores)),
        "brier_isotonic": float(brier_score_loss(y_test, test_probabilities)),
        "brier_constant_prevalence": float(brier_score_loss(y_test, np.full(len(y_test), y_train.mean())))
    }
    pred_default = (test_scores >= 0.5).astype(int)
    pred_selected = (test_scores >= selected_threshold).astype(int)

    threshold_rows = [threshold_metrics(y_test, test_scores, 0.5, "default_0.5"), threshold_metrics(y_test, test_scores, selected_threshold, "selected_oof_f1")]
    metrics = pd.DataFrame(threshold_rows)
    metrics["roc_auc"] = float(roc_auc_score(y_test, test_scores))
    metrics["average_precision"] = float(average_precision_score(y_test, test_scores))
    metrics.to_csv(RESULTS_DIR / "evaluation_metrics.csv", index=False)

    threshold_table.to_csv(RESULTS_DIR / "threshold_selection_oof.csv", index=False)

    oof = pd.DataFrame({"y_true": y_train.to_numpy(), "oof_score": oof_scores, "oof_fold": oof_folds})
    oof.to_csv(RESULTS_DIR / "training_oof_predictions.csv", index=False)

    predictions = pd.DataFrame({
        "row_index": np.arange(len(X_test)),
        ID_COLUMN: id_test.to_numpy(),
        "y_true": y_test.to_numpy(),
        "churn_score": test_scores,
        "churn_probability_calibrated": test_probabilities,
        "prediction_0_5": pred_default,
        "prediction_selected": pred_selected,
        "risk_group": assign_risk_group(test_scores, risk_low_cut, risk_high_cut)
    })
    predictions.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    top_k = top_k_analysis(y_test, test_scores)
    top_k.to_csv(RESULTS_DIR / "top_k_business_analysis.csv", index=False)

    confidence = bootstrap_confidence_intervals(y_test, test_scores, selected_threshold)
    confidence.to_csv(RESULTS_DIR / "evaluation_confidence_intervals.csv", index=False)

    retention_summary, retention_curves = retention_analysis(y_train, oof_scores, X_train, y_test, test_scores, test_probabilities, X_test)
    retention_summary.to_csv(RESULTS_DIR / "retention_scenarios.csv", index=False)
    retention_curves.to_csv(RESULTS_DIR / "retention_profit_curves.csv", index=False)
    plot_retention_profit(retention_curves, retention_summary)

    fixed_fraction = float(retention_summary.set_index("scenario").loc[SENSITIVITY_REFERENCE_SCENARIO, "best_fraction_selected_on_oof"])
    sensitivity = retention_sensitivity(y_train, oof_scores, X_train["MonthlyCharges"], y_test, test_scores, test_probabilities, X_test["MonthlyCharges"], fixed_fraction)
    sensitivity.to_csv(RESULTS_DIR / "retention_sensitivity.csv", index=False)
    break_even = retention_break_even(y_train, oof_scores, X_train["MonthlyCharges"], y_test, test_scores, X_test["MonthlyCharges"], retention_summary)
    break_even.to_csv(RESULTS_DIR / "retention_break_even.csv", index=False)
    plot_retention_sensitivity(sensitivity)

    final_metadata = {
        "phase": "CRISP-DM 5 - evaluation",
        "winner_combo_id": combo_id,
        "winner_model": str(winner["model"]),
        "winner_balancing": str(winner["balancing"]),
        "selection_metric_phase_4": "Average Precision",
        "selected_threshold": selected_threshold,
        "threshold_selection_data": "out-of-fold predictions from a single stratified 5-fold CV on training data, one prediction per customer (seed differs from the hyperparameter search)",
        "oof_limitation": "hyperparameters were tuned on the same training data, so OOF-based quantities may be slightly optimistic; the held-out test set is the independent check",
        "threshold_objective": "f1",
        "default_threshold_also_reported": 0.5,
        "test_rows": len(X_test),
        "test_prevalence": float(y_test.mean()),
        "test_average_precision": float(average_precision_score(y_test, test_scores)),
        "test_roc_auc": float(roc_auc_score(y_test, test_scores)),
        "risk_group_cutoffs": {
            "low_to_medium": risk_low_cut,
            "medium_to_high": risk_high_cut,
            "derived_from": "low cut = selected threshold; high cut = max(training OOF score quantile, selected threshold)",
            "high_quantile": RISK_HIGH_QUANTILE
        },
        "calibration": {"method": "isotonic regression fitted on single 5-fold training OOF scores", "file": "models/final_calibrator.joblib", **calibration},
        "retention_value": f"{RETENTION_VALUE_MONTHS} * MonthlyCharges (revenue, not margin); net benefit is a simulation under assumed scenario parameters",
        "retention_policies": {
            "contact_fractions": "0 (no campaign) and 1%, 2%, ..., 100% of customers ranked by raw score",
            "fraction_selection": "argmax of the simulated net benefit on training OOF scores, ties to the smaller fraction; the test set only evaluates the frozen fraction",
            "sensitivity": {"reference_scenario": SENSITIVITY_REFERENCE_SCENARIO, "success_rates": list(SENSITIVITY_SUCCESS_RATES), "margins": list(SENSITIVITY_MARGINS), "margin_note": "margin 1.0 = revenue; lower values are illustrative, not the operator's margin"},
            "interpretation": "the model estimates churn propensity, not uplift; success_rate is an assumed campaign effect on churners and no effect on non-churners is assumed"
        },
        "input_columns": X_train.columns.tolist()
    }
    (RESULTS_DIR / "final_model_metadata.json").write_text(json.dumps(final_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    joblib.dump(pipeline, MODELS_DIR / "final_churn_pipeline.joblib")

    plot_pr_curve(y_test, test_scores)
    plot_roc_curve(y_test, test_scores)
    plot_threshold_analysis(threshold_table, selected_threshold)
    plot_confusion(y_test, pred_selected)
    plot_gains(y_test, test_scores, top_k)
    plot_calibration(y_test, test_scores, test_probabilities)

    print("Test metrics:\n")
    print(metrics.to_string(index=False))
    print("\nBootstrap confidence intervals, held-out test set, threshold from OOF:\n")
    print(confidence.round(3).to_string(index=False))
    print("\nCalibration - Brier score, lower is better -", {k: round(v, 4) for k, v in calibration.items()})
    print("\nTop-k business analysis:\n")
    print(top_k.to_string(index=False))
    print("\nRetention scenarios (assumption-based, see config.py):\n")
    print(retention_summary.round(3).to_string(index=False))
    print("\nBreak-even campaign success rate of the frozen contact lists (test set):\n")
    print(break_even.loc[break_even["split"].eq("test")].round(3).to_string(index=False))

    outputs = [
        MODELS_DIR / "final_calibrator.joblib",
        MODELS_DIR / "final_churn_pipeline.joblib",
        *(RESULTS_DIR / name for name in (
            "evaluation_metrics.csv", "threshold_selection_oof.csv", "training_oof_predictions.csv", "test_predictions.csv",
            "top_k_business_analysis.csv", "evaluation_confidence_intervals.csv", "retention_scenarios.csv", "retention_profit_curves.csv",
            "retention_sensitivity.csv", "retention_break_even.csv", "final_model_metadata.json"
        ))
    ]
    inputs = [PROCESSED_DIR / name for name in ("X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv", "customer_id_test.csv")]
    inputs += [RESULTS_DIR / "modeling_cv_summary.csv", MODELS_DIR / f"{combo_id}.joblib"]
    parameters = {
        "winner_combo_id": combo_id,
        "oof_random_state": OOF_RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_ci": BOOTSTRAP_CI,
        "risk_high_quantile": RISK_HIGH_QUANTILE,
        "selected_threshold": selected_threshold,
        "retention_scenarios": RETENTION_SCENARIOS,
        "sensitivity_success_rates": list(SENSITIVITY_SUCCESS_RATES),
        "sensitivity_margins": list(SENSITIVITY_MARGINS)
    }
    run_id = record_stage("05_evaluation", inputs, outputs, parameters, upstream)
    print(f"\nSaved final evaluated pipeline and phase 5 outputs (run {run_id})")


if __name__ == "__main__":
    main()