"""
Interpretability / XAI of the final model
"""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.colors import to_hex
from matplotlib.ticker import FixedLocator, NullLocator
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from config import (
    CV_FOLDS,
    CV_REPEATS,
    IMPORTANCE_MAX_DISPLAY,
    MODEL_RANKING_COLUMNS,
    MODELS_DIR,
    PERMUTATION_REPEATS,
    PROCESSED_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    SHAP_BACKGROUND_SAMPLES,
    SHAP_MAX_DISPLAY,
    SHAP_MAX_SAMPLES,
)
from src.explainability import (
    aggregate_local_shap,
    aggregate_shap_by_original,
    display_frame,
    pretty_feature_name,
    transform_for_model,
    variance_inflation_factors,
)
from src.manifest import record_stage, require_stages
from src.plotting import (
    ACCENT_COLOR,
    ACCENT_FILL_COLOR,
    MAIN_COLOR,
    REFERENCE_COLOR,
    comma_formatter,
    commas_in_figure,
    finish_axes,
    save_figure,
    set_style,
)
from src.preprocessing import get_feature_lists

INTERPRETABLE_LR_EXCLUDED = ["MonthlyCharges", "TotalCharges"]

SHAP_SCALES = {
    "XGBClassifier": ("raw margin = log-odds of churn (XGBClassifier.predict(output_margin=True)), before isotonic calibration", "log-szanse"),
    "LogisticRegression": ("decision function = log-odds of churn (LogisticRegression.decision_function), before isotonic calibration", "log-szanse"),
    "RandomForestClassifier": ("raw predicted probability of churn (RandomForestClassifier.predict_proba, mean of tree leaf class shares), before isotonic calibration", "surowe prawdopodobieństwo")
}
ADDITIVITY_TOLERANCE = 1e-4


def float32_floor(thresholds: np.ndarray) -> np.ndarray:
    rounded = thresholds.astype(np.float32)
    too_high = rounded.astype(np.float64) > thresholds
    rounded[too_high] = np.nextafter(rounded[too_high], np.float32(-np.inf))
    return rounded.astype(np.float64)


def build_explainer(model, background: pd.DataFrame):
    if type(model).__name__ == "RandomForestClassifier":
        background = background.astype(np.float32).astype(np.float64)
    explainer = shap.Explainer(model, shap.maskers.Independent(background, max_samples=len(background)))
    if type(model).__name__ == "RandomForestClassifier":
        explainer.model.thresholds = float32_floor(explainer.model.thresholds)
    return explainer


def explained_output(model, X_transformed: pd.DataFrame) -> np.ndarray:
    name = type(model).__name__
    if name == "XGBClassifier":
        return model.predict(X_transformed, output_margin=True)
    if name == "LogisticRegression":
        return model.decision_function(X_transformed)
    return model.predict_proba(X_transformed)[:, 1]


def additivity_error(explanation: shap.Explanation, model, X_transformed: pd.DataFrame) -> float:
    reconstructed = explanation.values.sum(axis=1) + np.asarray(explanation.base_values, dtype=float)
    return float(np.max(np.abs(reconstructed - explained_output(model, X_transformed))))


def load_data():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").iloc[:, 0].astype(int)
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").iloc[:, 0].astype(int)
    predictions = pd.read_csv(RESULTS_DIR / "test_predictions.csv")
    metadata = json.loads((RESULTS_DIR / "final_model_metadata.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(RESULTS_DIR / "modeling_cv_summary.csv")
    return X_train, y_train, X_test, y_test, predictions, metadata, summary


def plot_importance(frame: pd.DataFrame, value_col: str, label_col: str, xlabel: str, filename: str, error_col: str | None = None, note: str | None = None, bar_color: str = MAIN_COLOR, error_color: str = ACCENT_COLOR) -> None:
    data = frame.head(IMPORTANCE_MAX_DISPLAY).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 0.3 * len(data) + 1.0))
    xerr = data[error_col] if error_col else None
    ax.barh(data[label_col], data[value_col], xerr=xerr, height=0.65, color=bar_color, error_kw={"elinewidth": 1.0, "ecolor": error_color, "capsize": 2, "capthick": 1.0})
    ax.set_xlabel(xlabel)
    low = float((data[value_col] - (data[error_col] if error_col else 0)).min())
    ax.set_xlim(left=min(0.0, low * 1.1))
    if low < 0:
        ax.axvline(0, color=REFERENCE_COLOR, linewidth=0.8)
    if note:
        ax.text(0.98, 0.03, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=REFERENCE_COLOR)
    finish_axes(ax)
    save_figure(fig, filename)


def months_to_years(values):
    return values / 12.0


def tenure_years_name(*_):
    return np.array(["tenure_years"])


def interpretable_design(X: pd.DataFrame) -> ColumnTransformer:
    X_reduced = X.drop(columns=INTERPRETABLE_LR_EXCLUDED)
    _, categorical = get_feature_lists(X_reduced)
    return ColumnTransformer(
        transformers=[
            ("tenure_years", FunctionTransformer(months_to_years, feature_names_out=tenure_years_name), ["tenure"]),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical)
        ],
        verbose_feature_names_out=False
    )


def wald_table(model: LogisticRegression, design: np.ndarray, names: list[str]) -> pd.DataFrame:
    X = np.column_stack([np.ones(len(design)), design])
    beta = np.r_[model.intercept_, model.coef_[0]]
    p = 1.0 / (1.0 + np.exp(-X @ beta))
    covariance = np.linalg.inv(X.T @ (X * (p * (1.0 - p))[:, None]))
    se = np.sqrt(np.diag(covariance))
    z = stats.norm.ppf(0.975)

    return pd.DataFrame({
        "feature": ["intercept"] + names,
        "coefficient": beta,
        "std_error": se,
        "odds_ratio": np.exp(beta),
        "or_ci_low": np.exp(beta - z * se),
        "or_ci_high": np.exp(beta + z * se),
        "p_value_wald": 2.0 * stats.norm.sf(np.abs(beta / se))
    })


def collinearity_table(X_train: pd.DataFrame, interpretable: ColumnTransformer) -> pd.DataFrame:
    numeric, categorical = get_feature_lists(X_train)
    full = ColumnTransformer(transformers=[("num", "passthrough", numeric), ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical)], verbose_feature_names_out=False).set_output(transform="pandas")
    vif_full = variance_inflation_factors(full.fit_transform(X_train)).rename(columns={"vif": "vif_all_features"})
    reduced = interpretable.set_output(transform="pandas").fit_transform(X_train).rename(columns={"tenure_years": "tenure"})
    vif_reduced = variance_inflation_factors(reduced).rename(columns={"vif": "vif_interpretable_lr"})
    return vif_full.merge(vif_reduced, on="feature", how="left")


def plot_odds_ratios(table: pd.DataFrame, original_features: list[str]) -> None:
    data = table.loc[table["feature"].ne("intercept")].sort_values("odds_ratio")
    fig, ax = plt.subplots(figsize=(5.6, 0.24 * len(data) + 1.0))
    y_pos = np.arange(len(data))
    ax.errorbar(data["odds_ratio"], y_pos, xerr=[data["odds_ratio"] - data["or_ci_low"], data["or_ci_high"] - data["odds_ratio"]], fmt="o", color=MAIN_COLOR, ecolor=MAIN_COLOR, elinewidth=1, capsize=2, markersize=3.5)
    ax.axvline(1.0, color=REFERENCE_COLOR, linewidth=1, linestyle="--")
    ax.set_xscale("log")
    low, high = data["or_ci_low"].min() / 1.1, data["or_ci_high"].max() * 1.1
    ax.set_xlim(low, high)
    ticks = [2.0 ** k for k in range(int(np.floor(np.log2(low))), int(np.ceil(np.log2(high))) + 1) if low <= 2.0 ** k <= high]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(comma_formatter())
    ax.set_yticks(y_pos)
    ax.set_yticklabels([pretty_feature_name(name, original_features) for name in data["feature"]])
    ax.set_ylim(-0.7, len(data) - 0.3)
    ax.set_xlabel("Iloraz szans (skala logarytmiczna)", labelpad=6)
    finish_axes(ax)
    save_figure(fig, "fig_lr_odds_ratios.pdf")


def analyze_interpretable_logistic_regression(summary: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    preprocessor = interpretable_design(X_train)
    model = LogisticRegression(C=np.inf, max_iter=5000)
    pipeline = Pipeline([("design", preprocessor), ("model", model)])

    cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
    cv_ap = cross_val_score(pipeline, X_train, y_train, scoring="average_precision", cv=cv, n_jobs=-1)
    best_lr = summary.loc[summary["model"].eq("Logistic Regression")].sort_values(MODEL_RANKING_COLUMNS, ascending=False).iloc[0]

    pipeline.fit(X_train, y_train)
    design = pipeline.named_steps["design"].transform(X_train)
    names = pipeline.named_steps["design"].get_feature_names_out().tolist()
    table = wald_table(pipeline.named_steps["model"], np.asarray(design, dtype=float), names)
    table.to_csv(RESULTS_DIR / "interpretable_lr_odds_ratios.csv", index=False)

    vif = collinearity_table(X_train, interpretable_design(X_train))
    vif.to_csv(RESULTS_DIR / "interpretability_vif.csv", index=False)

    references = {
        feature: str(levels[0])
        for feature, levels in zip(
            pipeline.named_steps["design"].named_transformers_["cat"].feature_names_in_,
            pipeline.named_steps["design"].named_transformers_["cat"].categories_,
            strict=True
        )
    }
    lr_metadata = {
        "purpose": "transparent reference model for interpreting directions and sizes of effects; not the predictive model",
        "fitted_on": "training set",
        "excluded_features": INTERPRETABLE_LR_EXCLUDED,
        "exclusion_reason": "MonthlyCharges is almost fully determined by the service columns and TotalCharges by tenure * MonthlyCharges (see interpretability_vif.csv); keeping them makes single coefficients unstable and not interpretable",
        "tenure_unit": "years (tenure / 12)",
        "reference_categories": references,
        "penalty": "none (C = inf, maximum likelihood)",
        "confidence_intervals": "95% Wald, odds ratio scale; asymptotic, descriptive (no multiple-testing correction)",
        "cv_average_precision_mean": float(cv_ap.mean()),
        "cv_average_precision_std": float(cv_ap.std(ddof=1)),
        "best_tuned_lr_combo_id": str(best_lr["combo_id"]),
        "best_tuned_lr_cv_average_precision": float(best_lr["cv_average_precision"]),
        "interpretation_note": "odds ratios are conditional associations given the other features, not causal effects"
    }
    (RESULTS_DIR / "interpretable_lr_metadata.json").write_text(json.dumps(lr_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_odds_ratios(table, X_train.columns.tolist())

    print(f"Interpretable LR - CV AP {cv_ap.mean():.4f} (best tuned LR {best_lr['cv_average_precision']:.4f})")
    print("VIF, all features vs interpretable LR (top 6):")
    print(vif.head(6).round(2).to_string(index=False))


def analyze_permutation_importance(X_test: pd.DataFrame, y_test: pd.Series) -> None:
    pipeline = joblib.load(MODELS_DIR / "final_churn_pipeline.joblib")
    result = permutation_importance(pipeline, X_test, y_test, scoring="average_precision", n_repeats=PERMUTATION_REPEATS, random_state=RANDOM_STATE, n_jobs=1)

    frame = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    frame.to_csv(RESULTS_DIR / "permutation_importance_original_features.csv", index=False)

    plot_permutation_importance(frame)


def plot_permutation_importance(frame: pd.DataFrame) -> None:
    note = f"Słupki błędów (odchylenie standardowe, {PERMUTATION_REPEATS} powtórzeń)"
    plot_importance(frame, "importance_mean", "feature", "Spadek AP po permutacji cechy", "fig_permutation_importance.pdf", error_col="importance_std", note=note, bar_color=ACCENT_FILL_COLOR, error_color=MAIN_COLOR)


def positive_class_explanation(explanation: shap.Explanation) -> shap.Explanation:
    if explanation.values.ndim == 2:
        return explanation
    return explanation[:, :, 1]


def with_display_data(explanation: shap.Explanation, shown: pd.DataFrame, original_features: list[str]) -> shap.Explanation:
    names = [pretty_feature_name(name, original_features) for name in shown.columns]
    return shap.Explanation(values=explanation.values, base_values=explanation.base_values, data=shown.to_numpy(), feature_names=names)


def move_overflowing_waterfall_labels(fig) -> None:
    ax = fig.axes[0]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bars = [(patch, patch.get_window_extent(renderer)) for patch in ax.patches]

    for text in ax.texts:
        if to_hex(text.get_color()) != "#ffffff":
            continue
        box = text.get_window_extent(renderer)
        y_center = (box.y0 + box.y1) / 2
        x_center = (box.x0 + box.x1) / 2
        candidates = [(patch, bb) for patch, bb in bars if bb.y0 <= y_center <= bb.y1 and bb.x0 <= x_center <= bb.x1]
        if not candidates:
            continue
        patch, bb = min(candidates, key=lambda item: item[1].width)
        if box.width <= 0.75 * bb.width:
            continue
        positive = not text.get_text().startswith(("-", "\u2212"))
        x_display = bb.x1 + 4 if positive else bb.x0 - 4
        x_data = ax.transData.inverted().transform((x_display, y_center))[0]
        text.set_x(x_data)
        text.set_horizontalalignment("left" if positive else "right")
        text.set_color(patch.get_facecolor())


def translate_shap_axes(fig, xlabel: str, has_colorbar: bool) -> None:
    ax = fig.axes[0]
    ax.set_xlabel(xlabel, labelpad=4 if has_colorbar else 22, fontsize=11)
    labels = [label.get_text().replace("other features", "pozostałych cech").replace("Sum of", "Suma") for label in ax.get_yticklabels()]
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(labels)
    if has_colorbar:
        colorbar = fig.axes[-1]
        colorbar.set_ylabel("Wartość cechy", fontsize=11, labelpad=2)
        colorbar.set_yticks(colorbar.get_yticks())
        ticks = colorbar.set_yticklabels([label.get_text().replace("High", "Wysoka").replace("Low", "Niska") for label in colorbar.get_yticklabels()], fontsize=10)
        ticks[0].set_verticalalignment("bottom")
        ticks[-1].set_verticalalignment("top")
        colorbar.tick_params(pad=3)
    commas_in_figure(fig)


def choose_local_cases(predictions: pd.DataFrame) -> pd.DataFrame:
    pred = predictions["prediction_selected"].astype(int)
    truth = predictions["y_true"].astype(int)

    groups = {
        "TP": predictions[(truth == 1) & (pred == 1)],
        "FP": predictions[(truth == 0) & (pred == 1)],
        "FN": predictions[(truth == 1) & (pred == 0)],
        "TN": predictions[(truth == 0) & (pred == 0)]
    }

    rows = []
    for case_type, frame in groups.items():
        if frame.empty:
            continue
        median_score = frame["churn_score"].median()
        row = frame.loc[(frame["churn_score"] - median_score).abs().idxmin()].copy()
        row["case_type"] = case_type
        rows.append(row)

    return pd.DataFrame(rows)


def compute_shap(X_train: pd.DataFrame, X_test: pd.DataFrame) -> dict:
    pipeline = joblib.load(MODELS_DIR / "final_churn_pipeline.joblib")
    X_train_t, model = transform_for_model(pipeline, X_train)
    X_test_t, _ = transform_for_model(pipeline, X_test)
    model_name = type(model).__name__
    units_en, units_pl = SHAP_SCALES.get(model_name, ("raw model output, before isotonic calibration", "surowy wynik modelu"))

    rng = np.random.default_rng(RANDOM_STATE)
    background_n = min(SHAP_BACKGROUND_SAMPLES, len(X_train_t))
    background = X_train_t.iloc[rng.choice(len(X_train_t), size=background_n, replace=False)]
    sample_n = min(SHAP_MAX_SAMPLES, len(X_test_t))
    sample_idx = rng.choice(len(X_test_t), size=sample_n, replace=False)

    print(f"Building SHAP explainer using {background_n} background rows...")
    explainer = build_explainer(model, background)
    explanation = positive_class_explanation(explainer(X_test_t.iloc[sample_idx]))
    max_additivity_error = additivity_error(explanation, model, X_test_t.iloc[sample_idx])
    print(f"SHAP additivity check against the {units_en}: max |error| = {max_additivity_error:.2e}")
    if max_additivity_error > ADDITIVITY_TOLERANCE:
        raise RuntimeError(f"SHAP values do not add up to the explained model output within {ADDITIVITY_TOLERANCE:g} (max |error| = {max_additivity_error:.2e}); the explanations would not describe the model")

    return {
        "explainer": explainer,
        "explanation": explanation,
        "X_test_t": X_test_t,
        "sample_idx": sample_idx,
        "sample_n": sample_n,
        "background_n": background_n,
        "units_en": units_en,
        "units_pl": units_pl,
        "additivity_error": max_additivity_error
    }


def plot_shap_figures(shap_run: dict, aggregated: pd.DataFrame, X_test: pd.DataFrame, local_cases: pd.DataFrame) -> None:
    explainer, explanation, X_test_t, units_pl = shap_run["explainer"], shap_run["explanation"], shap_run["X_test_t"], shap_run["units_pl"]
    X_test_shown = display_frame(X_test_t, X_test)

    plot_importance(aggregated, "mean_abs_shap", "original_feature", f"Średnia wartość bezwzględna SHAP [{units_pl}]", "fig_shap_global_original.pdf")

    shap.plots.beeswarm(with_display_data(explanation, X_test_shown.iloc[shap_run["sample_idx"]], X_test.columns.tolist()), max_display=SHAP_MAX_DISPLAY, show=False)
    fig = plt.gcf()
    translate_shap_axes(fig, f"SHAP [{units_pl}]", has_colorbar=True)
    save_figure(fig, "fig_shap_beeswarm.pdf")

    for _, case in local_cases.iterrows():
        row_index = int(case["row_index"])
        local = positive_class_explanation(explainer(X_test_t.iloc[[row_index]]))[0]
        values, data, names = aggregate_local_shap(local.values, X_test_t.columns.tolist(), X_test.iloc[row_index])
        shap.plots.waterfall(shap.Explanation(values=values, base_values=float(local.base_values), data=data, feature_names=names), max_display=10, show=False)
        fig = plt.gcf()
        translate_shap_axes(fig, f"SHAP [{units_pl}]", has_colorbar=False)
        move_overflowing_waterfall_labels(fig)
        save_figure(fig, f"fig_shap_local_{case['case_type'].lower()}.pdf")


def run_shap(X_train: pd.DataFrame, X_test: pd.DataFrame, predictions: pd.DataFrame, metadata: dict) -> None:
    shap_run = compute_shap(X_train, X_test)
    explanation, X_test_t = shap_run["explanation"], shap_run["X_test_t"]
    units_en, sample_n, background_n = shap_run["units_en"], shap_run["sample_n"], shap_run["background_n"]
    max_additivity_error, tolerance = shap_run["additivity_error"], ADDITIVITY_TOLERANCE

    encoded = pd.DataFrame({"feature": X_test_t.columns, "mean_abs_shap": np.abs(explanation.values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    encoded.to_csv(RESULTS_DIR / "shap_global_encoded.csv", index=False)

    aggregated = aggregate_shap_by_original(X_test_t.columns.tolist(), explanation.values, X_test.columns.tolist())
    aggregated.to_csv(RESULTS_DIR / "shap_global_original_features.csv", index=False)

    local_cases = choose_local_cases(predictions)
    local_cases.to_csv(RESULTS_DIR / "shap_local_cases.csv", index=False)

    plot_shap_figures(shap_run, aggregated, X_test, local_cases)

    shap_metadata = {
        "final_model": metadata["winner_combo_id"],
        "global_sample_rows": int(sample_n),
        "background_rows": int(background_n),
        "local_case_types": local_cases["case_type"].tolist(),
        "local_case_selection": "median-score case within each of TP, FP, FN, TN",
        "shap_output_units": units_en,
        "additivity_max_abs_error": max_additivity_error,
        "additivity_tolerance": tolerance,
        "calibration_note": "SHAP explains the raw model output used for the decision threshold, risk groups and ranking, not the isotonic-calibrated probability used in the business calculations; isotonic calibration is monotone, so the customer ranking is preserved (up to ties) but contribution sizes are not on the calibrated scale; for class-weighted or SMOTENC models the raw probability is shifted upwards relative to the churn rate",
        "base_value": float(np.mean(np.asarray(explanation.base_values))),
        "plot_values": "numeric features shown in original units; SHAP values computed on the model input (standardized numeric features)",
        "local_plot_aggregation": "in local waterfall plots SHAP values of one-hot columns are summed per original feature and shown with the original category",
        "interpretation_note": "SHAP values describe contributions to model predictions and are not causal effects",
        "collinearity_note": "interventional SHAP and permutation importance split credit between strongly related features (tenure, TotalCharges, MonthlyCharges and the service columns) in a way that depends on the model; their importance should be read jointly, see interpretability_vif.csv"
    }
    (RESULTS_DIR / "shap_metadata.json").write_text(json.dumps(shap_metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    set_style()
    upstream = require_stages(["03_data_preparation", "04_modeling", "05_evaluation"])
    X_train, y_train, X_test, y_test, predictions, metadata, summary = load_data()

    print("\nInterpretability / xAI\n")
    analyze_permutation_importance(X_test, y_test)
    run_shap(X_train, X_test, predictions, metadata)
    analyze_interpretable_logistic_regression(summary, X_train, y_train)

    inputs = [PROCESSED_DIR / name for name in ("X_train.csv", "y_train.csv", "X_test.csv", "y_test.csv")]
    inputs += [RESULTS_DIR / "test_predictions.csv", RESULTS_DIR / "final_model_metadata.json", RESULTS_DIR / "modeling_cv_summary.csv", MODELS_DIR / "final_churn_pipeline.joblib"]
    outputs = [RESULTS_DIR / name for name in (
        "permutation_importance_original_features.csv", "shap_global_encoded.csv", "shap_global_original_features.csv", "shap_local_cases.csv",
        "shap_metadata.json", "interpretable_lr_odds_ratios.csv", "interpretability_vif.csv", "interpretable_lr_metadata.json"
    )]
    parameters = {"shap_max_samples": SHAP_MAX_SAMPLES, "shap_background_samples": SHAP_BACKGROUND_SAMPLES, "permutation_repeats": PERMUTATION_REPEATS, "random_state": RANDOM_STATE}
    run_id = record_stage("05b_interpretability", inputs, outputs, parameters, upstream)
    print(f"Interpretability outputs saved to results/ and figures/ (run {run_id})")


if __name__ == "__main__":
    main()