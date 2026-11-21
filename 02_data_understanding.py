"""
Phase 2 CRISP-DM - data understanding

2.1 Collect initial data
2.2 Describe data
2.3 Explore data
2.4 Verify data quality
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

from config import (
    ID_COLUMN,
    POSITIVE_LABEL,
    RAW_DATA_PATH,
    TARGET,
    TOTAL_CHARGES,
)
from src.plotting import (
    CHURN_COLORS,
    MAIN_COLOR,
    REFERENCE_COLOR,
    commas_in_figure,
    finish_axes,
    number,
    percent,
    percent_formatter,
    save_figure,
    set_style,
)
from src.stats import holm_adjust_pvalues

CHURN_TICKS = {"No": "Nie", "Yes": "Tak"}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
SELECTED_CATEGORICAL_PLOTS = ["Contract", "PaymentMethod", "InternetService"]
SELECTED_BOXPLOTS = ["tenure", "MonthlyCharges"]
BOXPLOT_LABELS = {"tenure": "tenure [miesiące]", "MonthlyCharges": "MonthlyCharges [USD]"}
FILE_NAMES = {"Contract": "contract", "PaymentMethod": "payment_method", "InternetService": "internet_service", "tenure": "tenure", "MonthlyCharges": "monthly_charges"}


def describe_data(df: pd.DataFrame) -> None:
    print(f"\nShape: {df.shape}\n")
    print(df.dtypes)
    print()

    tc = pd.to_numeric(df[TOTAL_CHARGES], errors="coerce")

    print(df[["tenure", "MonthlyCharges"]].assign(TotalCharges=tc).describe().round(2))
    print()

    for col in df.columns.drop([ID_COLUMN] + NUMERIC_COLS):
        print(col, df[col].value_counts().to_dict())


def plot_target_distribution(df: pd.DataFrame) -> None:
    counts = df[TARGET].value_counts().reindex(["No", "Yes"])

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    bars = ax.bar([CHURN_TICKS[k] for k in counts.index], counts.values, width=0.6, color=[CHURN_COLORS[k] for k in counts.index])

    for bar, value in zip(bars, counts.values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 80, f"{value}\n({percent(value / len(df))})", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Churn")
    ax.set_ylabel("Liczba klientów")
    ax.set_ylim(0, counts.max() * 1.20)

    finish_axes(ax)
    save_figure(fig, "fig_churn_distribution.pdf")


def plot_churn_rate_by(df: pd.DataFrame, col: str, churn: pd.Series) -> None:
    rate = churn.groupby(df[col].astype(str)).mean().sort_values(ascending=False)
    overall = churn.mean()

    fig, ax = plt.subplots(figsize=(5.6, 0.45 * len(rate) + 1.4))
    bars = ax.barh(range(len(rate)), rate.values, height=0.62, color=CHURN_COLORS["Yes"])

    for bar, value in zip(bars, rate.values, strict=True):
        ax.text(value + 0.008, bar.get_y() + bar.get_height() / 2, percent(value), va="center", fontsize=8)

    ax.axvline(overall, color=REFERENCE_COLOR, linewidth=1, linestyle="--", label=f"Ogółem ({percent(overall)})")
    ax.set_yticks(range(len(rate)))
    ax.set_yticklabels(rate.index)
    ax.set_xlabel("Odsetek rezygnacji")
    ax.set_ylabel(col)
    ax.set_xlim(0, max(rate.max(), overall) * 1.18)
    ax.xaxis.set_major_formatter(percent_formatter())
    ax.invert_yaxis()
    ax.legend(loc="lower right")

    finish_axes(ax)
    save_figure(fig, f"fig_churn_rate_by_{FILE_NAMES[col]}.pdf")


def plot_boxplot(df: pd.DataFrame, col: str) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.7))

    sns.boxplot(data=df, x=TARGET, y=col, hue=TARGET, order=["No", "Yes"], palette=CHURN_COLORS, width=0.55, linewidth=1.0, legend=False, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([CHURN_TICKS["No"], CHURN_TICKS["Yes"]])
    ax.set_xlabel("Churn")
    ax.set_ylabel(BOXPLOT_LABELS.get(col, col))

    finish_axes(ax)
    save_figure(fig, f"fig_boxplot_{FILE_NAMES[col]}.pdf")


def plot_cramers_v(df: pd.DataFrame, top_n: int = 10) -> None:
    cats = df.columns.drop([ID_COLUMN, TARGET] + NUMERIC_COLS)
    rows = []

    for col in cats:
        table = pd.crosstab(df[col].astype(str), df[TARGET])
        chi2, p_value, dof, expected = chi2_contingency(table, correction=False)

        denominator = len(df) * (min(table.shape) - 1)
        cramers_v = np.sqrt(chi2 / denominator)

        rows.append({
            "feature": col,
            "cramers_v": float(cramers_v),
            "chi2": float(chi2),
            "degrees_of_freedom": int(dof),
            "p_value": float(p_value),
            "min_expected_count": float(expected.min())
        })

    results = pd.DataFrame(rows)
    results["p_value_holm"] = holm_adjust_pvalues(results["p_value"].to_numpy())
    results["significant_holm"] = results["p_value_holm"] < 0.05

    display_results = results.sort_values("cramers_v", ascending=False).reset_index(drop=True)

    print("\nChi-square tests and Cramer's V with Holm correction:")
    print(display_results[["feature", "cramers_v", "chi2", "degrees_of_freedom", "p_value", "p_value_holm", "min_expected_count", "significant_holm"]].to_string(index=False))

    top = display_results.head(top_n).sort_values("cramers_v")
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    bars = ax.barh(range(len(top)), top["cramers_v"], height=0.62, color=MAIN_COLOR)

    for bar, value in zip(bars, top["cramers_v"], strict=True):
        ax.text(value + 0.006, bar.get_y() + bar.get_height() / 2, number(value, 2), va="center", fontsize=8)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"])
    ax.set_xlabel("V Craméra (związek z Churn)")
    ax.set_xlim(0, top["cramers_v"].max() * 1.15)

    finish_axes(ax)
    save_figure(fig, "fig_cramers_v.pdf")


def plot_correlation_matrix(df: pd.DataFrame, churn: pd.Series) -> None:
    corr = df[NUMERIC_COLS].assign(Churn=churn.astype(int)).corr()

    fig, ax = plt.subplots(figsize=(5.2, 4.2))

    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8, "ticks": [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]}, ax=ax)

    ax.set_xticklabels(corr.columns, rotation=90, ha="center")
    ax.set_yticklabels(corr.columns, rotation=0)
    commas_in_figure(fig)
    for text in ax.texts:
        text.set_text(text.get_text().replace("-", "−"))

    save_figure(fig, "fig_correlation_matrix.pdf")


def verify_data_quality(df: pd.DataFrame) -> None:
    tc = pd.to_numeric(df[TOTAL_CHARGES], errors="coerce")

    internet_dep = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    no_internet = df["InternetService"].eq("No")

    print("NaN:", int(df.isna().sum().sum()))
    features = df.drop(columns=[ID_COLUMN])
    feature_cols = features.columns.drop(TARGET).tolist()
    conflicting = features.groupby(feature_cols, dropna=False)[TARGET].nunique().gt(1).sum()
    print("duplicated rows excluding ID:", int(features.duplicated().sum()))
    print("feature profiles with conflicting Churn labels:", int(conflicting))
    print("duplicated IDs:", int(df[ID_COLUMN].duplicated().sum()))
    print("TotalCharges empty:", int(tc.isna().sum()))
    print("all at tenure=0:", bool(df.loc[tc.isna(), "tenure"].eq(0).all()))
    print("rows with tenure=0:", int(df["tenure"].eq(0).sum()))
    print("internet levels consistent:", all(df.loc[no_internet, c].eq("No internet service").all() for c in internet_dep))
    print("phone levels consistent:", bool(df.loc[df["PhoneService"].eq("No"), "MultipleLines"].eq("No phone service").all()))
    ratio = tc / (df["tenure"] * df["MonthlyCharges"]).replace(0, np.nan)
    print("TotalCharges / (tenure * MonthlyCharges), 5/50/95 pct:", ratio.quantile([0.05, 0.5, 0.95]).round(2).tolist())
    print("corr(tenure, TotalCharges):", round(float(df["tenure"].corr(tc)), 3))

    service_cols = ["PhoneService", "MultipleLines", "InternetService"] + internet_dep
    design = np.column_stack([np.ones(len(df)), pd.get_dummies(df[service_cols], drop_first=True).to_numpy(dtype=float)])
    coef = np.linalg.lstsq(design, df["MonthlyCharges"].to_numpy(dtype=float), rcond=None)[0]
    residual = df["MonthlyCharges"].to_numpy(dtype=float) - design @ coef
    r2_price = 1.0 - residual.var() / df["MonthlyCharges"].var(ddof=0)
    print("R2 of MonthlyCharges explained by service columns only:", round(float(r2_price), 4))
    print("corr(TotalCharges, tenure * MonthlyCharges):", round(float(tc.corr(df["tenure"] * df["MonthlyCharges"])), 4))


def main() -> None:
    set_style()
    df = pd.read_excel(RAW_DATA_PATH)
    describe_data(df)

    df2 = df.copy()
    df2[TOTAL_CHARGES] = pd.to_numeric(df2[TOTAL_CHARGES], errors="coerce")
    df2["SeniorCitizen"] = df2["SeniorCitizen"].map({0: "No", 1: "Yes"})

    churn = df2[TARGET].eq(POSITIVE_LABEL)

    plot_target_distribution(df2)

    for col in SELECTED_CATEGORICAL_PLOTS:
        plot_churn_rate_by(df2, col, churn)

    for col in SELECTED_BOXPLOTS:
        plot_boxplot(df2, col)

    plot_cramers_v(df2)
    plot_correlation_matrix(df2, churn)

    print("\nMedian numeric values by Churn:")
    print(df2.groupby(TARGET)[["tenure", "MonthlyCharges"]].median())
    print("\nChurn rate by SeniorCitizen:")
    print(churn.groupby(df2["SeniorCitizen"]).mean().round(3).to_dict())
    print("\nData quality verification:")
    verify_data_quality(df)
    print()


if __name__ == "__main__":
    main()
