import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from config import (
    RETENTION_VALUE_MONTHS,
    SENSITIVITY_MARGINS,
    SENSITIVITY_SUCCESS_RATES,
)
from src.evaluation import (
    break_even_probability,
    break_even_success_rate,
    contact_net_benefit,
    expected_contact_benefit,
    expected_value_policy,
    policy_outcome,
    retention_profit_curve,
    select_contact_fraction,
    select_threshold,
    top_fraction_mask,
)
from src.stats import corrected_resampled_ttest

ROOT = Path(__file__).resolve().parents[1]
PARAMS = {"cost": 10.0, "success_rate": 0.5, "incentive_months": 1}


def load_script(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = load_script("05_evaluation.py", "evaluation_script")
modeling = load_script("04_modeling.py", "modeling_script")
comparison = load_script("05d_model_comparison.py", "comparison_script")

TIED_SCORES = np.array([0.9, 0.8, 0.8, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
TIED_LABELS = pd.Series([1, 0, 1, 1, 0, 0, 1, 0, 0, 0])


def test_top_k_uses_stable_order_for_ties_and_ceil_counts():
    result = evaluation.top_k_analysis(TIED_LABELS, TIED_SCORES)
    assert result["customers"].tolist() == [1, 2, 3, 5]
    assert result["positives_captured"].tolist() == [1, 1, 2, 3]
    assert result["capture_rate"].tolist() == pytest.approx([0.25, 0.25, 0.5, 0.75])
    assert result["lift"].tolist() == pytest.approx([1.0 / 0.4, 0.5 / 0.4, (2 / 3) / 0.4, 0.6 / 0.4])
    pd.testing.assert_frame_equal(result, evaluation.top_k_analysis(TIED_LABELS, TIED_SCORES))


def test_gains_curve_is_monotone_and_ends_at_one():
    x, captured = evaluation.gains_curve(TIED_LABELS, TIED_SCORES)
    assert x.tolist() == pytest.approx(np.r_[0.0, np.arange(1, 11) / 10].tolist())
    assert captured.tolist() == pytest.approx([0, 0.25, 0.25, 0.5, 0.75, 0.75, 0.75, 1, 1, 1, 1])
    assert np.all(np.diff(captured) >= 0)


def test_select_threshold_maximises_f1_with_the_ge_rule():
    y = pd.Series([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    threshold, table = select_threshold(y, scores)
    assert threshold == pytest.approx(0.35)
    for _, row in table.iterrows():
        assert row["f1"] == pytest.approx(f1_score(y, (scores >= row["threshold"]).astype(int)))
    assert table["selected"].sum() == 1


def test_select_threshold_ties_go_to_the_lowest_threshold():
    threshold, table = select_threshold(pd.Series([1, 0, 0, 1]), np.array([0.9, 0.8, 0.7, 0.6]))
    assert table["f1"].max() == pytest.approx(2 / 3)
    assert threshold == pytest.approx(0.6)


def test_retention_profit_curve_includes_no_campaign_and_matches_hand_computation():
    y = pd.Series([1, 0, 1])
    curve = retention_profit_curve(y, np.array([0.9, 0.2, 0.5]), pd.Series([100.0, 50.0, 10.0]), PARAMS, np.array([0.0, 1 / 3, 0.5, 1.0]))
    assert curve["customers"].tolist() == [0, 1, 2, 3]
    assert curve["net_benefit"].tolist() == pytest.approx([0.0, 540.0, 585.0, 525.0])
    assert curve["random_net_benefit"].tolist() == pytest.approx([0.0, 175.0, 350.0, 525.0])
    assert select_contact_fraction(curve) == 2


def test_unprofitable_campaign_selects_no_contact():
    fractions = np.r_[0.0, np.linspace(0.01, 1.0, 100)]
    curve = retention_profit_curve(pd.Series([0] * 20), np.linspace(1, 0, 20), pd.Series([50.0] * 20), PARAMS, fractions)
    best = select_contact_fraction(curve)
    assert fractions[best] == 0.0 and curve.loc[best, "net_benefit"] == 0.0 and curve.loc[best, "customers"] == 0

    free = retention_profit_curve(pd.Series([0] * 5), np.linspace(1, 0, 5), pd.Series([50.0] * 5), {"cost": 0.0, "success_rate": 0.5, "incentive_months": 0}, fractions)
    assert fractions[select_contact_fraction(free)] == 0.0


def test_top_fraction_mask_counts_and_order():
    mask = top_fraction_mask(np.array([0.2, 0.9, 0.9, 0.1]), 0.5)
    assert mask.tolist() == [False, True, True, False]
    assert not top_fraction_mask(np.array([0.2, 0.9]), 0.0).any()
    assert top_fraction_mask(np.arange(100) / 100, 0.07).sum() == 7


def test_expected_value_policy_contacts_only_positive_expected_benefit():
    p_star = break_even_probability(100.0, PARAMS)
    probabilities = np.array([0.9, 0.05, p_star + 1e-6, p_star - 1e-6])
    y = pd.Series([1, 0, 0, 1])
    charges = pd.Series([100.0] * 4)
    contacted, realized = expected_value_policy(y, probabilities, charges, PARAMS)
    assert contacted == 2
    assert realized == pytest.approx(contact_net_benefit(np.array([1, 0]), np.array([100.0, 100.0]), 10.0, 0.5, 1).sum())


def test_margin_scales_retained_value_but_not_incentive_or_cost():
    full = contact_net_benefit(np.array([1, 0]), np.array([100.0, 100.0]), 10.0, 0.5, 1, margin=1.0)
    half = contact_net_benefit(np.array([1, 0]), np.array([100.0, 100.0]), 10.0, 0.5, 1, margin=0.5)
    assert full.tolist() == pytest.approx([0.5 * (1200 - 100) - 10, -110])
    assert half.tolist() == pytest.approx([0.5 * (600 - 100) - 10, -110])
    p_star = break_even_probability(100.0, {**PARAMS, "margin": 0.5})
    assert expected_contact_benefit(np.array([p_star]), np.array([100.0]), 10.0, 0.5, 1, margin=0.5)[0] == pytest.approx(0.0, abs=1e-9)


def test_break_even_success_rate_matches_algebra_and_zero_benefit():
    rate, status = break_even_success_rate(np.array([1, 0]), np.array([100.0, 100.0]), cost=10.0, incentive_months=1)
    assert status == "feasible" and rate == pytest.approx(120 / 1100)
    benefit = contact_net_benefit(np.array([1, 0]), np.array([100.0, 100.0]), 10.0, rate, 1)
    assert benefit.sum() == pytest.approx(0.0, abs=1e-9)

    rate_half_margin, _ = break_even_success_rate(np.array([1, 0]), np.array([100.0, 100.0]), 10.0, 1, margin=0.5)
    assert rate_half_margin == pytest.approx(120 / 500)


def test_break_even_success_rate_statuses_are_not_clipped():
    y = np.array([1] + [0] * 11)
    rate, status = break_even_success_rate(y, np.full(12, 100.0), cost=10.0, incentive_months=1)
    assert status == "infeasible_above_one" and rate == pytest.approx((11 * 100 + 12 * 10) / 1100)

    assert break_even_success_rate(np.array([0, 0]), np.array([50.0, 50.0]), 10.0, 1)[1] == "never_profitable"
    assert np.isnan(break_even_success_rate(np.array([0, 0]), np.array([50.0, 50.0]), 10.0, 1)[0])
    assert break_even_success_rate(np.array([1]), np.array([100.0]), 10.0, 1, margin=1 / RETENTION_VALUE_MONTHS)[1] == "never_profitable"
    rate, status = break_even_success_rate(np.array([]), np.array([]), 10.0, 1)
    assert status == "no_contact" and np.isnan(rate)


def test_policy_outcome_random_baseline_is_proportional_expectation():
    y = np.array([1, 0, 1, 0])
    charges = np.array([100.0, 50.0, 10.0, 20.0])
    outcome = policy_outcome(y, charges, np.array([True, False, True, False]), PARAMS)
    benefit = contact_net_benefit(y, charges, 10.0, 0.5, 1)
    assert outcome["customers_contacted"] == 2
    assert outcome["net_benefit"] == pytest.approx(benefit[[0, 2]].sum())
    assert outcome["random_net_benefit"] == pytest.approx(0.5 * benefit.sum())
    assert outcome["advantage_vs_random"] == pytest.approx(outcome["net_benefit"] - outcome["random_net_benefit"])


@pytest.fixture
def retention_data():
    rng = np.random.default_rng(0)

    def make(n):
        y = pd.Series(rng.binomial(1, 0.3, n))
        scores = np.clip(0.3 + 0.35 * y + rng.normal(0, 0.2, n), 0, 1)
        return y, scores, pd.Series(rng.uniform(20, 110, n))

    return make(400), make(200)


def test_sensitivity_policies_are_fixed_without_test_labels(retention_data):
    (y_tr, s_tr, c_tr), (y_te, s_te, c_te) = retention_data
    probabilities = np.clip(s_te, 0, 1)
    result = evaluation.retention_sensitivity(y_tr, s_tr, c_tr, y_te, s_te, probabilities, c_te, fixed_fraction=0.2)
    assert len(result) == len(SENSITIVITY_MARGINS) * len(SENSITIVITY_SUCCESS_RATES) * 5

    shuffled = evaluation.retention_sensitivity(y_tr, s_tr, c_tr, pd.Series(np.zeros(len(y_te), dtype=int)), s_te, probabilities, c_te, fixed_fraction=0.2)
    pd.testing.assert_series_equal(result["fraction_selected_on_oof"], shuffled["fraction_selected_on_oof"])
    pd.testing.assert_series_equal(result["customers_contacted"], shuffled["customers_contacted"])

    fixed = result.loc[result["split"].eq("test") & result["policy"].eq("fixed_top_fraction")]
    assert fixed["customers_contacted"].nunique() == 1
    for _, group in fixed.groupby("margin"):
        assert np.allclose(np.diff(group["net_benefit"].to_numpy(), n=2), 0.0, atol=1e-6)


def test_break_even_table_gives_zero_benefit_at_the_reported_rate(retention_data):
    (y_tr, s_tr, c_tr), (y_te, s_te, c_te) = retention_data
    summary, _ = evaluation.retention_analysis(y_tr, s_tr, pd.DataFrame({"MonthlyCharges": c_tr}), y_te, s_te, np.clip(s_te, 0, 1), pd.DataFrame({"MonthlyCharges": c_te}))
    table = evaluation.retention_break_even(y_tr, s_tr, c_tr, y_te, s_te, c_te, summary)

    feasible = table.loc[table["break_even_status"].isin(["feasible", "infeasible_above_one"])]
    assert len(feasible) > 0
    for _, row in feasible.iterrows():
        y, scores, charges = (y_tr, s_tr, c_tr) if row["split"] == "oof" else (y_te, s_te, c_te)
        contact = top_fraction_mask(scores, row["fraction"])
        params = {**evaluation.RETENTION_SCENARIOS[row["scenario"]], "success_rate": row["break_even_success_rate"], "margin": row["margin"]}
        assert policy_outcome(y, charges, contact, params)["net_benefit"] == pytest.approx(0.0, abs=1e-6)

    random_rows = table.loc[table["policy"].eq("contact_all_or_random_selection")]
    assert (random_rows["customers_contacted"] == random_rows["split"].map({"oof": 400, "test": 200})).all()


def test_retention_analysis_can_decline_the_campaign(retention_data, monkeypatch):
    (y_tr, s_tr, c_tr), (y_te, s_te, c_te) = retention_data
    monkeypatch.setattr(evaluation, "RETENTION_SCENARIOS", {"expensive": {"cost": 5000.0, "success_rate": 0.3, "incentive_months": 1}})
    summary, curves = evaluation.retention_analysis(y_tr, s_tr, pd.DataFrame({"MonthlyCharges": c_tr}), y_te, s_te, np.clip(s_te, 0, 1), pd.DataFrame({"MonthlyCharges": c_te}))
    row = summary.iloc[0]
    assert row["best_fraction_selected_on_oof"] == 0.0
    assert row["test_customers_contacted_at_best_fraction"] == 0 and row["test_net_benefit_at_best_fraction"] == 0.0
    assert curves["fraction"].iloc[0] == 0.0

    table = evaluation.retention_break_even(y_tr, s_tr, c_tr, y_te, s_te, c_te, summary)
    assert set(table.loc[table["policy"].eq("top_fraction_selected_on_oof"), "break_even_status"]) == {"no_contact"}


def _write_cv_results(path: Path, best: np.ndarray, reverse_columns: bool) -> None:
    columns = [f"split{k}_test_average_precision" for k in range(len(best))]
    frame = pd.DataFrame([np.full(len(best), 0.1), best], columns=columns)
    frame = frame[columns[::-1]] if reverse_columns else frame
    frame.insert(0, "rank_test_average_precision", [2, 1])
    frame.to_csv(path, index=False)


def test_paired_comparison_pairs_folds_numerically(tmp_path, monkeypatch):
    monkeypatch.setattr(modeling, "RESULTS_DIR", tmp_path)
    winner = np.linspace(0.60, 0.70, 15)
    diff = np.array([-0.03, 0.01, -0.02, 0.0, -0.01, -0.04, 0.02, -0.03, -0.01, -0.02, 0.01, -0.05, -0.02, 0.0, -0.01])
    _write_cv_results(tmp_path / "cv_results_w.csv", winner, reverse_columns=False)
    _write_cv_results(tmp_path / "cv_results_o.csv", winner + diff, reverse_columns=True)
    _write_cv_results(tmp_path / "cv_results_same.csv", winner, reverse_columns=True)

    result = modeling.paired_comparison_vs_winner(pd.DataFrame({"combo_id": ["w", "o", "same"]}), n_train=1000).set_index("combo_id")
    t_stat, p_value = corrected_resampled_ttest(diff, 800, 200)
    assert result.loc["o", "mean_ap_difference_vs_winner"] == pytest.approx(diff.mean())
    assert result.loc["o", "t_statistic"] == pytest.approx(t_stat)
    assert result.loc["o", "p_value_nadeau_bengio"] == pytest.approx(p_value)
    assert result.loc["same", "t_statistic"] == 0.0 and result.loc["same", "p_value_nadeau_bengio"] == 1.0
    assert result.loc["o", "p_value_holm"] == pytest.approx(min(1.0, 2 * p_value))


def test_paired_bootstrap_is_deterministic_and_winner_difference_is_zero(monkeypatch):
    monkeypatch.setattr(comparison, "BOOTSTRAP_ITERATIONS", 200)
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.3, 300)
    scores = {"w": np.clip(0.3 + 0.4 * y + rng.normal(0, 0.2, 300), 0, 1), "o": rng.uniform(size=300)}
    thresholds = {"w": 0.5, "o": 0.5}

    result = comparison.paired_bootstrap(y, scores, thresholds, "w").set_index("combo_id")
    pd.testing.assert_frame_equal(result, comparison.paired_bootstrap(y, scores, thresholds, "w").set_index("combo_id"))
    assert result.loc["w", ["average_precision_diff_vs_winner", "average_precision_diff_ci_low", "average_precision_diff_ci_high"]].tolist() == [0.0, 0.0, 0.0]
    assert result.loc["o", "average_precision_diff_ci_high"] < 0
    assert (result["average_precision_ci_low"] <= result["average_precision_ci_high"]).all()
