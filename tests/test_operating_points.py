import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix, f1_score

from config import RETENTION_SCENARIOS
from src.evaluation import (
    contact_net_benefit,
    decision_summary,
    operating_points,
    retention_profit_curve,
    select_min_cost,
    select_min_fpr_at_recall,
    select_threshold,
)

ROOT = Path(__file__).resolve().parents[1]


def load_script(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


thresholds_stage = load_script("05e_threshold_analysis.py", "threshold_stage")

TIED_SCORES = np.array([0.9, 0.8, 0.8, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
TIED_LABELS = pd.Series([1, 0, 1, 1, 0, 0, 1, 0, 0, 0])


@pytest.fixture
def random_data():
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.27, 600)
    scores = np.round(np.clip(0.3 + 0.3 * y + rng.normal(0, 0.2, 600), 0, 1), 2)
    return pd.Series(y), scores


def test_operating_points_match_confusion_matrix_for_every_threshold(random_data):
    y, scores = random_data
    points = operating_points(y, scores)
    assert points["threshold"].iloc[0] == np.inf
    assert len(points) == len(np.unique(scores)) + 1

    for _, row in points.iterrows():
        tn, fp, fn, tp = confusion_matrix(y, (scores >= row["threshold"]).astype(int), labels=[0, 1]).ravel()
        assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (tp, fp, fn, tn)
        assert row["recall"] == pytest.approx(tp / (tp + fn))
        assert row["fnr"] == pytest.approx(fn / (tp + fn))
        assert row["fpr"] == pytest.approx(fp / (fp + tn))
        assert row["tnr"] == pytest.approx(tn / (fp + tn))
        assert row["contacted"] == tp + fp
        assert row["contact_rate"] == pytest.approx((tp + fp) / len(y))
        if tp + fp:
            assert row["precision"] == pytest.approx(tp / (tp + fp))
            assert row["f1"] == pytest.approx(f1_score(y, scores >= row["threshold"]))

    assert np.all(np.diff(points["recall"]) >= 0) and np.all(np.diff(points["fpr"]) >= 0)
    assert points["recall"].iloc[-1] == 1.0 and points["fpr"].iloc[-1] == 1.0


def test_no_contact_row_and_max_f1_agree_with_phase_5_rule(random_data):
    y, scores = random_data
    points = operating_points(y, scores)
    first = points.iloc[0]
    assert (first["tp"], first["fp"], first["contacted"], first["recall"], first["f1"]) == (0, 0, 0, 0.0, 0.0)
    assert np.isnan(first["precision"])

    f1_threshold, _ = select_threshold(y, scores)
    best = points.loc[points["f1"].eq(points["f1"].max()), "threshold"].min()
    assert best == pytest.approx(f1_threshold)


def test_tied_scores_switch_together():
    points = operating_points(TIED_LABELS, TIED_SCORES)
    row = points.loc[points["threshold"].eq(0.8)].iloc[0]
    assert (row["tp"], row["fp"], row["contacted"]) == (3, 1, 4)
    assert 2 not in points["contacted"].tolist()


def test_min_fpr_at_recall_hand_example_and_boundary():
    points = operating_points(TIED_LABELS, TIED_SCORES)
    row = select_min_fpr_at_recall(points, 0.75)
    assert row["threshold"] == 0.8 and row["recall"] == 0.75 and row["fp"] == 1
    row = select_min_fpr_at_recall(points, 0.76)
    assert row["threshold"] == 0.3 and row["recall"] == 1.0 and row["fp"] == 3 and row["fn"] == 0
    assert select_min_fpr_at_recall(points, 1.0)["threshold"] == 0.3

    y = pd.Series([1] * 9 + [0, 1])
    exact = select_min_fpr_at_recall(operating_points(y, np.r_[np.linspace(0.99, 0.91, 9), 0.5, 0.1]), 0.9)
    assert exact["threshold"] == pytest.approx(0.91) and exact["recall"] == 0.9 and exact["fp"] == 0


def test_min_fpr_ties_prefer_higher_recall():
    y = pd.Series([1, 0, 1, 1, 0])
    points = operating_points(y, np.array([0.9, 0.7, 0.6, 0.5, 0.1]))
    row = select_min_fpr_at_recall(points, 0.6)
    assert row["threshold"] == 0.5 and row["recall"] == 1.0 and row["fp"] == 1


@pytest.mark.parametrize("target", [0.0, -0.1, 1.01])
def test_min_fpr_rejects_invalid_targets(target):
    with pytest.raises(ValueError):
        select_min_fpr_at_recall(operating_points(TIED_LABELS, TIED_SCORES), target)


@pytest.mark.parametrize("labels", [[0, 0, 0], [1, 1, 1]])
def test_operating_points_need_both_classes(labels):
    with pytest.raises(ValueError):
        operating_points(pd.Series(labels), np.array([0.1, 0.5, 0.9]))


def test_min_cost_extremes_and_ties():
    points = operating_points(TIED_LABELS, TIED_SCORES)
    expensive = select_min_cost(points, cost_fn=1.0, cost_fp=100.0)
    assert expensive["threshold"] == 0.9 and expensive["fp"] == 0
    assert select_min_cost(operating_points(pd.Series([0, 1, 0]), np.array([0.9, 0.5, 0.1])), cost_fn=1.0, cost_fp=100.0)["threshold"] == np.inf
    assert select_min_cost(points, cost_fn=100.0, cost_fp=1.0)["fn"] == 0
    middle = select_min_cost(points, cost_fn=1.0, cost_fp=1.0)
    assert middle["cost"] == (points["fn"] + points["fp"]).min()
    flat = operating_points(pd.Series([0, 1]), np.array([0.9, 0.1]))
    assert select_min_cost(flat, cost_fn=1.0, cost_fp=1.0)["contacted"] == 0
    with pytest.raises(ValueError):
        select_min_cost(points, cost_fn=-1.0, cost_fp=1.0)


def test_decision_summary_uses_churn_as_positive_class():
    y = np.array([1, 1, 0, 0, 1])
    contact = np.array([True, False, True, False, True])
    summary = decision_summary(y, contact)
    tn, fp, fn, tp = confusion_matrix(y, contact.astype(int), labels=[0, 1]).ravel()
    assert (summary["tp"], summary["fp"], summary["fn"], summary["tn"]) == (tp, fp, fn, tn) == (2, 1, 1, 1)
    assert summary["recall"] == pytest.approx(2 / 3) and summary["fnr"] == pytest.approx(1 / 3) and summary["fpr"] == pytest.approx(0.5)
    assert np.isnan(decision_summary(y, np.zeros(5, dtype=bool))["precision"])


def test_cost_with_scenario_costs_matches_retention_optimum_for_constant_charges():
    rng = np.random.default_rng(5)
    y = pd.Series(rng.binomial(1, 0.3, 400))
    scores = rng.uniform(size=400) + 0.4 * y.to_numpy()
    charges = pd.Series(np.full(400, 70.0))
    params = RETENTION_SCENARIOS["base"]
    cost_fn, cost_fp = thresholds_stage.scenario_error_costs(params, 70.0)
    assert cost_fn == pytest.approx(contact_net_benefit(np.array([1]), np.array([70.0]), 20.0, 0.3, 1)[0])
    assert cost_fp == pytest.approx(70.0 + 20.0)

    chosen = select_min_cost(operating_points(y, scores), cost_fn, cost_fp)
    fractions = np.arange(401) / 400
    curve = retention_profit_curve(y, scores, charges, params, fractions)
    assert chosen["contacted"] == curve.loc[curve["net_benefit"].idxmax(), "customers"]
    assert curve["net_benefit"].max() == pytest.approx(y.sum() * cost_fn - chosen["cost"])


def test_strategies_ignore_test_labels_and_capacity_check():
    rng = np.random.default_rng(7)
    y_train = pd.Series(rng.binomial(1, 0.27, 500))
    oof = np.clip(0.25 + 0.35 * y_train.to_numpy() + rng.normal(0, 0.2, 500), 0, 1)
    y_test = rng.binomial(1, 0.27, 200)
    test = np.clip(0.25 + 0.35 * y_test + rng.normal(0, 0.2, 200), 0, 1)

    strategies = thresholds_stage.select_strategies(y_train, oof, 0.4, 2.0, 1.0)
    assert strategies["strategy"].tolist() == list(thresholds_stage.STRATEGIES)
    results, masks = thresholds_stage.evaluate_strategies(strategies, {"test": (y_test, test)})
    shuffled, shuffled_masks = thresholds_stage.evaluate_strategies(strategies, {"test": (rng.permutation(y_test), test)})
    pd.testing.assert_series_equal(results["contacted"], shuffled["contacted"])
    assert all(np.array_equal(masks[k], shuffled_masks[k]) for k in masks)
    assert results.set_index("strategy").loc["top_capacity", "contacted"] == 20

    capacity = thresholds_stage.capacity_check(operating_points(y_train, oof), y_test, test)
    assert capacity["jointly_feasible_on_oof"] == (capacity["oof_min_contact_rate_for_target_recall"] <= thresholds_stage.CONTACT_CAPACITY)
    assert not capacity["jointly_feasible_on_oof"]
    assert capacity["oof_max_recall_within_capacity"] <= capacity["perfect_ranking_max_recall_within_capacity"] < thresholds_stage.TARGET_RECALL


def test_cost_ratio_analysis_contacts_more_when_misses_get_more_expensive():
    rng = np.random.default_rng(11)
    y_train = pd.Series(rng.binomial(1, 0.27, 500))
    oof = rng.uniform(size=500) + 0.5 * y_train.to_numpy()
    y_test = pd.Series(rng.binomial(1, 0.27, 200))
    test = rng.uniform(size=200) + 0.5 * y_test.to_numpy()
    table = thresholds_stage.cost_ratio_analysis(y_train, oof, y_test, test, {"base": (200.0, 85.0)})
    grid = table.loc[table["source"].eq("assumed_ratio")]
    assert np.all(np.diff(grid["oof_contact_rate"]) >= 0)
    assert np.all(np.diff(grid["oof_threshold"]) <= 0)
    assert (table["test_cost"] == table["test_fn"] * table["cost_ratio_fn_to_fp"] + table["test_fp"]).all()


def test_paired_fpr_bootstrap_is_deterministic_and_zero_for_the_reference(monkeypatch):
    monkeypatch.setattr(thresholds_stage, "BOOTSTRAP_ITERATIONS", 50)
    rng = np.random.default_rng(2)
    y = rng.binomial(1, 0.3, 300)
    scores = {"w": np.clip(0.3 + 0.4 * y + rng.normal(0, 0.2, 300), 0, 1), "o": rng.uniform(size=300)}
    result = thresholds_stage.paired_bootstrap_fpr(y, scores, "w").set_index("combo_id")
    pd.testing.assert_frame_equal(result, thresholds_stage.paired_bootstrap_fpr(y, scores, "w").set_index("combo_id"))
    assert result.loc["w", ["oof_fpr_diff_ci_low", "oof_fpr_diff_ci_high"]].tolist() == [0.0, 0.0]
    assert result.loc["o", "oof_fpr_diff_ci_low"] > 0
