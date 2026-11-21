import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import ID_COLUMN

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("counterfactuals_summary", ROOT / "05c_counterfactuals.py")
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)

THRESHOLD, MARGIN = 0.5, 0.05
BANK = "Bank transfer (automatic)"


class AdditiveModel:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        score = (
            0.7
            - 0.2 * X["Contract"].eq("One year")
            - 0.5 * X["Contract"].eq("Two year")
            - 0.12 * X["OnlineSecurity"].eq("Yes")
            - 0.12 * X["TechSupport"].eq("Yes")
            - 0.06 * X["PaymentMethod"].isin(cf.AUTOMATIC_PAYMENTS)
        ).to_numpy(dtype=float)
        return np.column_stack([1 - score, score])


ROW = pd.Series({
    "Contract": "Month-to-month", "PaymentMethod": "Electronic check", "InternetService": "DSL", "PhoneService": "Yes",
    "MultipleLines": "No", "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No", "MonthlyCharges": 50.0
})
X_TRAIN = pd.DataFrame({"MonthlyCharges": [20.0, 80.0]})


def search(excluded=()):
    return cf.search_counterfactuals(AdditiveModel(), None, X_TRAIN, ROW, THRESHOLD, 3, MARGIN, 5, excluded, reprice=False)


def as_pairs(solution):
    return [(c["feature"], c["new"]) for c in solution["changes"]]


def test_search_returns_the_expected_minimal_solutions_in_rank_order():
    solutions = search()
    assert [as_pairs(s) for s in solutions] == [
        [("Contract", "Two year")],
        [("Contract", "One year"), ("OnlineSecurity", "Yes")],
        [("Contract", "One year"), ("TechSupport", "Yes")],
        [("Contract", "One year"), ("PaymentMethod", BANK)],
        [("PaymentMethod", BANK), ("OnlineSecurity", "Yes"), ("TechSupport", "Yes")]
    ]
    assert [s["n_changes"] for s in solutions] == [1, 2, 2, 2, 3]
    assert [s["change_cost"] for s in solutions] == [2, 2, 2, 2, 3]
    assert [s["score_after"] for s in solutions] == pytest.approx([0.2, 0.38, 0.38, 0.44, 0.40])


def test_every_solution_is_valid_and_subset_minimal():
    model = AdditiveModel()
    for solution in search():
        assert solution["score_after"] < THRESHOLD - MARGIN
        for size in range(1, len(solution["changes"])):
            for subset in itertools.combinations(solution["changes"], size):
                candidate = cf.build_candidate(ROW, subset)
                assert model.predict_proba(pd.DataFrame([candidate]))[0, 1] >= THRESHOLD - MARGIN


def test_variant_without_contract_and_determinism():
    solutions = search(excluded=("Contract",))
    assert [as_pairs(s) for s in solutions] == [[("PaymentMethod", BANK), ("OnlineSecurity", "Yes"), ("TechSupport", "Yes")]]
    assert all(c["feature"] != "Contract" for s in solutions for c in s["changes"])

    first, second = search(), search()
    assert [as_pairs(s) for s in first] == [as_pairs(s) for s in second]
    assert [s["score_after"] for s in first] == [s["score_after"] for s in second]


def test_no_counterfactual_when_actions_cannot_reach_the_target():
    row = ROW.copy()
    row["InternetService"] = "No"
    assert cf.search_counterfactuals(AdditiveModel(), None, X_TRAIN, row, THRESHOLD, 3, MARGIN, 5, ("Contract",), reprice=False) == []


def output_rows():
    base = {"score_after": np.nan, "n_changes": np.nan, "plausibility_after": np.nan, "monthly_charges_after": np.nan, "changed_features": ""}

    def found(variant, cid, y, rank, n, before, after, mc_after, features):
        return {"variant": variant, ID_COLUMN: cid, "y_true": y, "rank": rank, "counterfactual_found": True, "score_before": before, "score_after": after,
                "n_changes": n, "plausibility_before": 0.1, "plausibility_after": 0.2, "monthly_charges_before": 50.0, "monthly_charges_after": mc_after, "changed_features": features}

    def missing(variant, cid, y, before):
        return {**base, "variant": variant, ID_COLUMN: cid, "y_true": y, "rank": np.nan, "counterfactual_found": False, "score_before": before,
                "plausibility_before": 0.1, "monthly_charges_before": 50.0}

    return pd.DataFrame([
        found("all_levers", "c1", 1, 1, 1, 0.8, 0.3, 50.0, "Contract"),
        found("all_levers", "c1", 1, 2, 2, 0.8, 0.4, 55.0, "OnlineSecurity;TechSupport"),
        missing("all_levers", "c2", 1, 0.9),
        found("all_levers", "c3", 0, 1, 2, 0.6, 0.4, 55.0, "OnlineSecurity;TechSupport"),
        found("without_contract", "c1", 1, 1, 2, 0.8, 0.4, 55.0, "OnlineSecurity;TechSupport"),
        missing("without_contract", "c2", 1, 0.9)
    ])


def test_summary_splits_true_and_false_positives_with_per_customer_denominators():
    summary = cf.summarize(output_rows()).set_index(["variant", "customer_group"])

    overall = summary.loc[("all_levers", "all")]
    assert (overall["customers"], overall["customers_with_counterfactual"]) == (3, 2)
    assert overall["share_with_counterfactual"] == pytest.approx(2 / 3)
    assert overall["median_score_before_all"] == pytest.approx(0.8)
    assert overall["median_n_changes_rank1"] == pytest.approx(1.5)
    assert overall["share_single_change_rank1"] == pytest.approx(0.5)
    assert overall["median_score_reduction_rank1"] == pytest.approx(0.35)
    assert overall["median_monthly_charges_change_rank1"] == pytest.approx(2.5)

    tp, fp = summary.loc[("all_levers", "TP")], summary.loc[("all_levers", "FP")]
    assert (tp["customers"], tp["customers_with_counterfactual"], tp["share_with_counterfactual"]) == (2, 1, 0.5)
    assert tp["median_score_before_all"] == pytest.approx(0.85) and tp["median_n_changes_rank1"] == 1
    assert (fp["customers"], fp["customers_with_counterfactual"], fp["share_with_counterfactual"]) == (1, 1, 1.0)

    empty = summary.loc[("without_contract", "FP")]
    assert empty["customers"] == 0 and empty["customers_with_counterfactual"] == 0
    assert np.isnan(empty["share_with_counterfactual"]) and np.isnan(empty["median_n_changes_rank1"])


def test_intervention_frequency_denominators():
    frequency = cf.intervention_frequency(output_rows()).set_index(["variant", "feature"])

    assert frequency.loc[("all_levers", "OnlineSecurity"), "n_solutions"] == 2
    assert frequency.loc[("all_levers", "Contract"), "n_customers_rank1"] == 1
    assert frequency.loc[("all_levers", "TechSupport"), "share_of_analysed_customers"] == pytest.approx(1 / 3)
    assert frequency.loc[("all_levers", "TechSupport"), "share_of_customers_with_counterfactual"] == pytest.approx(1 / 2)
    assert frequency.loc[("without_contract", "OnlineSecurity"), "share_of_analysed_customers"] == pytest.approx(1 / 2)
    assert frequency.loc[("without_contract", "OnlineSecurity"), "share_of_customers_with_counterfactual"] == pytest.approx(1.0)
