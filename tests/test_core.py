import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import RETENTION_VALUE_MONTHS
from src.data import clean_feature_values
from src.evaluation import (
    assign_risk_group,
    break_even_probability,
    contact_net_benefit,
    expected_contact_benefit,
)
from src.explainability import (
    aggregate_local_shap,
    original_feature_name,
    variance_inflation_factors,
)
from src.stats import corrected_resampled_ttest, holm_adjust_pvalues

ROOT = Path(__file__).resolve().parents[1]


def load_script(filename: str):
    spec = importlib.util.spec_from_file_location("script", ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scoring = load_script("06_deployment_scoring.py")
counterfactuals = load_script("05c_counterfactuals.py")


def test_holm_known_values():
    adjusted = holm_adjust_pvalues([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06, 0.02])


def test_corrected_ttest_matches_formula():
    d = np.array([0.01, 0.02, 0.00, 0.03, 0.01])
    t_stat, p_value = corrected_resampled_ttest(d, n_train=80, n_test=20)
    expected_t = d.mean() / np.sqrt((1 / 5 + 20 / 80) * d.var(ddof=1))
    assert t_stat == pytest.approx(expected_t)
    assert 0.0 < p_value < 1.0
    assert corrected_resampled_ttest(np.zeros(5), 80, 20) == (0.0, 1.0)


def test_clean_feature_values():
    df = pd.DataFrame({
        "tenure": [0, 5],
        "TotalCharges": [" ", "50.5"],
        "SeniorCitizen": [0, 1],
        "OnlineSecurity": ["No internet service", "Yes"],
        "MultipleLines": ["No phone service", "Yes"]
    })
    out = clean_feature_values(df)
    assert out["TotalCharges"].tolist() == [0.0, 50.5]
    assert out["SeniorCitizen"].tolist() == ["No", "Yes"]
    assert out["OnlineSecurity"].tolist() == ["No", "Yes"]
    assert out["MultipleLines"].tolist() == ["No", "Yes"]

    df.loc[1, "TotalCharges"] = " "
    with pytest.raises(ValueError):
        clean_feature_values(df)


def test_clean_feature_values_separates_missing_from_invalid_total_charges():
    base = pd.DataFrame({"tenure": [0], "TotalCharges": [" "], "SeniorCitizen": [0]})
    assert clean_feature_values(base)["TotalCharges"].tolist() == [0.0]
    assert clean_feature_values(base.assign(TotalCharges=[np.nan]))["TotalCharges"].tolist() == [0.0]

    with pytest.raises(ValueError, match="Non-numeric"):
        clean_feature_values(base.assign(TotalCharges=["garbage"]))
    with pytest.raises(ValueError, match="tenure > 0"):
        clean_feature_values(base.assign(tenure=[5]))


def test_risk_groups_follow_threshold():
    groups = assign_risk_group(np.array([0.10, 0.30, 0.60]), low_cut=0.30, high_cut=0.50)
    assert groups.tolist() == ["Low", "Medium", "High"]


@pytest.fixture
def reference():
    return pd.DataFrame({"tenure": [1, 10, 70], "MonthlyCharges": [20.0, 60.0, 100.0], "Contract": ["Month-to-month", "One year", "Two year"]})


def test_validate_input_errors(reference):
    base = pd.DataFrame({"tenure": [5], "MonthlyCharges": [50.0], "Contract": ["One year"]})

    with pytest.raises(ValueError):
        scoring.validate_input(base.assign(tenure=[np.nan]), reference, allow_unseen=False)
    with pytest.raises(ValueError):
        scoring.validate_input(base.assign(MonthlyCharges=["abc"]), reference, allow_unseen=False)
    with pytest.raises(ValueError):
        scoring.validate_input(base.assign(tenure=[-1]), reference, allow_unseen=False)
    with pytest.raises(ValueError):
        scoring.validate_input(base.assign(Contract=["Three year"]), reference, allow_unseen=False)


def test_validate_input_flags(reference):
    base = pd.DataFrame({"tenure": [5, 500], "MonthlyCharges": [50.0, 50.0], "Contract": ["One year", "Three year"]})
    _, flags = scoring.validate_input(base, reference, allow_unseen=True)
    assert flags.tolist() == ["ok", "tenure_outside_training_range;Contract_unseen_category"]


def test_psi_is_skipped_for_small_batches_and_quiet_for_large_ones():
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"tenure": rng.integers(0, 73, 5000), "Contract": rng.choice(["a", "b", "c"], 5000)})

    small = scoring.monitoring_report(reference, reference.sample(30, random_state=1))
    assert set(small["status"]) == {"skipped_small_batch"}

    large = scoring.monitoring_report(reference, reference.sample(1000, random_state=1))
    assert set(large["status"]) == {"ok"}
    assert scoring.distribution_shift(reference["tenure"].to_numpy(), reference["tenure"].to_numpy(), False) == 0.0


def test_psi_detects_shift_of_constant_numeric_feature():
    constant = np.ones(1000)
    assert scoring.distribution_shift(constant, constant, False) == 0.0
    assert scoring.distribution_shift(constant, np.full(1000, 2.0), False) >= 0.25
    assert scoring.distribution_shift(constant, np.zeros(1000), False) >= 0.25


def test_counterfactual_changes_are_retention_actions():
    row = pd.Series({
        "Contract": "One year",
        "PaymentMethod": "Electronic check",
        "InternetService": "DSL",
        "PhoneService": "No",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "MultipleLines": "No"
    })
    changes = {(c["feature"], c["new"], c["cost"]) for c in counterfactuals.atomic_changes(row)}
    assert changes == {
        ("Contract", "Two year", 1),
        ("PaymentMethod", "Bank transfer (automatic)", 1),
        ("PaymentMethod", "Credit card (automatic)", 1),
        ("OnlineSecurity", "Yes", 1),
        ("DeviceProtection", "Yes", 1),
        ("TechSupport", "Yes", 1),
        ("StreamingMovies", "Yes", 1)
    }

    no_internet = row.copy()
    no_internet["InternetService"] = "No"
    features = {c["feature"] for c in counterfactuals.atomic_changes(no_internet, excluded=("Contract",))}
    assert features == {"PaymentMethod"}


def test_original_feature_name_uses_longest_prefix():
    originals = ["Contract", "PaymentMethod", "tenure"]
    assert original_feature_name("Contract_Month-to-month", originals) == "Contract"
    assert original_feature_name("PaymentMethod_Credit card (automatic)", originals) == "PaymentMethod"
    assert original_feature_name("tenure", originals) == "tenure"


def test_contact_net_benefit_charges_incentive_to_non_churners():
    charges = np.array([50.0, 50.0])
    benefit = contact_net_benefit(np.array([1, 0]), charges, cost=10.0, success_rate=0.5, incentive_months=1.0)
    value = 50.0 * RETENTION_VALUE_MONTHS
    assert benefit.tolist() == pytest.approx([0.5 * (value - 50.0) - 10.0, -50.0 - 10.0])


def test_break_even_probability_gives_zero_expected_benefit():
    params = {"cost": 20.0, "success_rate": 0.3, "incentive_months": 1.0}
    p = break_even_probability(70.0, params)
    assert 0.0 < p < 1.0
    assert expected_contact_benefit(np.array([p]), np.array([70.0]), **params)[0] == pytest.approx(0.0, abs=1e-9)


def test_vif_detects_linear_dependence():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=500), rng.normal(size=500)
    vif = variance_inflation_factors(pd.DataFrame({"a": a, "b": b, "noise": rng.normal(size=500)})).set_index("feature")["vif"]
    assert vif.max() < 1.1
    vif = variance_inflation_factors(pd.DataFrame({"a": a, "b": b, "sum": a + b})).set_index("feature")["vif"]
    assert np.isinf(vif).all()


def test_encode_target_accepts_only_exact_binary_labels():
    from src.data import encode_target

    assert encode_target(pd.Series(["No", "Yes"])).tolist() == [0, 1]
    assert encode_target(pd.Series([0, 1, 1])).tolist() == [0, 1, 1]
    assert encode_target(pd.Series([0.0, 1.0])).tolist() == [0, 1]
    for bad in (pd.Series([0.0, 0.2, 1.0]), pd.Series([0, 2]), pd.Series(["No", "Maybe"]), pd.Series([0.0, np.nan])):
        with pytest.raises(ValueError):
            encode_target(bad)


def test_validate_input_rejects_infinite_fractional_and_inconsistent_services(reference):
    base = pd.DataFrame({"tenure": [5], "MonthlyCharges": [50.0], "Contract": ["One year"]})
    with pytest.raises(ValueError):
        scoring.validate_input(base.assign(MonthlyCharges=[np.inf]), reference, allow_unseen=False)
    with pytest.raises(ValueError):
        scoring.validate_input(base.assign(tenure=[5.5]), reference, allow_unseen=False)

    services = pd.DataFrame({"InternetService": ["No", "DSL"], "OnlineSecurity": ["Yes", "Yes"], "PhoneService": ["No", "Yes"], "MultipleLines": ["Yes", "No"]})
    assert scoring.inconsistent_services(services) == {"OnlineSecurity_without_internet": 1, "MultipleLines_without_phone": 1}
    assert scoring.inconsistent_services(services.iloc[[1]]) == {}


def test_read_table_rejects_xls(tmp_path):
    with pytest.raises(ValueError):
        scoring.read_table(tmp_path / "customers.xls")


def test_aggregate_local_shap_sums_one_hot_columns_per_original_feature():
    row = pd.Series({"tenure": 5, "Contract": "Month-to-month", "PaymentMethod": "Electronic check"})
    encoded = ["tenure", "Contract_Month-to-month", "Contract_One year", "Contract_Two year", "PaymentMethod_Electronic check"]
    values, data, names = aggregate_local_shap(np.array([0.5, 0.3, 0.1, 0.2, -0.4]), encoded, row)
    assert names == ["tenure", "Contract", "PaymentMethod"]
    assert values == pytest.approx([0.5, 0.6, -0.4])
    assert data.tolist() == [5, "Month-to-month", "Electronic check"]