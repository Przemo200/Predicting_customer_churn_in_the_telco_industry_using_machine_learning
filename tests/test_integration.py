"""
Integration tests on the real dataset with small, quickly fitted models
"""

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shap

from config import RAW_DATA_PATH
from src.data import INTERNET_DEPENDENT
from src.explainability import transform_for_model

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.5
MARGIN = 0.05
IMMUTABLE = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "TotalCharges", "PhoneService", "InternetService", "PaperlessBilling"]


def load_script(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preparation = load_script("03_data_preparation.py", "preparation")
modeling = load_script("04_modeling.py", "modeling")
counterfactuals = load_script("05c_counterfactuals.py", "counterfactuals")
scoring = load_script("06_deployment_scoring.py", "scoring_integration")
interpretability = load_script("05b_interpretability.py", "interpretability_integration")


@pytest.fixture(scope="module")
def data():
    raw = pd.read_excel(RAW_DATA_PATH)
    X, y, ids = preparation.prepare_data(raw)
    X_train, X_test, y_train, _, _, _ = preparation.split_data(X, y, ids)
    return raw, X, X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True)


@pytest.fixture(scope="module")
def lr_pipeline(data):
    _, _, X_train, _, y_train = data
    return modeling.build_pipeline("lr", "none", X_train, y_train).fit(X_train, y_train)


def small_pipeline(model_key, balancing, X_train, y_train):
    pipeline = modeling.build_pipeline(model_key, balancing, X_train, y_train)
    if model_key in ("rf", "xgb"):
        pipeline.set_params(model__n_estimators=30, model__max_depth=4)
    return pipeline.fit(X_train, y_train)


def test_scoring_path_reproduces_training_features(data):
    raw, X, X_train, _, _ = data
    X_scoring, ids = scoring.prepare_scoring_features(raw, X.columns.tolist())
    pd.testing.assert_frame_equal(X_scoring.reset_index(drop=True), X.reset_index(drop=True))
    assert ids.tolist() == raw["customerID"].astype(str).tolist()

    validated, flags = scoring.validate_input(X_scoring, X_train, allow_unseen=False)
    assert len(validated) == len(raw)
    assert flags.isin(["ok"]).mean() > 0.99


@pytest.mark.parametrize("variant", list(counterfactuals.VARIANTS))
def test_counterfactuals_are_valid_minimal_and_actionable(data, lr_pipeline, variant):
    _, _, X_train, X_test, _ = data
    settings = counterfactuals.VARIANTS[variant]
    price_model, _ = counterfactuals.fit_price_model(X_train)
    minimum_price = float(X_train["MonthlyCharges"].min())
    target = THRESHOLD - MARGIN

    scores = lr_pipeline.predict_proba(X_test)[:, 1]
    rows = X_test.loc[scores >= THRESHOLD].head(25)
    found = 0

    def rebuild(row, changes):
        candidate = counterfactuals.build_candidate(row, tuple(changes))
        if settings["reprice"]:
            candidate["MonthlyCharges"] = counterfactuals.updated_monthly_charges(price_model, row, pd.DataFrame([candidate]), minimum_price)[0]
        return candidate

    def score(candidate):
        return lr_pipeline.predict_proba(pd.DataFrame([candidate])[X_test.columns])[:, 1][0]

    for _, row in rows.iterrows():
        solutions = counterfactuals.search_counterfactuals(lr_pipeline, price_model, X_train, row, THRESHOLD, 3, MARGIN, 5, settings["excluded"], settings["reprice"])
        found += bool(solutions)

        for solution in solutions:
            candidate = solution["candidate"]
            changes = solution["changes"]

            assert score(candidate) < target
            assert score(candidate) == pytest.approx(solution["score_after"])
            pd.testing.assert_series_equal(rebuild(row, changes), candidate, check_names=False)

            for size in range(1, len(changes)):
                for subset in itertools.combinations(changes, size):
                    assert score(rebuild(row, subset)) >= target

            assert candidate[IMMUTABLE].equals(row[IMMUTABLE])
            for feature in settings["excluded"]:
                assert candidate[feature] == row[feature]
            if candidate["InternetService"] == "No":
                assert (candidate[INTERNET_DEPENDENT] == "No").all()
            if candidate["PhoneService"] == "No":
                assert candidate["MultipleLines"] == "No"

            assert np.isfinite(candidate["MonthlyCharges"]) and candidate["MonthlyCharges"] >= minimum_price
            if not settings["reprice"]:
                assert candidate["MonthlyCharges"] == row["MonthlyCharges"]

    assert found > 0


def test_counterfactual_search_rejects_non_positive_target(data, lr_pipeline):
    _, _, X_train, X_test, _ = data
    price_model, _ = counterfactuals.fit_price_model(X_train)
    with pytest.raises(ValueError):
        counterfactuals.search_counterfactuals(lr_pipeline, price_model, X_train, X_test.iloc[0], 0.04, 3, 0.05, 5)


@pytest.mark.parametrize(("model_key", "balancing"), [("lr", "class_weight"), ("rf", "none"), ("xgb", "none"), ("xgb", "smotenc")])
def test_shap_values_add_up_to_model_output(data, model_key, balancing):
    _, _, X_train, X_test, y_train = data
    pipeline = small_pipeline(model_key, balancing, X_train, y_train)
    X_train_t, model = transform_for_model(pipeline, X_train)
    X_test_t, _ = transform_for_model(pipeline, X_test.head(50))

    explainer = shap.Explainer(model, shap.maskers.Independent(X_train_t.sample(100, random_state=0), max_samples=100))
    explanation = explainer(X_test_t)
    values = explanation.values if explanation.values.ndim == 2 else explanation.values[:, :, 1]
    base = np.asarray(explanation.base_values)
    base = base if base.ndim == 1 else base[:, 1]

    probability = pipeline.predict_proba(X_test.head(50))[:, 1]
    expected = probability if model_key == "rf" else np.log(probability / (1.0 - probability))
    assert np.allclose(values.sum(axis=1) + base, expected, atol=1e-2 if model_key == "rf" else 1e-4)


@pytest.mark.parametrize(("balancing", "max_depth", "min_samples_leaf"), [("none", None, 1), ("class_weight", 12, 2)])
def test_deep_random_forest_shap_is_additive(data, balancing, max_depth, min_samples_leaf):
    _, _, X_train, X_test, y_train = data
    pipeline = modeling.build_pipeline("rf", balancing, X_train, y_train)
    pipeline.set_params(model__n_estimators=50, model__max_depth=max_depth, model__min_samples_leaf=min_samples_leaf)
    pipeline.fit(X_train, y_train)
    X_train_t, model = transform_for_model(pipeline, X_train)
    X_test_t, _ = transform_for_model(pipeline, X_test.head(300))
    background = X_train_t.sample(50, random_state=0)
    probability = model.predict_proba(X_test_t.to_numpy())[:, 1]

    plain = shap.Explainer(model, shap.maskers.Independent(background, max_samples=50))
    plain_values = plain(X_test_t, check_additivity=False)
    plain_sum = plain_values.values[:, :, 1].sum(axis=1) + np.asarray(plain_values.base_values)[:, 1]
    assert np.abs(plain_sum - probability).max() > 1e-3

    explanation = interpretability.positive_class_explanation(interpretability.build_explainer(model, background)(X_test_t))
    reconstructed = explanation.values.sum(axis=1) + np.asarray(explanation.base_values)
    assert np.abs(reconstructed - probability).max() < interpretability.ADDITIVITY_TOLERANCE
    assert np.allclose(np.asarray(explanation.base_values), model.predict_proba(background.to_numpy())[:, 1].mean())
