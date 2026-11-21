"""
Evaluation helpers used in phases 5 and 6
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from config import CV_FOLDS, OOF_RANDOM_STATE, RETENTION_VALUE_MONTHS


def oof_probabilities(pipeline, X: pd.DataFrame, y: pd.Series, seed: int = OOF_RANDOM_STATE, verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    scores = np.full(len(X), np.nan, dtype=float)
    folds = np.full(len(X), -1, dtype=int)

    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        model = clone(pipeline)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        scores[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
        folds[valid_idx] = fold_idx

        if verbose:
            print(f"OOF fold {fold_idx}/{CV_FOLDS} completed")

    if np.isnan(scores).any():
        raise RuntimeError("Some training observations did not receive an OOF prediction")

    return scores, folds


def select_threshold(y_true: pd.Series, scores: np.ndarray) -> tuple[float, pd.DataFrame]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    rows = pd.DataFrame({"threshold": thresholds, "precision": precision[:-1], "recall": recall[:-1]})
    denominator = rows["precision"] + rows["recall"]
    rows["f1"] = np.where(denominator > 0, 2 * rows["precision"] * rows["recall"] / denominator, 0.0)

    best_idx = rows["f1"].idxmax()
    threshold = float(rows.loc[best_idx, "threshold"])
    rows["selected"] = False
    rows.loc[best_idx, "selected"] = True

    return threshold, rows


def threshold_metrics(y_true: pd.Series, scores: np.ndarray, threshold: float, label: str) -> dict:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    return {
        "threshold_label": label,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


def assign_risk_group(scores: np.ndarray, low_cut: float, high_cut: float) -> np.ndarray:
    return np.where(scores >= high_cut, "High", np.where(scores >= low_cut, "Medium", "Low"))


def contact_net_benefit(y: np.ndarray, monthly_charges: np.ndarray, cost: float, success_rate: float, incentive_months: float, margin: float = 1.0) -> np.ndarray:
    value = margin * monthly_charges * RETENTION_VALUE_MONTHS
    incentive = monthly_charges * incentive_months
    return np.where(y == 1, success_rate * (value - incentive), -incentive) - cost


def expected_contact_benefit(probabilities: np.ndarray, monthly_charges: np.ndarray, cost: float, success_rate: float, incentive_months: float, margin: float = 1.0) -> np.ndarray:
    value = margin * monthly_charges * RETENTION_VALUE_MONTHS
    incentive = monthly_charges * incentive_months
    return probabilities * success_rate * (value - incentive) - (1.0 - probabilities) * incentive - cost


def break_even_probability(monthly_charges: float, params: dict) -> float:
    value = params.get("margin", 1.0) * monthly_charges * RETENTION_VALUE_MONTHS
    incentive = monthly_charges * params["incentive_months"]
    return (incentive + params["cost"]) / (params["success_rate"] * (value - incentive) + incentive)


def benefit_kwargs(params: dict) -> dict:
    return {
        "cost": params["cost"],
        "success_rate": params["success_rate"],
        "incentive_months": params["incentive_months"],
        "margin": params.get("margin", 1.0)
    }


def contact_count(fraction: float, n: int) -> int:
    return int(np.ceil(np.round(fraction * n, 9)))


def top_fraction_mask(scores: np.ndarray, fraction: float) -> np.ndarray:
    order = np.argsort(-np.asarray(scores), kind="stable")
    mask = np.zeros(len(order), dtype=bool)
    mask[order[:contact_count(fraction, len(order))]] = True
    return mask


def retention_profit_curve(y_true: pd.Series, scores: np.ndarray, monthly_charges: pd.Series, params: dict, fractions: np.ndarray) -> pd.DataFrame:
    order = np.argsort(-np.asarray(scores), kind="stable")
    benefit = contact_net_benefit(np.asarray(y_true)[order], np.asarray(monthly_charges, dtype=float)[order], **benefit_kwargs(params))
    cumulative = np.r_[0.0, np.cumsum(benefit)]
    customers = np.array([contact_count(f, len(benefit)) for f in fractions], dtype=int)

    return pd.DataFrame({
        "fraction": fractions,
        "customers": customers,
        "net_benefit": cumulative[customers],
        "random_net_benefit": customers / len(benefit) * float(benefit.sum())
    })


def select_contact_fraction(curve: pd.DataFrame) -> int:
    return int(curve.sort_values(["net_benefit", "fraction"], ascending=[False, True], kind="stable").index[0])


def expected_value_policy(y_true: pd.Series, probabilities: np.ndarray, monthly_charges: pd.Series, params: dict) -> tuple[int, float]:
    charges = np.asarray(monthly_charges, dtype=float)
    kwargs = benefit_kwargs(params)
    contact = expected_contact_benefit(np.asarray(probabilities), charges, **kwargs) > 0
    realized = contact_net_benefit(np.asarray(y_true), charges, **kwargs)

    return int(contact.sum()), float(realized[contact].sum())


def policy_outcome(y: np.ndarray, monthly_charges: np.ndarray, contact: np.ndarray, params: dict) -> dict:
    benefit = contact_net_benefit(np.asarray(y), np.asarray(monthly_charges, dtype=float), **benefit_kwargs(params))
    contact = np.asarray(contact, dtype=bool)
    customers = int(contact.sum())
    net = float(benefit[contact].sum())
    random_net = customers / len(benefit) * float(benefit.sum())

    return {
        "customers_contacted": customers,
        "share_contacted": customers / len(benefit),
        "net_benefit": net,
        "random_net_benefit": random_net,
        "advantage_vs_random": net - random_net
    }


def break_even_success_rate(y: np.ndarray, monthly_charges: np.ndarray, cost: float, incentive_months: float, margin: float = 1.0) -> tuple[float, str]:
    y = np.asarray(y)
    charges = np.asarray(monthly_charges, dtype=float)
    if len(y) == 0:
        return float("nan"), "no_contact"

    value = margin * charges * RETENTION_VALUE_MONTHS
    incentive = charges * incentive_months
    gain = float(np.sum(np.where(y == 1, value - incentive, 0.0)))
    loss = float(np.sum(np.where(y == 0, incentive, 0.0)) + cost * len(y))

    if gain <= 0:
        if loss <= 0 and gain == 0:
            return 0.0, "feasible"
        return float("nan"), "never_profitable"

    rate = loss / gain
    return float(rate), "feasible" if rate <= 1.0 else "infeasible_above_one"


def confusion_rates(tp, fp, fn, tn) -> dict:
    tp, fp, fn, tn = (np.asarray(v, dtype=float) for v in (tp, fp, fn, tn))
    positives, negatives, contacted = tp + fn, fp + tn, tp + fp
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "recall": tp / positives,
            "fnr": fn / positives,
            "fpr": fp / negatives,
            "tnr": tn / negatives,
            "precision": np.where(contacted > 0, tp / contacted, np.nan),
            "f1": 2 * tp / (2 * tp + fp + fn),
            "contact_rate": contacted / (positives + negatives)
        }


def operating_points(y_true, scores) -> pd.DataFrame:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    positives = int(y.sum())
    if positives == 0 or positives == len(y):
        raise ValueError("Operating points need at least one churner and one non-churner")

    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    block_end = np.r_[np.flatnonzero(np.diff(s_sorted) != 0), len(s) - 1]
    tp = np.r_[0, np.cumsum(y_sorted)[block_end]]
    fp = np.r_[0, np.cumsum(1 - y_sorted)[block_end]]
    fn = positives - tp
    tn = len(y) - positives - fp

    return pd.DataFrame({
        "threshold": np.r_[np.inf, s_sorted[block_end]],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "contacted": tp + fp,
        **confusion_rates(tp, fp, fn, tn)
    })


def select_min_fpr_at_recall(points: pd.DataFrame, min_recall: float) -> pd.Series:
    if not 0.0 < min_recall <= 1.0:
        raise ValueError(f"min_recall must be in (0, 1], got {min_recall}")

    feasible = points.loc[points["recall"] >= min_recall]
    return feasible.sort_values(["fpr", "recall", "threshold"], ascending=[True, False, False], kind="stable").iloc[0]


def select_min_cost(points: pd.DataFrame, cost_fn: float, cost_fp: float) -> pd.Series:
    if cost_fn < 0 or cost_fp < 0:
        raise ValueError("Costs must be non-negative")

    cost = points["fn"] * cost_fn + points["fp"] * cost_fp
    return points.assign(cost=cost).sort_values(["cost", "contacted"], ascending=[True, True], kind="stable").iloc[0]


def decision_summary(y_true, contact) -> dict:
    y = np.asarray(y_true).astype(int)
    contact = np.asarray(contact, dtype=bool)
    tp = int(np.sum(contact & (y == 1)))
    fp = int(np.sum(contact & (y == 0)))
    fn = int(np.sum(~contact & (y == 1)))
    tn = int(np.sum(~contact & (y == 0)))
    rates = {key: float(value) for key, value in confusion_rates(tp, fp, fn, tn).items()}

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "contacted": tp + fp, **rates}
