"""
Project configuration shared by all CRISP-DM phases
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

RAW_DATA_PATH = PROJECT_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.xlsx"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FIGURES_DIR = PROJECT_DIR / "figures"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

for _dir in (PROCESSED_DIR, FIGURES_DIR, MODELS_DIR, RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

ID_COLUMN = "customerID"
TARGET = "Churn"
POSITIVE_LABEL = "Yes"
TOTAL_CHARGES = "TotalCharges"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
CV_REPEATS = 3
SEARCH_ITER_BY_MODEL = {"lr": 60, "rf": 60, "xgb": 60}
OOF_RANDOM_STATE = RANDOM_STATE + 1

MODEL_RANKING_COLUMNS = ["cv_average_precision", "cv_roc_auc"]

TOP_K_FRACTIONS = (0.10, 0.20, 0.30, 0.50)
RISK_HIGH_QUANTILE = 0.80

SHAP_MAX_SAMPLES = 600
SHAP_BACKGROUND_SAMPLES = 200
SHAP_MAX_DISPLAY = 15
IMPORTANCE_MAX_DISPLAY = 10

COUNTERFACTUAL_MAX_CHANGES = 3
COUNTERFACTUAL_MARGIN = 0.05
COUNTERFACTUAL_TOP_SOLUTIONS = 5

BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_CI = 0.95

RETENTION_VALUE_MONTHS = 12
RETENTION_SCENARIOS = {
    "pessimistic": {"cost": 40.0, "success_rate": 0.15, "incentive_months": 2},
    "base": {"cost": 20.0, "success_rate": 0.30, "incentive_months": 1},
    "optimistic": {"cost": 10.0, "success_rate": 0.50, "incentive_months": 1}
}

SENSITIVITY_REFERENCE_SCENARIO = "base"
SENSITIVITY_SUCCESS_RATES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
SENSITIVITY_MARGINS = (1.0, 0.5, 0.25)

TARGET_RECALL = 0.90
CONTACT_CAPACITY = 0.10
COST_RATIOS_FN_TO_FP = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0)

PSI_WARNING = 0.10
PSI_ALERT = 0.25
PSI_MIN_ROWS = 500
PSI_BINS = 10

PERMUTATION_REPEATS = 10