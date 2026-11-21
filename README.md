# Customer churn prediction in telecommunications

Predicting customer churn in the telco industry using machine learning, with a focus on interpretability and business considerations

## Project overview

Customer churn - customers cancelling their subscription - is a key cost driver for telecom operators. The project follows CRISP-DM and combines three parts:

- **Prediction** - tuned and compared classification models that rank customers by churn risk
- **Interpretability** - global and local explanations of the selected model and counterfactual explanations restricted to plausible retention actions
- **Business analysis** - a scenario-based simulation of retention campaigns built on the model's ranking

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) - an IBM sample dataset describing a fictional telecom company, distributed on Kaggle by BlastChar. It has 7043 customers and 21 columns (customer ID, demographics, services, account information and the target `Churn`). The project uses `data/WA_Fn-UseC_-Telco-Customer-Churn.xlsx`, the Kaggle CSV converted to XLSX without changes to its content. Check the dataset page for its terms of use.

## Methodology

- CRISP-DM phases 2-6 implemented as separate scripts - phase 1 (business understanding) is covered in the thesis
- One stratified 80/20 train/test split - all preprocessing and resampling are fitted inside training folds only
- Logistic Regression, Random Forest and XGBoost, each with no balancing, class weights and SMOTENC
- Randomized hyperparameter search with repeated stratified 5-fold CV - the final model is selected by mean average precision and variants are compared with the Nadeau-Bengio corrected t-test with Holm correction (exploratory)
- Decision threshold (max F1), isotonic calibration, risk groups and retention policies are fixed on out-of-fold predictions from the training set - the test set is used only to evaluate the frozen model
- Test metrics with bootstrap confidence intervals, top-k lift and cumulative gains - all nine variants are also compared on the test set without re-selecting the winner (the `*_diff_vs_winner` columns of `test_comparison_all_models.csv` are means of paired bootstrap differences, variant minus winner, and can differ slightly from the difference of the point estimates)
- Permutation importance, SHAP (on the raw model output, before calibration), an unpenalized logistic regression with odds ratios as a transparent reference and counterfactual explanations over a constrained set of retention actions
- Retention simulation with three scenarios, sensitivity to campaign success rate and margin and the break-even success rate of each contact policy
- Business operating points - thresholds 0.5, max F1, min FPR at recall >= 90% and min FN/FP cost, plus contact with the top 10% of customers, all chosen on training OOF scores and evaluated once on the test set; ROC curves with operating points, confusion matrices, FN/FP trade-off, cost-ratio analysis and a comparison of all nine variants at the recall requirement
- Model capacity - validation curves for logistic-regression C, random-forest max_depth and min_samples_leaf, XGBoost max_depth and XGBoost learning rate vs number of trees, on the training set only
- Demonstration batch scoring with input validation and PSI drift monitoring
- Stages 03-05f record data, code and artefact hashes in a run manifest - later stages and scoring refuse to use inconsistent artefacts

## Project structure

```
02_data_understanding.py    EDA and data quality checks
03_data_preparation.py      cleaning and train/test split
04_modeling.py              hyperparameter search and comparison of 9 variants
05_evaluation.py            threshold, calibration, test evaluation, retention simulation
05b_interpretability.py     permutation importance, SHAP, reference logistic regression
05c_counterfactuals.py      counterfactual explanations
05d_model_comparison.py     test comparison of all variants and baselines
05e_threshold_analysis.py   business operating points: thresholds for FN/FP requirements and costs
05f_model_capacity.py       hyperparameters and model capacity
06_deployment_scoring.py    batch scoring and monitoring
run_pipeline.py             runs the stages in order with logs
clean_outputs.py            backs up and removes generated artefacts before a clean run
config.py                   paths, seeds and parameters
src/                        shared code
tests/                      unit and integration tests
data/                       raw dataset
```

## Installation and usage

Python 3.11 on Windows (PowerShell):

```powershell
git clone https://github.com/Przemo200/Predicting_customer_churn_in_the_telco_industry_using_machine_learning
cd Predicting_customer_churn_in_the_telco_industry_using_machine_learning
py -3.11 -m venv .venv
./.venv/Scripts/Activate.ps1
python -m pip install -r requirements-lock.txt
python -m pip check
```
```powershell
git status
python -m ruff check .
python -m pytest -q
```

`requirements.txt` and `requirements-dev.txt` list the direct dependencies and the test tools.

Running the full pipeline - stages 02-06, one log file per stage, stops at the first failure. Phase 4 performs about 8100 model fits - in the final run it took about 24 minutes and the full pipeline about 33 minutes, depending on the number of CPU cores:

```powershell
python run_pipeline.py
```

Rerun selected stages if needed, using compatible existing artifacts:

```powershell
python run_pipeline.py --stages 05 05b 05c 05d 05e 05f 06
```

Scoring needs the trained model, so run the pipeline first. Demonstration scoring on the original dataset:

```powershell
python 06_deployment_scoring.py
```

Batch scoring of another CSV or XLSX file with the 19 customer features (the target variable Churn is not required). Categories not seen in training stop the run unless `--allow-unseen` is given, in which case such rows are scored and flagged:

```powershell
python 06_deployment_scoring.py path/to/customers.xlsx --output results/new_scores.csv
```

## Outputs

All outputs are generated by the pipeline and are not stored in the repository:

- `data/processed/` - train/test split
- `models/` - tuned pipelines, the final model and the calibrator
- `results/` - metrics, predictions, explanations, retention analyses and metadata (CSV, JSON)
- `figures/` - figures used in the thesis
- `logs/` - console output of each stage
- `results/run_manifest.json` - run IDs, git commit, code fingerprint, library versions, parameters and SHA-256 hashes of the inputs and outputs of stages 03-05f

Clean run from scratch - `clean_outputs.py` lists the generated files and with `--execute --confirm CLEAN` copies them to a new `archive/pre_clean_<timestamp>/` folder, verifies the copy by SHA-256 and empties `data/processed/`, `models/`, `results/`, `figures/` and `logs/`. Raw data, code and existing `archive/` folders are never touched

```powershell
python clean_outputs.py
python clean_outputs.py --execute --confirm CLEAN
python run_pipeline.py
```


## About this repository

This repository is the release version of the code described in the engineering thesis. It was published as a single snapshot of a private development repository, so the intermediate development history is not included

## Limitations

- The data are a single snapshot of a fictional sample without dates: the timing of features relative to churn cannot be verified, some customer profiles repeat and results do not transfer directly to real operators
- The cross-validation score of the selected model is slightly optimistic and the OOF predictions behind the threshold and the calibrator come from the same training data as the tuning, so OOF-based numbers may be optimistic too. The held-out test set gives the final evaluation of the frozen procedure on this split. Bootstrap intervals are conditional on the frozen model. The statistical comparison of variants is exploratory
- A threshold that meets recall >= 90% on OOF predictions does not guarantee 90% recall on new data, the test set reports what the frozen threshold achieves and is never used to adjust it. FN/FP cost ratios are assumptions
- SMOTENC creates synthetic customers (e.g. fractional `tenure`). Isotonic calibration fitted on out-of-fold scores is applied to the model refitted on all training data
- SHAP values and counterfactuals explain model predictions, not causal effects of customer attributes or retention actions. A service the final model does not use directly (DeviceProtection) can still appear in a counterfactual because adding it raises the repriced MonthlyCharges - such a counterfactual reflects the model's step-wise price effect, not a recommended action
- The model estimates churn propensity, not uplift. Business results are hypothetical, revenue-based simulations under assumed costs, incentives and success rates
- Scoring the full dataset is a demonstration, not a validation. PSI shows distribution changes, not loss of performance. `gender` and `SeniorCitizen` as model inputs would need a fairness review in a real deployment
