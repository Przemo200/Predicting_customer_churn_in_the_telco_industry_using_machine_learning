"""
Phase 3 CRISP-DM - data preparation

3.1 Select data
3.2 Clean data
3.3 Construct data (no derived features)
3.4 Integrate data (not applicable because of single dataset)
3.5 Format data
"""

from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    ID_COLUMN,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET,
    TEST_SIZE,
)
from src.data import clean_feature_values, encode_target
from src.manifest import record_stage
from src.preprocessing import build_preprocessor, get_feature_lists


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = clean_feature_values(df)

    if ID_COLUMN not in df.columns:
        raise ValueError(f"Missing ID column: {ID_COLUMN}")
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    customer_ids = df[ID_COLUMN].astype(str)
    y = encode_target(df[TARGET])
    X = df.drop(columns=[ID_COLUMN, TARGET])

    return X, y, customer_ids


def split_data(X: pd.DataFrame, y: pd.Series, customer_ids: pd.Series):
    return train_test_split(X, y, customer_ids, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def define_preprocessing(X_train: pd.DataFrame) -> dict:
    numeric_features, categorical_features = get_feature_lists(X_train)
    feature_names = build_preprocessor(numeric_features, categorical_features).fit(X_train).get_feature_names_out().tolist()

    return {"numeric_features": numeric_features, "categorical_features": categorical_features, "feature_names": feature_names}


def main() -> None:
    raw = pd.read_excel(RAW_DATA_PATH)
    X, y, customer_ids = prepare_data(raw)

    X_train, X_test, y_train, y_test, id_train, id_test = split_data(X, y, customer_ids)
    artifacts = define_preprocessing(X_train)

    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)
    id_train.to_frame(name=ID_COLUMN).to_csv(PROCESSED_DIR / "customer_id_train.csv", index=False)
    id_test.to_frame(name=ID_COLUMN).to_csv(PROCESSED_DIR / "customer_id_test.csv", index=False)

    (PROCESSED_DIR / "feature_names_after_encoding.json").write_text(json.dumps(artifacts["feature_names"], indent=2, ensure_ascii=False), encoding="utf-8")

    metadata = {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "positive_share_train": float(y_train.mean()),
        "positive_share_test": float(y_test.mean()),
        "input_features": X_train.columns.tolist(),
        "numeric_features": artifacts["numeric_features"],
        "categorical_features": artifacts["categorical_features"],
        "encoded_feature_count": len(artifacts["feature_names"]),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE
    }
    (PROCESSED_DIR / "data_preparation_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    outputs = [PROCESSED_DIR / name for name in (
        "X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv", "customer_id_train.csv", "customer_id_test.csv",
        "feature_names_after_encoding.json", "data_preparation_metadata.json"
    )]
    run_id = record_stage("03_data_preparation", [RAW_DATA_PATH], outputs, {"random_state": RANDOM_STATE, "test_size": TEST_SIZE, "stratified": True})

    print("train:", X_train.shape, round(y_train.mean(), 4))
    print("test:", X_test.shape, round(y_test.mean(), 4))
    print("encoded features:", len(artifacts["feature_names"]))
    print(f"run {run_id} recorded in results/run_manifest.json")


if __name__ == "__main__":
    main()