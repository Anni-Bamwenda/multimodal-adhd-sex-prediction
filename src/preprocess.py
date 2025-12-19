"""
src/preprocess.py
Preprocessing pipeline for multi-modal ADHD dataset.
Load → clean → merge → save

Handles:
- Categorical metadata
- Quantitative metadata
- Functional connectome matrices
- Multi-output labels (ADHD_Outcome, Sex_F)

Example usage:
python src/preprocess.py\
  --data_dir data \
  --output_dir data/processed

  
Outputs:
- X_train.npy
- y_train.npy
- X_test.npy
- feature_names.json
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import argparse
import os

# Ensure openpyxl is available for Excel file handling
try:
    import openpyxl  # noqa: F401
except ImportError as e:
    raise ImportError(
        "Missing dependency 'openpyxl'. Install it with: pip install openpyxl"
    ) from e


# -------------------- Constants --------------------
ID_COL = "participant_id"
TARGET_COLS = ["ADHD_Outcome", "Sex_F"]


# -------------------- Loaders --------------------
def load_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    expected = {ID_COL, *TARGET_COLS}

    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing target columns: {missing}")

    return df[[ID_COL] + TARGET_COLS]


# -------------------- Preprocessing --------------------
def prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = {
        c: f"{prefix}_{c}"
        for c in df.columns
        if c != ID_COL
    }
    return df.rename(columns=cols)


def preprocess_categorical(df: pd.DataFrame) -> pd.DataFrame:
    return prefix_columns(df, "categorical")


def preprocess_quantitative(df: pd.DataFrame) -> pd.DataFrame:
    return prefix_columns(df, "quantitative")


def preprocess_connectome(df: pd.DataFrame) -> pd.DataFrame:
    return prefix_columns(df, "fc")


# -------------------- Merge --------------------
def merge_modalities(
    cat: pd.DataFrame,
    quant: pd.DataFrame,
    conn: pd.DataFrame
) -> pd.DataFrame:
    df = cat.merge(quant, on=ID_COL, how="inner")
    df = df.merge(conn, on=ID_COL, how="inner")

    if df.empty:
        raise ValueError("Merged dataframe is empty. Check participant_id alignment.")

    return df


def merge_labels(X: pd.DataFrame, y: pd.DataFrame):
    merged = X.merge(y, on=ID_COL, how="inner")

    X_out = merged.drop(columns=TARGET_COLS + [ID_COL])
    y_out = merged[TARGET_COLS]

    return X_out, y_out


# -------------------- Feature Groups --------------------
def get_feature_groups(df: pd.DataFrame) -> dict:
    return {
        "categorical": [c for c in df.columns if c.startswith("categorical_")],
        "quantitative": [c for c in df.columns if c.startswith("quantitative_")],
        "fc": [c for c in df.columns if c.startswith("fc_")]
    }

# -------------------- Feature Names --------------------
def get_feature_names(df: pd.DataFrame) -> dict:
    return {
        "categorical": [c for c in df.columns if c.startswith("categorical_")],
        "quantitative": [c for c in df.columns if c.startswith("quantitative_")],
        "fc": [c for c in df.columns if c.startswith("fc_")]
    }

# -------------------- Main --------------------
def main(data_dir: Path, output_dir: Path):
    train_dir = data_dir / "TRAIN"
    test_dir = data_dir / "TEST"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- TRAIN ----
    train_cat = preprocess_categorical(load_excel(train_dir / "TRAIN_CATEGORICAL_METADATA_new.xlsx"))
    train_quant = preprocess_quantitative(load_excel(train_dir / "TRAIN_QUANTITATIVE_METADATA_new.xlsx"))
    train_conn = preprocess_connectome(load_csv(train_dir / "TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv"))

    X_train = merge_modalities(train_cat, train_quant, train_conn)
    y_train = load_labels(train_dir / "TRAINING_SOLUTIONS.xlsx")

    X_train, y_train = merge_labels(X_train, y_train)

    # ---- TEST ----
    test_cat = preprocess_categorical(load_excel(test_dir / "TEST_CATEGORICAL.xlsx"))
    test_quant = preprocess_quantitative(load_excel(test_dir / "TEST_QUANTITATIVE_METADATA.xlsx"))
    test_conn = preprocess_connectome(load_csv(test_dir / "TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv"))

    X_test = merge_modalities(test_cat, test_quant, test_conn)
    X_test = X_test.drop(columns=[ID_COL])

    # ---- Save ----
    np.save(output_dir / "X_train.npy", X_train.to_numpy(dtype=float))
    np.save(output_dir / "y_train.npy", y_train.to_numpy(dtype=float))
    np.save(output_dir / "X_test.npy", X_test.to_numpy(dtype=float))

    feature_groups = get_feature_groups(X_train)
    pd.Series(feature_groups).to_json(output_dir / "feature_groups.json")
    
    feature_names = X_train.columns.tolist()
    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f)


    print("Preprocessing complete!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    main(args.data_dir, args.output_dir)
