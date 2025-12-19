"""
src/train_model.py

Leakage-free training pipeline:
- train/val split
- feature selection on TRAIN only
- model training on TRAIN only

Example usage(CLI):
python src/train_model.py \
  --data-dir data/processed \
  --output-dir data/processed/model \
Author: Anni Bamwenda
"""

import json
import argparse
import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from feature_select import fit_feature_selector, apply_feature_selector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/processed/model")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    X = np.load(data_dir / "X_train.npy")
    y = np.load(data_dir / "y_train.npy")

    with open(data_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    # Split data
    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=y[:, 0],  # stratify on ADHD_Outcome
    )

    # Feature selection on TRAIN ONLY
    print("Performing feature selection...")
    selected_features = fit_feature_selector(
        X_train,
        y_train,
        feature_names,
    )

    # Apply selection
    X_train_sel = apply_feature_selector(X_train, feature_names, selected_features)
    X_val_sel = apply_feature_selector(X_val, feature_names, selected_features)

    # Train model
    print("Training model...")
    base_rf = RandomForestClassifier(
        n_estimators=400,
        random_state=args.random_state,
        n_jobs=-1,
    )

    model = MultiOutputClassifier(base_rf)
    model.fit(X_train_sel, y_train)

    # Save artifacts
    print("Saving artifacts...")
    joblib.dump(model, output_dir / "model.pkl")
    np.save(output_dir / "X_val.npy", X_val_sel)
    np.save(output_dir / "y_val.npy", y_val)

    with open(output_dir / "selected_features.json", "w") as f:
        json.dump(selected_features, f, indent=2)

    print(f"Training complete. Selected {len(selected_features)} features.")


if __name__ == "__main__":
    main()
