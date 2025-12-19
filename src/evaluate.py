"""
src/evaluate.py

Evaluation of train/val data and test prediction script.

Assumes train_model.py has already:
- split train/validation
- fit feature selection on TRAIN only
- trained the model
- saved:
    - model.pkl
    - X_val.npy
    - y_val.npy
    - selected_features.json

This script:
1. Evaluates on validation set
2. Reports per-label metrics
3. Predicts on test set (no labels)

Example usage(CLI):
python src/evaluate.py \
  --data-dir data/processed \
  --model-dir data/processed/model \
  --output-dir data/processed/eval

Author: Anni Bamwenda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from feature_select import apply_feature_selector


TARGET_NAMES = ["ADHD_Outcome", "Sex_F"]


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob) -> dict:
    metrics = {}

    for i, target in enumerate(TARGET_NAMES):
        metrics[target] = {
            "accuracy": accuracy_score(y_true[:, i], y_pred[:, i]),
            "precision": precision_score(y_true[:, i], y_pred[:, i]),
            "recall": recall_score(y_true[:, i], y_pred[:, i]),
            "f1": f1_score(y_true[:, i], y_pred[:, i]),
            "roc_auc": roc_auc_score(y_true[:, i], y_prob[i][:, 1]),
        }

    return metrics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate model and predict test set")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--model-dir", type=str, default="data/processed/model")
    parser.add_argument("--output-dir", type=str, default="data/processed/eval")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------
    # Load artifacts
    # --------------------
    print("Loading evaluation artifacts...")

    model = joblib.load(model_dir / "model.pkl")

    X_val = np.load(model_dir / "X_val.npy")
    y_val = np.load(model_dir / "y_val.npy")

    X_test = np.load(data_dir / "X_test.npy")

    with open(data_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    with open(model_dir / "selected_features.json") as f:
        selected_features = json.load(f)

    # --------------------
    # Validation evaluation
    # --------------------
    print("Evaluating on validation set...")

    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)

    metrics = compute_metrics(y_val, y_val_pred, y_val_prob)

    # Save metrics
    with open(output_dir / "validation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "validation_report.txt", "w") as f:
        for target, scores in metrics.items():
            f.write(f"{target}\n")
            for k, v in scores.items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("\n")

    print("Validation metrics:")
    for target, scores in metrics.items():
        print(target)
        for k, v in scores.items():
            print(f"  {k}: {v:.4f}")

    # --------------------
    # Test prediction
    # --------------------
    print("Generating test predictions...")

    X_test_sel = apply_feature_selector(
        X_test,
        feature_names,
        selected_features,
    )

    test_preds = model.predict(X_test_sel)

    test_df = pd.DataFrame(test_preds, columns=TARGET_NAMES)
    test_df.to_csv(output_dir / "test_predictions.csv", index=False)

    print(f"Saved test predictions to {output_dir / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
