"""XGBoost classifier training script for NTL detection.

Runs inside SageMaker training job via ModelTrainer (SDK v3).
Expects CSV input with a target column (default: 'FLAG').
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)


def parse_args():
    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--scale-pos-weight", type=float, default=10.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--target-column", type=str, default="FLAG")
    # SageMaker environment
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
    )
    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
    )
    parser.add_argument(
        "--test",
        type=str,
        default=os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test"),
    )
    return parser.parse_args()


def load_data(path, target_column):
    """Load all CSVs from a SageMaker input channel directory."""
    files = [f for f in os.listdir(path) if f.endswith(".csv")]
    df = pd.concat([pd.read_csv(os.path.join(path, f)) for f in files])
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def main():
    args = parse_args()

    print(f"Loading training data from {args.train}")
    X_train_full, y_train_full = load_data(args.train, args.target_column)
    print(f"Training set: {X_train_full.shape[0]} samples, {X_train_full.shape[1]} features")
    print(f"Class distribution: {dict(y_train_full.value_counts())}")

    print(f"Loading test data from {args.test}")
    X_test, y_test = load_data(args.test, args.target_column)
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    # Carve validation set from training data for early stopping (not test set)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full,
    )
    print(f"Train/val split: {X_train.shape[0]} train, {X_val.shape[0]} val")

    # Compute scale_pos_weight from actual training data if not explicitly overridden
    computed_spw = float((y_train == 0).sum()) / max((y_train == 1).sum(), 1)
    spw = args.scale_pos_weight if args.scale_pos_weight != 10.0 else computed_spw
    print(f"scale_pos_weight: {spw:.2f} (computed: {computed_spw:.2f})")

    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        scale_pos_weight=spw,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=20,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary"
    )
    auc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Theft"]))

    # Save model
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Save feature names
    feature_names = list(X_train.columns)
    with open(os.path.join(args.model_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    # Save metrics
    metrics = {
        "classification_metrics": {
            "precision": {"value": precision},
            "recall": {"value": recall},
            "f1": {"value": f1},
            "roc_auc": {"value": auc},
        }
    }
    with open(os.path.join(args.model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {args.model_dir}/metrics.json")


if __name__ == "__main__":
    main()
