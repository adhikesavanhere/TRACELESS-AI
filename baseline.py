
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# Project directories
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def expected_calibration_error(y_true, probabilities, bins=10):
    """Measure the gap between confidence and observed accuracy."""
    predictions = (probabilities >= 0.5).astype(int)
    confidence = np.maximum(probabilities, 1 - probabilities)
    correct = (predictions == y_true).astype(int)

    ece = 0.0
    bin_edges = np.linspace(0, 1, bins + 1)

    for i in range(bins):
        if i == bins - 1:
            mask = (confidence >= bin_edges[i]) & (
                confidence <= bin_edges[i + 1]
            )
        else:
            mask = (confidence >= bin_edges[i]) & (
                confidence < bin_edges[i + 1]
            )

        if mask.any():
            bin_accuracy = correct[mask].mean()
            bin_confidence = confidence[mask].mean()
            ece += mask.mean() * abs(bin_accuracy - bin_confidence)

    return float(ece)


def main():
    # 1. Load a reproducible benchmark dataset
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target

    print(f"Dataset samples: {len(X)}")
    print(f"Features: {X.shape[1]}")

    # 2. Create separate training, risk-training, and test partitions
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_risk, X_test, y_risk, y_test = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_remaining,
    )

    # 3. Train the primary prediction model
    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 4. Evaluate on the untouched clean test partition
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "ece": expected_calibration_error(y_test.to_numpy(), probabilities),
    }

    print("\n=== TRACELESS AI: Baseline Evaluation ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=dataset.target_names,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    # 5. Save model and data for the next implementation stages
    joblib.dump(
        {
            "model": model,
            "feature_names": list(X.columns),
            "target_names": list(dataset.target_names),
            "positive_class": 1,
        },
        ARTIFACTS_DIR / "baseline_model.joblib",
    )

    # Reference distribution for future drift comparisons
    X_train.to_csv(ARTIFACTS_DIR / "reference_data.csv", index=False)

    # Separate pool for training the error-risk estimator in a later stage
    risk_pool = X_risk.copy()
    risk_pool["target"] = y_risk
    risk_pool.to_csv(ARTIFACTS_DIR / "risk_training_data.csv", index=False)

    # Held-out test data for subsequent independent evaluation
    test_data = X_test.copy()
    test_data["target"] = y_test
    test_data.to_csv(ARTIFACTS_DIR / "clean_test_data.csv", index=False)

    with open(ARTIFACTS_DIR / "baseline_metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nArtifacts saved in:", ARTIFACTS_DIR)
    print("Baseline model training completed successfully.")


if __name__ == "__main__":
    main()
