
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

RANDOM_STATE = 42
RISK_THRESHOLD = 0.35
CONFIDENCE_THRESHOLD = 1.0 - RISK_THRESHOLD


def build_risk_features(model, reference, inputs):
    """Create signals that may help predict primary-model errors."""
    inputs = inputs[reference.columns].copy()

    # Primary model outputs
    probabilities = model.predict_proba(inputs)
    predictions = model.predict(inputs)

    confidence = probabilities.max(axis=1)
    sorted_probs = np.sort(probabilities, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]

    # Normalized prediction entropy: 0 = certain, 1 = maximally uncertain
    safe_probs = np.clip(probabilities, 1e-12, 1.0)
    entropy = -np.sum(safe_probs * np.log(safe_probs), axis=1)
    entropy /= np.log(probabilities.shape[1])

    # Distance from the reference distribution, measured feature by feature.
    means = reference.mean()
    stds = reference.std().replace(0, 1.0).fillna(1.0)

    z_scores = (inputs - means) / stds
    abs_z = z_scores.abs()

    risk_features = pd.DataFrame({
        "confidence": confidence,
        "prediction_entropy": entropy,
        "prediction_margin": margin,
        "mean_reference_distance": abs_z.mean(axis=1),
        "max_reference_distance": abs_z.max(axis=1),
        "missing_fraction": inputs.isna().mean(axis=1),
    }, index=inputs.index)

    return risk_features, predictions


def evaluate_risk(name, y_true, predictions, probabilities, risk_scores):
    """Evaluate risk ranking and abstention performance."""
    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)
    risk_scores = np.asarray(risk_scores)

    errors = (predictions != y_true).astype(int)

    metrics = {
        "test_samples": int(len(y_true)),
        "primary_model_errors": int(errors.sum()),
        "primary_model_error_rate": float(errors.mean()),
        "error_risk_auroc": None,
        "error_risk_auprc": None,
        "raw_confidence_auroc": None,
        "risk_model_coverage": None,
        "risk_model_selective_risk": None,
        "confidence_baseline_coverage": None,
        "confidence_baseline_selective_risk": None,
    }

    # Ranking metrics require both correct and incorrect predictions.
    if len(np.unique(errors)) == 2:
        metrics["error_risk_auroc"] = float(
            roc_auc_score(errors, risk_scores)
        )
        metrics["error_risk_auprc"] = float(
            average_precision_score(errors, risk_scores)
        )

        raw_risk = 1.0 - probabilities.max(axis=1)
        metrics["raw_confidence_auroc"] = float(
            roc_auc_score(errors, raw_risk)
        )

    # Selective prediction: accept only cases below the risk threshold.
    accepted = risk_scores <= RISK_THRESHOLD
    metrics["risk_model_coverage"] = float(accepted.mean())

    if accepted.any():
        metrics["risk_model_selective_risk"] = float(
            errors[accepted].mean()
        )

    # Baseline: accept cases with sufficiently high raw confidence.
    confidence_accepted = (
        probabilities.max(axis=1) >= CONFIDENCE_THRESHOLD
    )
    metrics["confidence_baseline_coverage"] = float(
        confidence_accepted.mean()
    )

    if confidence_accepted.any():
        metrics["confidence_baseline_selective_risk"] = float(
            errors[confidence_accepted].mean()
        )

    print(f"\n=== {name} ===")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    return metrics


def main():
    model_path = ARTIFACTS_DIR / "baseline_model.joblib"
    reference_path = ARTIFACTS_DIR / "reference_data.csv"
    risk_path = ARTIFACTS_DIR / "risk_training_data.csv"
    test_path = ARTIFACTS_DIR / "clean_test_data.csv"

    required = [model_path, reference_path, risk_path, test_path]
    missing = [str(path) for path in required if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing artifacts. Run baseline.py first:\n"
            + "\n".join(missing)
        )

    saved = joblib.load(model_path)
    model = saved["model"]

    reference = pd.read_csv(reference_path)
    risk_data = pd.read_csv(risk_path)
    test_data = pd.read_csv(test_path)

    feature_names = saved["feature_names"]

    X_risk = risk_data[feature_names]
    y_risk = risk_data["target"].to_numpy()

    X_test = test_data[feature_names]
    y_test = test_data["target"].to_numpy()

    # The risk model learns whether the primary model is correct.
    risk_features, risk_predictions = build_risk_features(
        model, reference, X_risk
    )
    risk_errors = (risk_predictions != y_risk).astype(int)

    print("Risk-training examples:", len(X_risk))
    print("Incorrect primary predictions:", int(risk_errors.sum()))

    if len(np.unique(risk_errors)) < 2:
        raise ValueError(
            "Risk-training data contains only one error class. "
            "More examples or a different split are needed."
        )

    # Keep some risk-training examples aside for diagnostics.
    # The final evaluation remains on the independent clean test set.
    X_meta_train, X_meta_valid, y_meta_train, y_meta_valid = (
        train_test_split(
            risk_features,
            risk_errors,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=risk_errors,
        )
    )

    risk_model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    risk_model.fit(X_meta_train, y_meta_train)

    # Validation diagnostic: validation examples were not used to fit
    # the risk estimator. No threshold is tuned on the final test set.
    valid_risk = risk_model.predict_proba(X_meta_valid)[:, 1]

    print("\n=== RISK-MODEL VALIDATION DIAGNOSTIC ===")
    print("Validation examples:", len(X_meta_valid))
    print("Validation error cases:", int(y_meta_valid.sum()))

    if len(np.unique(y_meta_valid)) == 2:
        print(
            "Validation error-risk AUROC:",
            round(roc_auc_score(y_meta_valid, valid_risk), 4),
        )
    else:
        print("Validation AUROC unavailable: only one error class.")

    # Refit on the full risk-training partition for final independent test.
    risk_model.fit(risk_features, risk_errors)

    # Evaluate primary and secondary models on untouched clean test data.
    test_features, test_predictions = build_risk_features(
        model, reference, X_test
    )
    test_probabilities = model.predict_proba(X_test)
    risk_scores = risk_model.predict_proba(test_features)[:, 1]

    metrics = evaluate_risk(
        "INDEPENDENT CLEAN TEST",
        y_test,
        test_predictions,
        test_probabilities,
        risk_scores,
    )

    # Save predictions so the dashboard can inspect individual cases.
    output = test_features.copy()
    output["actual_label"] = y_test
    output["predicted_label"] = test_predictions
    output["estimated_error_risk"] = risk_scores
    output["raw_confidence"] = test_probabilities.max(axis=1)
    output["actual_error"] = (
        test_predictions != y_test
    ).astype(int)

    output["decision"] = np.where(
        risk_scores <= RISK_THRESHOLD,
        "ACCEPT",
        "ABSTAIN",
    )

    output.to_csv(
        ARTIFACTS_DIR / "risk_test_predictions.csv", index=False
    )

    joblib.dump(
        {
            "model": risk_model,
            "feature_names": list(risk_features.columns),
            "risk_threshold": RISK_THRESHOLD,
        },
        ARTIFACTS_DIR / "risk_model.joblib",
    )

    with open(ARTIFACTS_DIR / "risk_metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nSaved:")
    print("- artifacts/risk_model.joblib")
    print("- artifacts/risk_test_predictions.csv")
    print("- artifacts/risk_metrics.json")
    print("\nError-risk estimation experiment completed.")


if __name__ == "__main__":
    main()
