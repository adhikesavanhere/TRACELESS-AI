
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

PSI_THRESHOLD = 0.20
KS_PVALUE_THRESHOLD = 0.05
EPSILON = 1e-6


def calculate_psi(reference, current, bins=10):
    """Calculate PSI using quantile bins defined by reference data."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) == 0 or len(current) == 0:
        return float("nan")

    # Define bins using the reference distribution only.
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))

    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    ref_pct = np.clip(ref_pct, EPSILON, None)
    cur_pct = np.clip(cur_pct, EPSILON, None)

    return float(
        np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    )


def detect_feature_drift(reference_df, current_df):
    """Compare each feature against its reference distribution."""
    results = []

    for feature in reference_df.columns:
        reference = reference_df[feature].dropna().to_numpy()
        current = current_df[feature].dropna().to_numpy()

        if len(reference) == 0 or len(current) == 0:
            continue

        psi = calculate_psi(reference, current)
        ks_result = ks_2samp(reference, current)

        # This is a simple prototype rule, not a calibrated alarm.
        drift_detected = (
            psi >= PSI_THRESHOLD
            or ks_result.pvalue < KS_PVALUE_THRESHOLD
        )

        results.append({
            "feature": feature,
            "psi": round(psi, 4),
            "ks_statistic": round(float(ks_result.statistic), 4),
            "ks_pvalue": float(ks_result.pvalue),
            "drift_detected": bool(drift_detected),
        })

    report = pd.DataFrame(results)

    return report.sort_values(
        ["drift_detected", "psi"],
        ascending=[False, False],
    ).reset_index(drop=True)


def create_shifted_data(reference_df, current_df):
    """Create a reproducible feature-shift stress test."""
    shifted = current_df.copy()

    # Deliberately shift two features by different amounts.
    radius_std = reference_df["mean radius"].std()
    texture_mean = reference_df["mean texture"].mean()

    shifted["mean radius"] = (
        shifted["mean radius"] + 1.5 * radius_std
    )
    shifted["mean texture"] = shifted["mean texture"] * 1.8

    print("\nInjected shifts:")
    print(f"mean radius: +{1.5:.1f} reference standard deviations")
    print(f"mean texture: multiplied by 1.8")

    return shifted


def summarize_report(name, report):
    detected = int(report["drift_detected"].sum())
    total = len(report)

    print(f"\n=== {name} ===")
    print(f"Features checked: {total}")
    print(f"Features flagged: {detected}")

    print("\nTop 10 features:")
    print(report.head(10).to_string(index=False))

    return {
        "scenario": name,
        "features_checked": total,
        "features_flagged": detected,
        "flagged_features": report.loc[
            report["drift_detected"], "feature"
        ].tolist(),
    }


def main():
    reference_path = ARTIFACTS_DIR / "reference_data.csv"
    test_path = ARTIFACTS_DIR / "clean_test_data.csv"

    if not reference_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Baseline artifacts are missing. Run baseline.py first."
        )

    reference = pd.read_csv(reference_path)
    test_data = pd.read_csv(test_path)

    # Remove the label: drift monitoring examines input features only.
    test_features = test_data.drop(columns=["target"])

    if list(reference.columns) != list(test_features.columns):
        raise ValueError("Reference and test feature columns do not match.")

    # Scenario A: normal held-out data
    clean_report = detect_feature_drift(reference, test_features)

    # Scenario B: simulated shift
    shifted_features = create_shifted_data(reference, test_features)
    shifted_report = detect_feature_drift(reference, shifted_features)

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    clean_report.to_csv(
        ARTIFACTS_DIR / "drift_report_clean.csv", index=False
    )
    shifted_report.to_csv(
        ARTIFACTS_DIR / "drift_report_shifted.csv", index=False
    )

    summaries = [
        summarize_report("CLEAN TEST DATA", clean_report),
        summarize_report("SIMULATED SHIFTED DATA", shifted_report),
    ]

    with open(ARTIFACTS_DIR / "drift_summary.json", "w") as file:
        json.dump(summaries, file, indent=4)

    print("\nReports saved in:", ARTIFACTS_DIR)
    print("Drift monitoring experiment completed.")


if __name__ == "__main__":
    main()
