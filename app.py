
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from drift_monitor import detect_feature_drift, create_shifted_data
from risk_estimator import build_risk_features


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

st.set_page_config(
    page_title="TRACELESS AI",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    primary = joblib.load(ARTIFACTS_DIR / "baseline_model.joblib")
    risk = joblib.load(ARTIFACTS_DIR / "risk_model.joblib")

    reference = pd.read_csv(ARTIFACTS_DIR / "reference_data.csv")
    test_data = pd.read_csv(ARTIFACTS_DIR / "clean_test_data.csv")

    with open(ARTIFACTS_DIR / "baseline_metrics.json") as file:
        baseline_metrics = json.load(file)

    with open(ARTIFACTS_DIR / "risk_metrics.json") as file:
        risk_metrics = json.load(file)

    return primary, risk, reference, test_data, baseline_metrics, risk_metrics


st.title("🛡️ TRACELESS AI")
st.subheader("Self-Auditing Machine Learning Reliability System")
st.caption(
    "Monitor distribution drift, estimate prediction-error risk, "
    "and decide when a prediction should be accepted or reviewed."
)

try:
    (
        primary_artifact,
        risk_artifact,
        reference,
        test_data,
        baseline_metrics,
        risk_metrics,
    ) = load_artifacts()
except FileNotFoundError:
    st.error(
        "Required model artifacts are missing. "
        "Run baseline.py and risk_estimator.py first."
    )
    st.stop()

primary_model = primary_artifact["model"]
risk_model = risk_artifact["model"]
feature_names = primary_artifact["feature_names"]
target_names = primary_artifact["target_names"]
risk_threshold = risk_artifact["risk_threshold"]

clean_features = test_data[feature_names].copy()

st.markdown("### 1. Select an evaluation scenario")

scenario = st.radio(
    "Incoming data",
    ["Clean test data", "Simulate distribution shift"],
    horizontal=True,
)

sample_count = st.slider(
    "Number of incoming samples",
    min_value=20,
    max_value=len(clean_features),
    value=len(clean_features),
    step=1,
)

batch = clean_features.iloc[:sample_count].copy()

if scenario == "Simulate distribution shift":
    batch = create_shifted_data(reference, batch)

# 2. Detect batch-level feature drift.
drift_report = detect_feature_drift(reference, batch)
flagged_count = int(drift_report["drift_detected"].sum())

# Prototype rule: three or more flagged features trigger escalation.
# This threshold must be validated before production use.
severe_drift = flagged_count >= 3

# 3. Generate primary predictions and secondary risk estimates.
risk_features, predictions = build_risk_features(
    primary_model, reference, batch
)
probabilities = primary_model.predict_proba(batch)
risk_scores = risk_model.predict_proba(risk_features)[:, 1]

# 4. Apply the decision policy.
if severe_drift:
    decisions = ["ESCALATE"] * len(batch)
else:
    decisions = [
        "ACCEPT" if score <= risk_threshold else "ABSTAIN"
        for score in risk_scores
    ]

st.markdown("### 2. Reliability overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Samples analyzed", len(batch))
col2.metric("Features flagged", f"{flagged_count} / {len(feature_names)}")
col3.metric("Mean estimated error risk", f"{risk_scores.mean():.1%}")
col4.metric(
    "Batch status",
    "ESCALATE" if severe_drift else "NORMAL MONITORING",
)

if severe_drift:
    st.error(
        "Severe drift rule triggered. Predictions are escalated "
        "for investigation rather than automatically trusted."
    )
else:
    st.info(
        "No severe batch-level drift trigger. Individual predictions "
        "are evaluated using the estimated error-risk threshold."
    )

st.markdown("### 3. Prediction decisions")

decision_counts = pd.Series(decisions).value_counts()

count_cols = st.columns(3)
for col, decision in zip(count_cols, ["ACCEPT", "ABSTAIN", "ESCALATE"]):
    col.metric(decision, int(decision_counts.get(decision, 0)))

results = pd.DataFrame({
    "sample_id": range(1, len(batch) + 1),
    "predicted_class": [
        target_names[int(prediction)] for prediction in predictions
    ],
    "model_confidence": probabilities.max(axis=1),
    "estimated_error_risk": risk_scores,
    "decision": decisions,
})

st.dataframe(
    results.style.format({
        "model_confidence": "{:.1%}",
        "estimated_error_risk": "{:.1%}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download prediction report",
    data=results.to_csv(index=False).encode("utf-8"),
    file_name="traceless_prediction_report.csv",
    mime="text/csv",
)

st.markdown("### 4. Feature drift analysis")

chart_data = drift_report.sort_values("psi", ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 4))
ax.barh(chart_data["feature"], chart_data["psi"])
ax.axvline(
    0.20,
    linestyle="--",
    label="Prototype PSI threshold",
)
ax.set_xlabel("Population Stability Index (PSI)")
ax.set_title("Top 10 feature distribution changes")
ax.invert_yaxis()
ax.legend()
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.dataframe(
    drift_report,
    use_container_width=True,
    hide_index=True,
)

st.markdown("### 5. Baseline and risk-model evaluation")

metric_cols = st.columns(4)
metric_cols[0].metric(
    "Baseline accuracy", f"{baseline_metrics['accuracy']:.2%}"
)
metric_cols[1].metric(
    "Baseline macro F1", f"{baseline_metrics['macro_f1']:.4f}"
)

auroc = risk_metrics["error_risk_auroc"]
auprc = risk_metrics["error_risk_auprc"]

metric_cols[2].metric(
    "Error-risk AUROC",
    f"{auroc:.4f}" if auroc is not None else "Unavailable",
)
metric_cols[3].metric(
    "Error-risk AUPRC",
    f"{auprc:.4f}" if auprc is not None else "Unavailable",
)

st.caption(
    "Evaluation metrics were measured on the saved clean test set. "
    "Simulated shifts are stress tests; their predictions must not be "
    "treated as ground-truth accuracy measurements."
)

with st.expander("Research notes and limitations"):
    st.write(
        """
        - PSI and KS identify distribution changes, not incorrect predictions.
        - The error-risk estimator was trained on a small sample.
        - The risk threshold and severe-drift rule are prototype settings.
        - This dashboard does not yet implement delayed ground-truth feedback,
          human-review workflow, or production-grade data-quality checks.
        - Validate thresholds across multiple seeds and independent shifts
          before making reliability claims.
        """
    )
