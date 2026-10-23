# TRACELESS AI 🛡️

## Self-Auditing Machine Learning Reliability System

TRACELESS AI is a machine learning reliability monitoring prototype that detects distribution drift, estimates prediction-error risk, and decides when predictions should be accepted, abstained from, or escalated for investigation.

### Key Features
- Random Forest classification
- Distribution drift detection using PSI and Kolmogorov–Smirnov (KS) tests
- Secondary model for prediction-error risk estimation
- Selective prediction: ACCEPT, ABSTAIN, and ESCALATE
- Interactive Streamlit dashboard
- Simulated distribution-shift testing

### Technology Stack
Python, Pandas, NumPy, Scikit-learn, SciPy, Streamlit, Matplotlib, and Joblib.

### Initial Evaluation Results

| Metric | Result |
|---|---:|
| Baseline accuracy | 91.23% |
| Baseline macro F1 | 0.9082 |
| Error-risk AUROC | 0.9707 |
| Error-risk AUPRC | 0.7537 |

These are initial results from a single experimental split. Further validation is needed before drawing reliability conclusions.

### Run Locally

Install dependencies:

```bash
pip install pandas numpy scikit-learn scipy streamlit matplotlib joblib
```

Run the scripts in order:

```bash
python baseline.py
python drift_monitor.py
python risk_estimator.py
python -m streamlit run app.py
```

### Limitations
- Uses a relatively small benchmark dataset.
- Simulated drift does not establish real-world performance.
- Drift does not necessarily mean predictions are incorrect.
- Risk thresholds require further validation.
- This is an educational research prototype, not a medical diagnostic system.

### Future Improvements
- Test on more datasets and random seeds.
- Compare error-risk estimation against confidence-only baselines.
- Improve probability calibration.
- Add delayed ground-truth feedback and a human-review workflow.
