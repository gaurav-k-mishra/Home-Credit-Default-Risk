# Home Credit Default Risk — Predictive Modeling

A machine learning project predicting loan default probability for Home Credit, built as part of IS 6812 (Spring 2025, David Eccles School of Business). Submitted to the Kaggle Home Credit Default Risk competition.

**Team:** Amiya Ranjan Sahoo · Gaurav Mishra · Jeffrey Zhang · Rob Sheehy

---

## Project Overview

Lenders face a persistent challenge: identifying which applicants are likely to default before a loan is approved. Misclassifying risky borrowers leads to avoidable financial losses. This project builds a binary classification pipeline that scores each applicant with a default probability (0–1) and maps that score to an automated loan decision — Approve, Review, or Decline — reducing losses while improving approval quality.

### What the model supports

- **Default probability scoring** — risk score from 0 (low risk) to 1 (high risk) per applicant
- **Automated loan decisions** — threshold-based Approve / Review / Decline routing
- **Feature engineering** — financial ratios, credit history aggregation, and employment/age conversions
- **Multi-model comparison** — XGBoost, Random Forest, Logistic Regression, and CART evaluated head-to-head
- **Class imbalance handling** — 11× class weighting to prevent the model from ignoring the minority default class
- **Business impact quantification** — projected annual savings and default reduction estimates

---

## Business Results

| Metric | Before Model | After Model |
|---|---|---|
| Default Rate | 8.1% | 2.9% |
| Annual Defaults | 7,447 | 2,640 |
| Defaults Prevented | — | 4,807 |
| Annual Savings | — | $38.5M |
| Loss Reduction | — | 65% |

---

## Modeling Approach

| Stage | Description |
|---|---|
| Data ingestion | Applicant demographics, financial history, payment behavior, and bureau records |
| Feature engineering | Days → years conversion, financial ratio creation, bureau history aggregation, 11× class weighting |
| Model selection | 4 models tested with 5-fold cross-validation; AUC used as primary evaluation metric |
| Best model | XGBoost — AUC 0.754 (cross-validated), 0.753 (validation), 0.742 (Kaggle public score) |
| Decision thresholds | Low score → Approve · Medium score → Review · High score → Decline |
| Deployment design | Real-time API scoring integrated into loan application workflow with annual retraining |

---

## Model Performance

| Model | AUC Score |
|---|---|
| XGBoost | 0.754 |
| Logistic Regression | 0.739 |
| Random Forest | 0.727 |
| CART | 0.592 |

---

## Key Drivers of Default Risk

- **Financial ratio** — debt-to-income and repayment capacity measures
- **Credit history** — past delinquencies and bureau-aggregated long-term behavior
- **Employment length** — longer tenure associated with lower default risk
- **Applicant age** — influences financial responsibility patterns

---

## Tech Stack

- **Language:** Python
- **ML libraries:** XGBoost, scikit-learn (Random Forest, Logistic Regression, CART)
- **Data:** pandas, NumPy
- **Validation:** 5-fold cross-validation
- **Competition:** Kaggle — Home Credit Default Risk
- **Environment:** Jupyter Notebook

---

## Key Highlights

- XGBoost achieved a Kaggle public score of 0.742, placing in the competitive range for this benchmark dataset
- 11× class weighting applied to counteract the heavily imbalanced default/non-default ratio
- Feature engineering pipeline converts raw days fields to interpretable years and constructs financial ratios
- Bureau history aggregated across multiple records to capture long-term credit behavior
- Decision threshold system aligned with business risk appetite — not just model accuracy
- Human oversight layer built into deployment design for borderline and high-risk cases
