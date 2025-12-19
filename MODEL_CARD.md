
---

# 📄 Model Card

```markdown
# Model Card: Multimodal ADHD & Sex Prediction Model

## Model Overview

This model predicts two binary outcomes:

- ADHD_Outcome
- Sex_F

using multimodal inputs including:
- categorical metadata
- quantitative metadata
- functional connectome features

The model is implemented as a **MultiOutput Random Forest classifier**.

---

## Intended Use

**Intended:**
- Research exploration
- Methodological benchmarking
- Educational demonstration of ML pipelines

**Not intended:**
- Clinical diagnosis
- Individual medical decision-making
- Deployment without further validation

---

## Training Data

- Subjects with complete multimodal data
- Labels available only in training split
- Validation data held out *prior to feature selection*

---

## Evaluation Procedure

- Train/validation split performed before any label-dependent operations
- Feature selection fit on training split only
- Validation metrics computed on unseen data
- Test set used only for final prediction

Metrics reported per label:
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

---

## Ethical Considerations

- **Sex prediction** is included as a modeling task, not as a normative claim.
- Potential bias may exist due to:
  - dataset imbalance
  - demographic skew
  - neuroimaging preprocessing artifacts

No fairness guarantees are claimed.

---

## Limitations

- Limited sample size relative to feature dimensionality
- Random Forest models may overfit high-dimensional connectome data
- No causal claims can be made
- Performance may not generalize across populations

---

## Recommendations

- Validate on external datasets
- Perform subgroup analysis
- Avoid deployment in sensitive or clinical contexts
- Treat results as exploratory

---

## Transparency

All preprocessing, feature selection, and evaluation steps are explicitly documented.
No hidden data leakage or implicit fitting steps are used.
