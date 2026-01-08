# Multimodal Prediction of ADHD Outcome and Sex

## Overview
This project implements an end-to-end machine learning system for jointly predicting ADHD diagnosis and sex using heterogeneous data sources, including socio-demographic metadata and functional brain imaging features.  

The goal of this project is to understand which **brain activity patterns are associated with ADHD** and whether these patterns differ between males and females.  

The implementation of this project  emphasizes reliable pipeline design, evaluation tradeoffs, and reproducibility, following best practices in modern research and production ML systems.

> This work was developed as part of the **WiDS Datathon 2025**.  
> Competition details are available [here](https://www.kaggle.com/competitions/widsdatathon2025/overview).

---
## Problem Framing

The task is formulated as a multi-output classification problem, where two correlated targets must be predicted simultaneously:

- ADHD diagnosis (binary)
- Sex (binary)

Key challenges in this dataset include:

- Missing and noisy clinical metadata

- Class imbalance

- High-dimensional functional connectome features

- Heterogeneous statistical properties across modalities

These constraints require careful preprocessing and model selection to avoid overfitting and brittle results.

---
## System Overview

**High-level pipeline:**

1. Data ingestion and validation
2. Modality-specific preprocessing of:
    - Socio-demographic metadata 
    - Quantitative clinical features  
    - Functional connectome representations
3. Feature engineering and alignment
4. Multi-output model training
5. Evaluation and error analysis

Each stage is implemented as a modular component to support clarity, reproducibility, and future extensibility.

## Key Design Choices

**Model choice (Random Forest):**  
Random Forests were selected to balance interpretability, robustness to mixed feature types, and performance on limited sample sizes, while avoiding the overfitting risk of more complex architectures.

**Multi-output learning:**  
Joint prediction was used to capture correlations between ADHD diagnosis and sex within a single, unified training pipeline.

**Feature engineering strategy:**  
Preprocessing was applied per modality to avoid leaking assumptions across heterogeneous data sources and to preserve signal integrity.

---
## Tradeoffs and Failure Modes

- The current approach does not scale efficiently to very high-dimensional connectome representations without additional dimensionality reduction.
- Random Forests limit representational capacity compared to deep learning models, but were intentionally chosen to prioritize stability and interpretability.
- Model performance is sensitive to missing-data patterns, making robust preprocessing and validation critical.

These tradeoffs were accepted to ensure reliable baseline behavior under real-world data constraints.

---
## Future Improvements

- Introduce dimensionality reduction or learned embeddings for functional connectome features
- Add systematic model monitoring and data drift analysis
- Explore alternative architectures while preserving pipeline modularity and evaluation rigor

---
## Repository Structure
**Please note that Raw TRAIN and TEST data are **not included** in this repository due to size constraints. 
They can be downloaded [here](https://www.kaggle.com/competitions/widsdatathon2025/data).
```
multimodal-adhd-sex-prediction/
├── notebooks/          # Exploratory analysis and prototyping
├── src/                # Modular pipeline implementation
│   ├── preprocess.py
│   ├── feature_select.py
│   ├── train_model.py
│   └── evaluate.py
├── results/            # Model outputs and evaluation artifacts
├── MODEL_CARD.md
├── README.md
└── requirements.txt
```

---

## How to Run the Pipeline

#### 1. Preprocessing
```bash
python src/preprocess.py \
  --data_dir data \
  --output_dir data/processed
```

#### 2. Training
```bash
python src/train_model.py \
  --data-dir data/processed \
  --output-dir data/processed/model \
```

#### 3. Evaluating
```bash
python src/evaluate.py \
  --data-dir data/processed \
  --model-dir data/processed/model \
  --output-dir data/processed/eval
```

---
## Evaluation

Validation metrics reported per target:  
- Accuracy  
- Precision  
- Recall  
- F1  
- ROC-AUC  

No aggregate metrics are used, as they can obscure task-specific performance and mask failure modes.

---
## Key Takeaway

This project demonstrates how machine learning models fit into a larger system, where data quality, evaluation strategy, and maintainability matter as much as model choice.

---
## Disclaimer

This project is for research and educational purposes only.
It is not a medical diagnostic tool.

---
## Author

Anni Bamwenda  
Software Engineer II • AI/ML Engineer  
🔗[LinkedIn](https://www.linkedin.com/in/annibamwenda/)     👩🏾‍💻[GitHub](https://github.com/Anni-Bamwenda)
