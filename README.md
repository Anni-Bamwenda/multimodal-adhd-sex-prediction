# Multimodal Prediction of ADHD Outcome and Sex from Neuroimaging Data

This repository implements a **multimodal machine learning pipeline** for predicting two targets:

- **ADHD_Outcome** (binary)
- **Sex_F** (binary)

using a combination of:
- categorical metadata
- quantitative clinical metadata
- functional connectome features

The project emphasizes **engineering rigor, reproducibility, and honest evaluation**,
following best practices used in research and production ML systems.

---

## 🔍 Problem Setting

The dataset consists of **heterogeneous modalities**:

- **Categorical metadata** (demographics, clinical flags)
- **Quantitative metadata** (numerical measures)
- **Functional connectome matrices** (high-dimensional neuroimaging features)

The task is **multi-output classification**:
each subject has *two correlated labels* that must be predicted jointly.

---

## 🧠 Core Design Principles

### 1. Strict Separation of Concerns

Each script has a single responsibility:

| Script | Responsibility |
|------|----------------|
| `preprocess.py` | Load, clean, merge modalities, and save numeric tensors |
| `feature_select.py` | Feature-selection utilities (no I/O, no splitting) |
| `train_model.py` | Train/validation split, feature selection, model training |
| `evaluate.py` | Validation metrics + test-set prediction only |

This prevents hidden coupling and makes the pipeline auditable.

---

### 2. Leakage-Free Evaluation (Critical)

Any operation that **learns from labels** is restricted to the **training split only**:

- Feature selection
- Model fitting

Validation data is treated as *future, unseen data*.
This ensures that reported metrics are **honest and defensible**.

---

### 3. Model-Driven Feature Selection

Feature selection is performed using a MultiOutputClassifier(RandomForestClassifier):


Feature importance is:
- computed per output
- aggregated across outputs
- thresholded by percentile

This aligns feature selection with the **inductive bias of the final model**.

---

### 4. Binary Artifacts for ML Stages

Intermediate ML artifacts are stored as:

- `.npy` arrays for features and labels
- `.json` for feature names and selected features

This avoids:
- dtype ambiguity
- CSV parsing errors
- floating-point drift

---

## 📂 Repository Structure
**Please note that the data folder has not been pushed to bitbucket for optimal performance reasons.
You can access TRAIN and TEST data [here](https://www.kaggle.com/competitions/widsdatathon2025/data)
```
multimodal-adhd-sex-prediction/
├── data/                    
│   ├── solution_template.csv
│   ├── processed\
│   │   ├── eval \
│   │   ├── model\
│   │   ├── feature_names.json
│   │   ├── X_test.npy
│   │   ├── X_train.npy
│   │   ├── y_train.npy
│   ├── TEST
│   └── TRAIN
├── src/                    
│   ├── preprocess.py
│   ├── feature_select.py
│   ├── train_model.py
│   └── evaluate.py
├── requirements.txt
├── MODEL_CARD.md
└── README.md
```

---

## ⚙️ How to Run the Pipeline

### 1. Preprocessing
```bash
python src/preprocess.py \
  --data_dir data \
  --output_dir data/processed
```

### 2. Training
```bash
python src/train_model.py \
  --data-dir data/processed \
  --output-dir data/processed/model \
```

### 3. Evaluating
```bash
python src/evaluate.py \
  --data-dir data/processed \
  --model-dir data/processed/model \
  --output-dir data/processed/eval
```

## 📊 Evaluation

Validation metrics reported per target:
Accuracy
Precision
Recall
F1
ROC-AUC

No aggregate score is used, as this can obscure task-specific performance.


## 📜 Disclaimer

This project is for research and educational purposes only.
It is not a medical diagnostic tool.

## 👩🏽‍💻 Author

Anni Bamwenda

Software Engineer II • Data Scientist • AI/ML Engineer

🔗 LinkedIn https://www.linkedin.com/in/annibamwenda/

🔗 GitHub: https://github.com/Anni-Bamwenda
