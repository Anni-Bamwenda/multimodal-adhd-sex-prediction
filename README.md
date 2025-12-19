# Multimodal Prediction of ADHD Outcome and Sex

**TL;DR**  
This repository implements a **leakage-free, multimodal machine learning pipeline**
for **multi-output classification** of:

- **ADHD_Outcome** (binary)
- **Sex_F** (binary)

using heterogeneous data sources including categorical metadata, quantitative clinical features,
and high-dimensional functional connectome matrices.

The project emphasizes **engineering rigor, reproducibility, and honest evaluation**, following
best practices used in modern research and production ML systems.

> This work was developed as part of the **WiDS Datathon 2025**.  
> Competition details are available [here](https://www.kaggle.com/competitions/widsdatathon2025/overview).

---
## 🔍 Problem Setting

The dataset contains three distinct modalities:

- **Categorical metadata** — demographics and clinical flags
- **Quantitative metadata** — numerical clinical measurements
- **Functional connectome matrices** — high-dimensional neuroimaging features

The task is **multi-output classification**, where each subject has *two correlated targets*
that are predicted jointly.

Rather than treating these as independent problems, this project models them together to
capture shared structure and correlations across tasks

---

## 🧠 Core Design Principles

### 1. Strict Separation of Concerns

Each script has a single, clearly defined responsibility:

| Script | Responsibility |
|------|----------------|
| `preprocess.py` | Load raw data, clean, merge modalities, and save numeric tensors |
| `feature_select.py` | Feature-selection utilities (pure functions, no I/O, no splitting) |
| `train_model.py` | Train/validation split, feature selection(train only), model training |
| `evaluate.py` | Validation metrics + test-set prediction only |

This structure makes the pipeline **auditable, testable, and resistant to hidden coupling**.

---

### 2. Leakage-Free Evaluation (Critical)

Any operation that **learns from labels** is restricted to the **training split only**:

- Feature selection
- Model fitting

Validation data is treated as *future, unseen data*.
This prevents optimistic bias and ensures reported metrics are **honest and defensible**.

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
- accidental index misalignment

---

## 📂 Repository Structure
**Please note that Raw TRAIN and TEST data are **not included** in this repository due to size constraints. 
They can be downloaded [here](https://www.kaggle.com/competitions/widsdatathon2025/data) directly from Kaggle.
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

No aggregate metrics are used, as they can obscure task-specific performance and mask failure modes.


## 📜 Disclaimer

This project is for research and educational purposes only.
It is not a medical diagnostic tool.

## 👩🏽‍💻 Author

Anni Bamwenda

Software Engineer II • Data Scientist • AI/ML Engineer

🔗 LinkedIn https://www.linkedin.com/in/annibamwenda/

🔗 GitHub: https://github.com/Anni-Bamwenda
