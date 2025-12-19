


"""

src/feature_select.py

Pure feature-selection utilities.
NO file I/O.
NO dataset splitting.

Will be used as a module by train_model.py

Author: Anni Bamwenda
"""

import numpy as np
from typing import List
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier


def fit_feature_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    n_estimators: int = 300,
    random_state: int = 42,
    percentile: float = 50,
) -> List[str]:
    """
    Fit MultiOutput RandomForest on TRAIN data only and return selected features.
    """

    base_rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )

    model = MultiOutputClassifier(base_rf)
    model.fit(X_train, y_train)

    importances = np.mean(
        [est.feature_importances_ for est in model.estimators_],
        axis=0,
    )

    threshold = np.percentile(importances, percentile)

    selected_features = [
        f for f, imp in zip(feature_names, importances) if imp >= threshold
    ]

    return selected_features


def apply_feature_selector(
    X: np.ndarray,
    feature_names: List[str],
    selected_features: List[str],
) -> np.ndarray:
    idx = [feature_names.index(f) for f in selected_features]
    return X[:, idx]
