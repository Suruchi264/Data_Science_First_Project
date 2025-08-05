import os
import sys
import pickle
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


# ---------------------------------------------------------------------------
# Save any Python object to disk
# ---------------------------------------------------------------------------
def save_object(file_path: str, obj) -> None:
    """
    Persist a Python object to disk using pickle.

    Parameters
    ----------
    file_path : str
        Destination path (including filename) for the pickle file.
    obj : Any
        Python object to be pickled.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


# ---------------------------------------------------------------------------
# Evaluate candidate models with grid-search and return test-set scores
# ---------------------------------------------------------------------------
def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: Dict[str, object],
    param: Dict[str, Dict],
) -> Dict[str, float]:
    """
    Train and evaluate multiple models using GridSearchCV.

    Returns a dictionary: {model_name: test_r2_score}
    """
    try:
        report = {}

        for name, model in models.items():
            gs = GridSearchCV(model, param[name], cv=3)
            gs.fit(X_train, y_train)

            # Re-train best estimator on full training data
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            test_score = r2_score(y_test, y_test_pred)

            report[name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


# ---------------------------------------------------------------------------
# Load a pickled object from disk (safer with existence check)
# ---------------------------------------------------------------------------
def load_object(file_path: str):
    """
    Load a pickled object from disk. Raises a clear error if the
    file does not exist.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
