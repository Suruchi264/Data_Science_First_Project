"""
model_trainer.py

• Trains multiple regression models, picks the best one, and
  now ALSO persists the fitted pre-processing object so that
  `predict_pipeline.py` can load both `model.pkl` and
  `preprocessor.pkl` at inference time.
"""

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


# ──────────────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


# ──────────────────────────────────────────────────────────────────────────────
#  Model-Trainer Class
# ──────────────────────────────────────────────────────────────────────────────
class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        train_array,
        test_array,
        preprocessor_obj: Any,  # NEW: fitted preprocessor passed in
    ) -> float:
        """
        Trains and selects the best regression model, then saves both the
        model and the pre-processor to disk so that the prediction pipeline
        can reload them later.
        """
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # -----------------------------------------------------------------
            # Candidate models + hyper-parameter grids
            # -----------------------------------------------------------------
            models: Dict[str, Any] = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params: Dict[str, Dict] = {
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                },
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            # -----------------------------------------------------------------
            # Train/evaluate each model
            # -----------------------------------------------------------------
            model_report: Dict[str, float] = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            # Identify best model
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No sufficiently accurate model found")

            logging.info(f"Best model: {best_model_name}  (R² = {best_model_score:.3f})")

            # -----------------------------------------------------------------
            # Persist BOTH model and preprocessor
            # -----------------------------------------------------------------
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            save_object(
                file_path=self.model_trainer_config.preprocessor_obj_file_path,
                obj=preprocessor_obj,
            )

            # Final test-set R²
            y_pred = best_model.predict(X_test)
            r2_sq = r2_score(y_test, y_pred)
            return r2_sq

        except Exception as e:
            raise CustomException(e, sys) from e
