import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, Any
import json
from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

from src.config import Config
from src.logger import logger
class ModelTrainer:

    MODEL_REGISTRY = {
        "linear": (
            LinearRegression,
            {
                "fit_intercept": [True, False],
                "positive": [True, False]
            }
        ),
        "random_forest": (
            RandomForestRegressor,
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            }
        ),
        "gradient_boosting": (
            GradientBoostingRegressor,
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5]
            }
        ),
        "svm": (
            SVR,
            {
                "C": [0.1, 1, 10],
                "epsilon": [0.01, 0.1],
                "kernel": ["linear", "rbf"]
            }
        ),
        "neural_network": (
            MLPRegressor,
            {
                "hidden_layer_sizes": [(50,), (100,), (50, 50)],
                "activation": ["relu", "tanh"],
                "alpha": [0.0001, 0.001]
            }
        )
    }

    def __init__(self, model_type: str = None):
        self.config = Config
        self.model_type = model_type or self.config.MODEL_TYPE
        self.model = None
        self.model_params = {}
        self.training_metrics = {}
        logger.info(f"Initialized ModelTrainer ({self.model_type})")


    def create_model(self):
        model_cls, params = ModelTrainer.MODEL_REGISTRY.get(
            self.model_type,
            (RandomForestRegressor, {})
        )

        model = model_cls(random_state=self.config.RANDOM_STATE) \
            if "random_state" in model_cls().get_params() else model_cls()

        self.model_params = params
        return model


    def train_model(self, X, y, tune: bool = True):
        logger.info("Training model...")
        model = self.create_model()

        if tune and self.model_params:
            grid = GridSearchCV(
                model,
                self.model_params,
                cv=5,
                scoring="neg_mean_squared_error",
                n_jobs=-1
            )
            grid.fit(X, y)
            self.model = grid.best_estimator_
            logger.info(f"Best params: {grid.best_params_}")
        else:
            model.fit(X, y)
            self.model = model

        return self.model


    @staticmethod
    def _regression_metrics(y_true, y_pred) -> Dict[str, float]:
        mse = mean_squared_error(y_true, y_pred)
        return {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "mape": np.mean(np.abs((y_true - y_pred) /
                                   np.maximum(np.abs(y_true), 1))) * 100
        }


    def evaluate_model(self, X_test, y_test, X_train=None, y_train=None):
        if self.model is None:
            raise ValueError("Train model before evaluation.")

        test_preds = self.model.predict(X_test)
        metrics = self._regression_metrics(y_test, test_preds)

        if X_train is not None:
            train_preds = self.model.predict(X_train)
            metrics["train_r2"] = r2_score(y_train, train_preds)

        self.training_metrics = metrics
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def cross_validate(self, X, y, cv: int = 5):
        model = self.model or self.create_model()
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)

        return {
            "cv_scores": scores.tolist(),
            "cv_mean": scores.mean(),
            "cv_std": scores.std()
        }

    def run_training_pipeline(self, prep: Dict[str, Any]):
        x_train, X_test = prep["X_train"], prep["X_test"]
        y_train, y_test = prep["y_train"], prep["y_test"]

        model = self.train_model(x_train, y_train)
        metrics = self.evaluate_model(X_test, y_test, x_train, y_train)

        cv = self.cross_validate(
            np.vstack([x_train, X_test]),
            pd.concat([y_train, y_test])
        )

        return {
            "model": model,
            "metrics": metrics,
            "cv_results": cv,
            "feature_columns": prep.get("feature_columns", []),
            "target_column": prep.get("target_column", "")
        }
