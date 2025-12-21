import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, Any
import pickle
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

    def __init__(self, model_type: str = None):
        self.config = Config
        self.model_type = model_type or self.config.MODEL_TYPE
        self.model = None
        self.model_params = {}
        self.training_metrics = {}
        logger.info(f"Initialized ModelTrainer with model type: {self.model_type}")


    def create_model(self) -> Any:
        try:
            logger.info(f"Creating {self.model_type} model...")

            if self.model_type == "linear":
                model = LinearRegression()
                params = {
                    'fit_intercept': [True, False],
                    'positive': [True, False]
                }

            elif self.model_type == "random_forest":
                model = RandomForestRegressor(random_state=self.config.RANDOM_STATE)
                params = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }

            elif self.model_type == "gradient_boosting":
                model = GradientBoostingRegressor(random_state=self.config.RANDOM_STATE)
                params = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }

            elif self.model_type == "svm":
                model = SVR()
                params = {
                    'C': [0.1, 1, 10],
                    'epsilon': [0.01, 0.1, 0.2],
                    'kernel': ['linear', 'rbf']
                }

            elif self.model_type == "neural_network":
                model = MLPRegressor(random_state=self.config.RANDOM_STATE, max_iter=1000)
                params = {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01]
                }

            else:
                logger.warning(f"Unknown model type: {self.model_type}. Using Random Forest.")
                model = RandomForestRegressor(random_state=self.config.RANDOM_STATE)
                params = {}

            self.model_params = params
            return model

        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            raise


    def train_model(self, X_train: np.ndarray, y_train: pd.Series,
                    hyperparameter_tuning: bool = True) -> Any:
        try:
            logger.info("Training model...")
            model = self.create_model()

            if hyperparameter_tuning and self.model_params:
                logger.info("Performing hyperparameter tuning...")

                grid_search = GridSearchCV(
                    model,
                    self.model_params,
                    cv=5,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=0
                )

                grid_search.fit(X_train, y_train)

                model = grid_search.best_estimator_
                best_params = grid_search.best_params_

                logger.info(f"Best parameters: {best_params}")
                logger.info(f"Best CV score: {-grid_search.best_score_:.4f}")

            else:
                model.fit(X_train, y_train)

            self.model = model
            logger.info(f"Model training completed: {type(model).__name__}")

            return model

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise


    def evaluate_model(self, X_test: np.ndarray, y_test: pd.Series,
                       X_train: np.ndarray = None, y_train: pd.Series = None) -> Dict[str, float]:
        try:
            logger.info("Evaluating model...")

            if self.model is None:
                raise ValueError("Model not trained. Call train_model first.")

            y_pred = self.model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1))) * 100

            train_metrics = {}
            if X_train is not None and y_train is not None:
                y_train_pred = self.model.predict(X_train)
                train_r2 = r2_score(y_train, y_train_pred)
                train_mse = mean_squared_error(y_train, y_train_pred)
                train_metrics = {
                    'train_r2': train_r2,
                    'train_mse': train_mse
                }

            self.training_metrics = {
                'test_mse': mse,
                'test_rmse': rmse,
                'test_mae': mae,
                'test_r2': r2,
                'test_mape': mape,
                **train_metrics
            }
            logger.info(f"Model Evaluation Metrics:")
            logger.info(f"  MSE: {mse:.4f}")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  MAE: {mae:.4f}")
            logger.info(f"  R²: {r2:.4f}")
            logger.info(f"  MAPE: {mape:.2f}%")

            if train_metrics:
                logger.info(f"  Train R²: {train_metrics['train_r2']:.4f}")

            return self.training_metrics

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise


    def cross_validate(self, X: np.ndarray, y: pd.Series, cv: int = 5) -> Dict[str, Any]:
        try:
            logger.info(f"Performing {cv}-fold cross-validation...")

            if self.model is None:
                model = self.create_model()
            else:
                model = self.model

            cv_scores = cross_val_score(
                model, X, y,
                cv=cv,
                scoring='r2',
                n_jobs=-1
            )

            cv_results = {
                'cv_scores': cv_scores.tolist(),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_min': cv_scores.min(),
                'cv_max': cv_scores.max()
            }

            logger.info(f"Cross-validation R² scores: {cv_scores}")
            logger.info(f"Mean CV R²: {cv_results['cv_mean']:.4f} (±{cv_results['cv_std']:.4f})")

            return cv_results

        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            raise


    def save_model(self, model_path: Path = None) -> Path:
        try:
            if self.model is None:
                raise ValueError("No model to save. Train the model first.")

            model_path = model_path or self.config.MODEL_PKL_PATH

            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)

            metadata = {
                'model_type': self.model_type,
                'training_date': datetime.now().isoformat(),
                'metrics': self.training_metrics,
                'model_class': type(self.model).__name__
            }

            metadata_path = model_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Model saved to {model_path}")
            logger.info(f"Metadata saved to {metadata_path}")

            return model_path

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise


    def load_model(self, model_path: Path = None) -> Any:
        try:
            model_path = model_path or self.config.MODEL_PKL_PATH

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)

            metadata_path = model_path.with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded model metadata: {metadata}")

            logger.info(f"Model loaded from {model_path}")
            logger.info(f"Model type: {type(self.model).__name__}")

            return self.model

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise


    def run_training_pipeline(self, preprocessing_results: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info("Starting training pipeline...")

            X_train = preprocessing_results['X_train']
            X_test = preprocessing_results['X_test']
            y_train = preprocessing_results['y_train']
            y_test = preprocessing_results['y_test']

            model = self.train_model(X_train, y_train, hyperparameter_tuning=True)

            metrics = self.evaluate_model(X_test, y_test, X_train, y_train)

            X_full = np.vstack([X_train, X_test])
            y_full = pd.concat([y_train, y_test])
            cv_results = self.cross_validate(X_full, y_full, cv=5)

            model_path = self.save_model()

            return {
                'model': model,
                'metrics': metrics,
                'cv_results': cv_results,
                'model_path': model_path,
                'feature_columns': preprocessing_results.get('feature_columns', []),
                'target_column': preprocessing_results.get('target_column', '')
            }

        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            raise
