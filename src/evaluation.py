import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple, List
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, mean_absolute_percentage_error
)

from src.config import Config
from src.logger import logger


class ModelEvaluator:

    def __init__(self):
        self.config = Config
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        logger.info("Initialized ModelEvaluator module")

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                          y_train_pred: np.ndarray = None,
                          y_train_true: np.ndarray = None) -> Dict[str, float]:
        try:
            logger.info("Calculating evaluation metrics...")

            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100

            evs = explained_variance_score(y_true, y_pred)

            residuals = y_true - y_pred
            residual_mean = residuals.mean()
            residual_std = residuals.std()

            bias = np.mean(residuals)

            nash_sutcliffe = 1 - (np.sum(residuals ** 2) / np.sum((y_true - y_true.mean()) ** 2))

            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'explained_variance': evs,
                'residual_mean': residual_mean,
                'residual_std': residual_std,
                'bias': bias,
                'nash_sutcliffe': nash_sutcliffe
            }

            if y_train_pred is not None and y_train_true is not None:
                train_r2 = r2_score(y_train_true, y_train_pred)
                train_mse = mean_squared_error(y_train_true, y_train_pred)
                metrics.update({
                    'train_r2': train_r2,
                    'train_mse': train_mse,
                    'overfitting_ratio': train_mse / mse if mse > 0 else 0
                })

            logger.info("Evaluation Metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.4f}")

            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   save_path: Path = None) -> Path:
        try:
            logger.info("Creating predictions vs actual plot...")

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            ax = axes[0, 0]
            ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5)

            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val],
                    'r--', lw=2, label='Perfect Prediction')

            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title('Predictions vs Actual Values')
            ax.legend()
            ax.grid(True, alpha=0.3)

            ax = axes[0, 1]
            residuals = y_true - y_pred
            ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='w', linewidth=0.5)
            ax.axhline(y=0, color='r', linestyle='--', lw=2)

            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title('Residuals vs Predicted Values')
            ax.grid(True, alpha=0.3)

            ax = axes[1, 0]
            ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--', lw=2)

            ax.set_xlabel('Residuals')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Residuals')
            ax.grid(True, alpha=0.3)

            ax = axes[1, 1]
            indices = np.arange(len(y_true))
            ax.plot(indices, y_true, 'b-', label='Actual', alpha=0.7, linewidth=2)
            ax.plot(indices, y_pred, 'r--', label='Predicted', alpha=0.7, linewidth=2)

            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Value')
            ax.set_title('Actual vs Predicted (Time Series)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            save_path = save_path or self.config.PLOTS_DIR / "predictions_vs_actual.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Plot saved to {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Error creating predictions plot: {str(e)}")
            raise

    def plot_learning_curves(self, train_sizes: np.ndarray,
                             train_scores: np.ndarray,
                             val_scores: np.ndarray,
                             save_path: Path = None) -> Path:
        try:
            logger.info("Creating learning curves plot...")

            plt.figure(figsize=(10, 6))

            plt.plot(train_sizes, train_scores.mean(axis=1), 'o-',
                     color='blue', label='Training Score', linewidth=2)
            plt.plot(train_sizes, val_scores.mean(axis=1), 's-',
                     color='green', label='Validation Score', linewidth=2)

            plt.fill_between(train_sizes,
                             train_scores.mean(axis=1) - train_scores.std(axis=1),
                             train_scores.mean(axis=1) + train_scores.std(axis=1),
                             alpha=0.1, color='blue')

            plt.fill_between(train_sizes,
                             val_scores.mean(axis=1) - val_scores.std(axis=1),
                             val_scores.mean(axis=1) + val_scores.std(axis=1),
                             alpha=0.1, color='green')

            plt.xlabel('Training Set Size')
            plt.ylabel('Score (R²)')
            plt.title('Learning Curves')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)

            save_path = save_path or self.config.PLOTS_DIR / "learning_curves.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Learning curves plot saved to {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Error creating learning curves: {str(e)}")
            raise

    def plot_feature_importance(self, model: Any, feature_names: List[str],
                                save_path: Path = None) -> Path:
        try:
            logger.info("Creating feature importance plot...")

            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
            else:
                logger.warning("Model doesn't have feature importances or coefficients")
                return None

            feature_importance_df = pd.DataFrame({
                'feature': feature_names[:len(importances)],
                'importance': importances
            }).sort_values('importance', ascending=True)

            plt.figure(figsize=(10, max(6, len(feature_names) * 0.3)))

            bars = plt.barh(feature_importance_df['feature'],
                            feature_importance_df['importance'])

            for bar in bars:
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height() / 2,
                         f'{width:.4f}', ha='left', va='center')

            plt.xlabel('Importance')
            plt.title('Feature Importance')
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()

            save_path = save_path or self.config.PLOTS_DIR / "feature_importance.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Feature importance plot saved to {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Error creating feature importance plot: {str(e)}")
            raise


    def create_performance_report(self, metrics: Dict[str, float],
                                  model_info: Dict[str, Any],
                                  save_path: Path = None) -> Path:
        try:
            logger.info("Creating performance report...")

            report = []
            report.append("=" * 60)
            report.append("MODEL PERFORMANCE REPORT")
            report.append("=" * 60)
            report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"Model Type: {model_info.get('model_type', 'Unknown')}")
            report.append(f"Input Shape: {model_info.get('input_shape', 'Unknown')}")
            report.append("")
            report.append("-" * 60)
            report.append("PERFORMANCE METRICS")
            report.append("-" * 60)

            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    report.append(f"{metric_name.upper():<25}: {value:.6f}")
                else:
                    report.append(f"{metric_name.upper():<25}: {value}")

            report.append("")
            report.append("-" * 60)
            report.append("INTERPRETATION")
            report.append("-" * 60)

            r2 = metrics.get('r2', 0)
            if r2 >= 0.9:
                report.append("R² ≥ 0.9: Excellent model fit")
            elif r2 >= 0.7:
                report.append("R² ≥ 0.7: Good model fit")
            elif r2 >= 0.5:
                report.append("R² ≥ 0.5: Moderate model fit")
            else:
                report.append("R² < 0.5: Poor model fit - consider model improvements")

            mape = metrics.get('mape', 100)
            if mape <= 10:
                report.append(f"MAPE = {mape:.1f}%: Highly accurate predictions")
            elif mape <= 20:
                report.append(f"MAPE = {mape:.1f}%: Good predictions")
            elif mape <= 50:
                report.append(f"MAPE = {mape:.1f}%: Reasonable predictions")
            else:
                report.append(f"MAPE = {mape:.1f}%: Poor predictions")

            report.append("=" * 60)
            report_text = "\n".join(report)

            save_path = save_path or self.config.OUTPUTS_DIR / "performance_report.txt"
            with open(save_path, 'w') as f:
                f.write(report_text)

            json_path = save_path.with_suffix('.json')
            json_report = {
                'metrics': metrics,
                'model_info': model_info,
                'generation_date': datetime.now().isoformat()
            }

            with open(json_path, 'w') as f:
                import json
                json.dump(json_report, f, indent=2)

            logger.info(f"Performance report saved to {save_path}")
            logger.info(f"JSON report saved to {json_path}")

            return save_path

        except Exception as e:
            logger.error(f"Error creating performance report: {str(e)}")
            raise


    def run_complete_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray,
                                model: Any = None, feature_names: List[str] = None,
                                y_train_true: np.ndarray = None,
                                y_train_pred: np.ndarray = None) -> Dict[str, Any]:
        try:
            logger.info("Starting complete evaluation pipeline...")

            metrics = self.calculate_metrics(y_true, y_pred, y_train_pred, y_train_true)

            plots = {}

            plot_path = self.plot_predictions_vs_actual(y_true, y_pred)
            plots['predictions_vs_actual'] = str(plot_path)

            if model is not None and feature_names is not None:
                plot_path = self.plot_feature_importance(model, feature_names)
                if plot_path:
                    plots['feature_importance'] = str(plot_path)

            model_info = {
                'model_type': type(model).__name__ if model else 'Unknown',
                'evaluation_date': datetime.now().isoformat(),
                'samples_evaluated': len(y_true)
            }

            report_path = self.create_performance_report(metrics, model_info)

            evaluation_summary = {
                'metrics': metrics,
                'plots': plots,
                'report_path': str(report_path),
                'model_info': model_info,
                'y_true_stats': {
                    'mean': float(y_true.mean()),
                    'std': float(y_true.std()),
                    'min': float(y_true.min()),
                    'max': float(y_true.max())
                },
                'y_pred_stats': {
                    'mean': float(y_pred.mean()),
                    'std': float(y_pred.std()),
                    'min': float(y_pred.min()),
                    'max': float(y_pred.max())
                }
            }

            summary_path = self.config.OUTPUTS_DIR / "evaluation_summary.json"
            with open(summary_path, 'w') as f:
                import json
                json.dump(evaluation_summary, f, indent=2)

            logger.info(f"Evaluation summary saved to {summary_path}")

            return evaluation_summary

        except Exception as e:
            logger.error(f"Error in complete evaluation: {str(e)}")
            raise