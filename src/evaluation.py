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

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        try:
            logger.info("Calculating evaluation metrics...")

            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            metrics = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   save_path: Path = None) -> Path:
        try:
            logger.info("Creating predictions vs actual plot...")

            plt.figure(figsize=(8, 6))
            plt.scatter(y_true, y_pred, alpha=0.6)
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Prediction")
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title("Predictions vs Actual")
            plt.legend()
            plt.grid(True)

            save_path = save_path or self.config.OUTPUTS_DIR / "predictions_vs_actual.png"
            plt.savefig(save_path, dpi=300)
            plt.close()
            return save_path

        except Exception as e:
            logger.error(f"Error creating predictions plot: {str(e)}")
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


    def generate_report(self, metrics: Dict[str, float], save_path: Path = None) -> Path:
        lines = [
            "="*40,
            "MODEL PERFORMANCE REPORT",
            f"Generated: {datetime.now()}",
            "="*40
        ]
        for k, v in metrics.items():
            lines.append(f"{k.upper():<10}: {v:.4f}")
        report_text = "\n".join(lines)

        save_path = save_path or self.config.OUTPUTS_DIR / "performance_report.txt"
        with open(save_path, 'w') as f:
            f.write(report_text)
        return save_path


    def run_complete_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray,
                                model: Any = None, feature_names: List[str] = None) -> Dict[str, Any]:
        try:
            logger.info("Starting complete evaluation pipeline...")

            metrics = self.calculate_metrics(y_true, y_pred)

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

            evaluation_summary = {
                'metrics': metrics,
                'plots': plots,
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