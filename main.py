#!/usr/bin/env python3

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import Config
from src.logger import logger
from src.data_acquisition import DataAcquisition
from src.preprocessing import DataPreprocessor
from src.training import ModelTrainer
from src.export_onnx import ONNXExporter
from src.inference import ModelInference
from src.evaluation import ModelEvaluator


class IoTPipeline:
    """Main orchestrator for the IoT data pipeline"""

    def __init__(self):
        self.config = Config
        self.pipeline_results = {}
        logger.info("=" * 60)
        logger.info("IoT DATA PIPELINE - INITIALIZED")
        logger.info("=" * 60)

    def run_data_acquisition(self, generate_synthetic: bool = True,
                             download_real: bool = True) -> dict:
        """Step 1: Data Acquisition"""
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: DATA ACQUISITION")
        logger.info("=" * 60)

        try:
            data_acquirer = DataAcquisition()
            data_results = {}

            if generate_synthetic:
                synthetic_data = data_acquirer.generate_synthetic_data()
                synthetic_data.to_csv(self.config.SYNTHETIC_DATA_PATH, index=False)
                data_results['synthetic'] = {
                    'path': str(self.config.SYNTHETIC_DATA_PATH),
                    'shape': synthetic_data.shape,
                    'columns': synthetic_data.columns.tolist()
                }
                logger.info(f"Synthetic data generated: {synthetic_data.shape}")

            if download_real:
                real_data = data_acquirer.get_real_data(True)
                data_results['real'] = {
                    'path': str(self.config.REAL_DATA_PATH),
                    'shape': real_data.shape,
                    'columns': real_data.columns.tolist()
                }
                logger.info(f"Real data acquired: {real_data.shape}")

            self.pipeline_results['data_acquisition'] = data_results
            return data_results

        except Exception as e:
            logger.error(f"Data acquisition failed: {str(e)}")
            raise

    def run_preprocessing(self, use_synthetic: bool = True) -> dict:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: DATA PREPROCESSING")
        logger.info("=" * 60)

        try:
            preprocessor = DataPreprocessor()

            data_path = self.config.SYNTHETIC_DATA_PATH if use_synthetic else self.config.REAL_DATA_PATH

            preprocessing_results = preprocessor.run_full_preprocessing(data_path)

            self.pipeline_results['preprocessing'] = {
                'clean_data_path': str(self.config.CLEAN_DATA_PATH),
                'X_train_shape': preprocessing_results['X_train'].shape,
                'X_test_shape': preprocessing_results['X_test'].shape,
                'feature_columns': preprocessing_results['feature_columns'],
                'target_column': preprocessing_results['target_column']
            }

            logger.info(f"Preprocessing completed:")
            logger.info(f"  Training data: {preprocessing_results['X_train'].shape}")
            logger.info(f"  Test data: {preprocessing_results['X_test'].shape}")
            logger.info(f"  Features: {len(preprocessing_results['feature_columns'])}")

            return preprocessing_results

        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise

    def run_training(self, preprocessing_results: dict,
                     model_type: str = None) -> dict:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: MODEL TRAINING")
        logger.info("=" * 60)

        try:
            trainer = ModelTrainer(model_type=model_type or self.config.MODEL_TYPE)

            training_results = trainer.run_training_pipeline(preprocessing_results)

            self.pipeline_results['training'] = {
                'model_path': str(training_results['model_path']),
                'metrics': training_results['metrics'],
                'cv_results': training_results['cv_results'],
                'model_type': type(training_results['model']).__name__
            }

            logger.info(f"Training completed:")
            logger.info(f"  Model: {self.pipeline_results['training']['model_type']}")
            logger.info(f"  Test R²: {training_results['metrics']['test_r2']:.4f}")
            logger.info(f"  Test RMSE: {training_results['metrics']['test_rmse']:.4f}")

            return training_results

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def run_onnx_export(self, training_results: dict,
                        preprocessing_results: dict) -> dict:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: ONNX EXPORT")
        logger.info("=" * 60)

        try:
            exporter = ONNXExporter()
            export_results = exporter.export_training_results(
                training_results,
                preprocessing_results['X_test'][:10]
            )

            self.pipeline_results['onnx_export'] = export_results

            logger.info(f"ONNX export completed:")
            logger.info(f"  ONNX Path: {export_results['onnx_path']}")
            logger.info(f"  Prediction Match: {export_results['prediction_match']}")

            return export_results

        except Exception as e:
            logger.error(f"ONNX export failed: {str(e)}")
            raise


    def run_inference(self, preprocessing_results: dict,
                      use_onnx: bool = True) -> dict:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: MODEL INFERENCE")
        logger.info("=" * 60)

        try:
            inference = ModelInference(use_onnx=use_onnx)
            inference.load_model()

            X_test = preprocessing_results['X_test']
            y_test = preprocessing_results['y_test']
            scaler = preprocessing_results.get('scaler')

            predictions = inference.predict(X_test, scaler)

            self.pipeline_results['inference'] = {
                'predictions_path': str(self.config.PREDICTIONS_PATH),
                'samples_predicted': len(predictions),
                'model_info': inference.get_model_info()
            }

            logger.info(f"Inference completed:")
            logger.info(f"  Samples predicted: {len(predictions)}")
            logger.info(f"  Predictions saved to: {self.config.PREDICTIONS_PATH}")

            return {
                'predictions': predictions,
                'y_test': y_test,
                'model': inference.model
            }

        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise


    def run_evaluation(self, inference_results: dict,
                       training_results: dict = None) -> dict:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 6: EVALUATION & VISUALIZATION")
        logger.info("=" * 60)

        try:
            evaluator = ModelEvaluator()
            predictions = inference_results['predictions']
            y_test = inference_results['y_test']

            y_train_pred = None
            y_train_true = None
            model = inference_results.get('model')
            feature_names = None

            if training_results:
                model = training_results.get('model')
                feature_names = training_results.get('feature_columns', [])

            evaluation_summary = evaluator.run_complete_evaluation(
                y_test, predictions,
                model=model,
                feature_names=feature_names,
                y_train_true=y_train_true,
                y_train_pred=y_train_pred
            )

            self.pipeline_results['evaluation'] = evaluation_summary

            logger.info(f"Evaluation completed:")
            logger.info(f"  R² Score: {evaluation_summary['metrics']['r2']:.4f}")
            logger.info(f"  RMSE: {evaluation_summary['metrics']['rmse']:.4f}")
            logger.info(f"  Plots generated: {len(evaluation_summary['plots'])}")

            return evaluation_summary

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise



    def run_pipeline(self, args):
        logger.info("\n" + "=" * 60)
        logger.info("STARTING COMPLETE IOT DATA PIPELINE")
        logger.info("=" * 60)

        start_time = datetime.now()

        try:
            if not args.skip_data:
                data_results = self.run_data_acquisition(
                    generate_synthetic=not args.no_synthetic,
                    download_real=not args.no_real
                )

            preprocessing_results = self.run_preprocessing(
                use_synthetic=not args.use_real_data
            )

            training_results = self.run_training(
                preprocessing_results,
                model_type=args.model_type
            )

            if not args.skip_onnx:
                onnx_results = self.run_onnx_export(
                    training_results,
                    preprocessing_results
                )

            inference_results = self.run_inference(
                preprocessing_results,
                use_onnx=not args.skip_onnx
            )

            evaluation_results = self.run_evaluation(
                inference_results,
                training_results
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            self.save_pipeline_results(duration)

            self.print_summary(duration)

            logger.info("\n" + "=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"\nPipeline failed: {str(e)}")
            logger.error("Check the logs for details.")
            sys.exit(1)

    def save_pipeline_results(self, duration: float):
        try:
            self.pipeline_results['metadata'] = {
                'pipeline_version': '1.0',
                'run_timestamp': datetime.now().isoformat(),
                'duration_seconds': duration,
                'config': {
                    'model_type': self.config.MODEL_TYPE,
                    'test_size': self.config.TEST_SIZE,
                    'random_state': self.config.RANDOM_STATE
                }
            }

            results_path = self.config.OUTPUTS_DIR / "pipeline_results.json"
            with open(results_path, 'w') as f:
                json.dump(self.pipeline_results, f, indent=2, default=str)

            logger.info(f"Pipeline results saved to {results_path}")

        except Exception as e:
            logger.error(f"Error saving pipeline results: {str(e)}")


    def print_summary(self, duration: float):
        print("\n" + "=" * 60)
        print("IOT DATA PIPELINE - SUMMARY")
        print("=" * 60)

        if 'training' in self.pipeline_results:
            metrics = self.pipeline_results['training']['metrics']
            print(f"\nModel Performance:")
            print(f"  R² Score: {metrics['test_r2']:.4f}")
            print(f"  RMSE: {metrics['test_rmse']:.4f}")
            print(f"  MAE: {metrics['test_mae']:.4f}")

        if 'evaluation' in self.pipeline_results:
            plots = self.pipeline_results['evaluation']['plots']
            print(f"\nGenerated Plots:")
            for plot_name in plots:
                print(f"  • {plot_name}")

        print(f"\nOutput Files:")
        paths = self.config.get_all_paths()
        for name, path in paths.items():
            if path.exists():
                print(f"  • {name}: {path}")

        print(f"\nPipeline Duration: {duration:.2f} seconds")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='IoT Data Pipeline - Advanced Computer Programming Final Project'
    )

    parser.add_argument('--skip-data', action='store_true',
                        help='Skip data acquisition step')
    parser.add_argument('--no-synthetic', action='store_true',
                        help='Do not generate synthetic data')
    parser.add_argument('--no-real', action='store_true',
                        help='Do not download real data')
    parser.add_argument('--use-real-data', action='store_true',
                        help='Use real data instead of synthetic for training')

    parser.add_argument('--model-type', type=str, default=None,
                        choices=['linear', 'random_forest', 'gradient_boosting', 'svm', 'neural_network'],
                        help='Type of model to train')

    parser.add_argument('--skip-onnx', action='store_true',
                        help='Skip ONNX export step')

    parser.add_argument('--quick', action='store_true',
                        help='Run quick test with minimal data')

    args = parser.parse_args()

    if args.quick:
        logger.info("Running in QUICK mode")

    pipeline = IoTPipeline()
    pipeline.run_pipeline(args)


if __name__ == "__main__":
    main()