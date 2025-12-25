import numpy as np
from pathlib import Path
from typing import List, Tuple, Any

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnx
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError as e:
    ONNX_AVAILABLE = False
    logger.warning(f"ONNX dependencies not available: {str(e)}")

from src.config import Config
from src.logger import logger


class ONNXExporter:

    def __init__(self):
        self.config = Config

        if not ONNX_AVAILABLE:
            logger.warning("ONNX libraries not installed. Install with: pip install skl2onnx onnx onnxruntime")

        logger.info("Initialized ONNXExporter module")

    def get_model_input_type(self, model, X_sample: np.ndarray) -> List[Tuple]:
        try:
            n_features = X_sample.shape[1]

            initial_type = [('float_input', FloatTensorType([None, n_features]))]

            return initial_type

        except Exception as e:
            logger.error(f"Error getting model input type: {str(e)}")
            raise

    def convert_to_onnx(self, model: Any, X_sample: np.ndarray,
                        onnx_path: Path = None) -> Path:
        try:
            if not ONNX_AVAILABLE:
                raise ImportError("ONNX libraries not installed")

            logger.info("Converting model to ONNX format...")

            initial_type = self.get_model_input_type(model, X_sample)

            onnx_model = convert_sklearn(
                model,
                initial_types=initial_type,
                target_opset=12
            )

            onnx_path = onnx_path or self.config.MODEL_ONNX_PATH
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            self.verify_onnx_model(onnx_path, X_sample)

            logger.info(f"ONNX model saved to {onnx_path}")

            return onnx_path

        except Exception as e:
            logger.error(f"Error converting to ONNX: {str(e)}")
            raise

    def verify_onnx_model(self, onnx_path: Path, X_sample: np.ndarray) -> bool:
        try:
            if not ONNX_AVAILABLE:
                return False

            logger.info("Verifying ONNX model...")

            onnx_model = onnx.load(onnx_path)

            onnx.checker.check_model(onnx_model)

            ort_session = ort.InferenceSession(onnx_path)

            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name

            onnx_pred = ort_session.run(
                [output_name],
                {input_name: X_sample.astype(np.float32)}
            )

            logger.info(f"ONNX model verification successful")
            logger.info(f"Input name: {input_name}, Output name: {output_name}")
            logger.info(f"Sample prediction shape: {onnx_pred[0].shape}")

            return True

        except Exception as e:
            logger.error(f"Error verifying ONNX model: {str(e)}")
            raise

    def compare_predictions(self, sklearn_model: Any, onnx_path: Path,
                            X_test: np.ndarray, tolerance: float = 1e-5) -> bool:
        try:
            if not ONNX_AVAILABLE:
                return False

            logger.info("Comparing sklearn and ONNX predictions...")

            sklearn_pred = sklearn_model.predict(X_test)

            ort_session = ort.InferenceSession(onnx_path)
            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name

            onnx_pred = ort_session.run(
                [output_name],
                {input_name: X_test.astype(np.float32)}
            )[0]

            diff = np.abs(sklearn_pred - onnx_pred.flatten())
            max_diff = diff.max()
            mean_diff = diff.mean()

            logger.info(f"Prediction comparison:")
            logger.info(f"  Max difference: {max_diff:.6f}")
            logger.info(f"  Mean difference: {mean_diff:.6f}")
            logger.info(f"  Tolerance: {tolerance}")

            if max_diff > tolerance:
                logger.warning(f"Significant difference between sklearn and ONNX predictions")
                return False

            logger.info("Predictions match within tolerance")
            return True

        except Exception as e:
            logger.error(f"Error comparing predictions: {str(e)}")
            raise

    def export_training_results(self, training_results: dict[str, Any],
                                X_sample: np.ndarray) -> dict[str, Any]:
        try:
            logger.info("Exporting training results to ONNX...")

            model = training_results['model']

            onnx_path = self.convert_to_onnx(model, X_sample)

            comparison_result = self.compare_predictions(
                model,
                onnx_path,
                X_sample[:10]
            )


            export_results = {
                'onnx_path': str(onnx_path),              'onnx_available': ONNX_AVAILABLE,
                'prediction_match': comparison_result,
                'model_type': type(model).__name__,
                'input_shape': X_sample.shape,
                'output_path': str(onnx_path)
            }


            metadata = {
                'export_date': np.datetime64('now').astype(str),
                'model_info': export_results,
                'training_metrics': training_results.get('metrics', {})
            }

            metadata_path = onnx_path.with_suffix('.metadata.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Export metadata saved to {metadata_path}")

            return export_results

        except Exception as e:
            logger.error(f"Error exporting training results: {str(e)}")
            raise