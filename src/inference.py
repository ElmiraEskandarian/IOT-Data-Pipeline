import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Union
import pickle
import json


try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available")

from src.config import Config
from src.logger import logger


class ModelInference:

    def __init__(self, model_path: Path = None, use_onnx: bool = True):
        self.config = Config
        self.model_path = model_path or (
            self.config.MODEL_ONNX_PATH if use_onnx else self.config.MODEL_PKL_PATH
        )
        self.use_onnx = use_onnx and ONNX_AVAILABLE
        self.model = None
        self.ort_session = None
        self.input_name = None
        self.output_name = None
        self.feature_columns = []

        logger.info(f"Initialized ModelInference with ONNX: {self.use_onnx}")

    def load_model(self) -> Any:
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            if self.use_onnx and ONNX_AVAILABLE:
                logger.info(f"Loading ONNX model from {self.model_path}")


                self.ort_session = ort.InferenceSession(str(self.model_path))


                self.input_name = self.ort_session.get_inputs()[0].name
                self.output_name = self.ort_session.get_outputs()[0].name

                logger.info(f"ONNX model loaded. Input: {self.input_name}, Output: {self.output_name}")
                return self.ort_session

            else:
                logger.info(f"Loading sklearn model from {self.model_path}")


                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)


                metadata_path = self.model_path.with_suffix('.json')
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    self.feature_columns = metadata.get('feature_columns', [])
                    logger.info(f"Loaded model metadata")

                logger.info(f"Sklearn model loaded: {type(self.model).__name__}")
                return self.model

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def prepare_input_data(self, data: Union[pd.DataFrame, np.ndarray, List],
                           scaler: Any = None) -> np.ndarray:

        try:

            if isinstance(data, pd.DataFrame):

                if self.feature_columns:

                    missing_cols = set(self.feature_columns) - set(data.columns)
                    if missing_cols:
                        logger.warning(f"Missing columns in input data: {missing_cols}")

                        for col in missing_cols:
                            data[col] = 0

                    input_data = data[self.feature_columns].values
                else:

                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    input_data = data[numeric_cols].values

            elif isinstance(data, list):
                input_data = np.array(data)
            else:
                input_data = data


            if input_data.ndim == 1:
                input_data = input_data.reshape(1, -1)


            if scaler is not None:
                input_data = scaler.transform(input_data)

            logger.info(f"Input data prepared. Shape: {input_data.shape}")
            return input_data.astype(np.float32)

        except Exception as e:
            logger.error(f"Error preparing input data: {str(e)}")
            raise

    def predict(self, input_data: Union[pd.DataFrame, np.ndarray, List],
                scaler: Any = None) -> np.ndarray:

        try:
            if self.model is None and self.ort_session is None:
                self.load_model()

            processed_data = self.prepare_input_data(input_data, scaler)

            if self.use_onnx and ONNX_AVAILABLE and self.ort_session is not None:

                predictions = self.ort_session.run(
                    [self.output_name],
                    {self.input_name: processed_data}
                )[0]

            else:

                if self.model is None:
                    raise ValueError("Model not loaded")

                predictions = self.model.predict(processed_data)

            logger.info(f"Made predictions for {len(predictions)} samples")
            return predictions.flatten()

        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

    def predict_batch(self, data_path: Path, scaler: Any = None) -> pd.DataFrame:
        try:
            logger.info(f"Making batch predictions from {data_path}")

            if data_path.suffix == '.csv':
                data = pd.read_csv(data_path)
            elif data_path.suffix == '.parquet':
                data = pd.read_parquet(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")

            predictions = self.predict(data, scaler)

            results = pd.DataFrame({
                'timestamp': pd.Timestamp.now(),
                'predictions': predictions,
                'input_file': str(data_path)
            })

            results.to_csv(self.config.PREDICTIONS_PATH, index=False)
            logger.info(f"Predictions saved to {self.config.PREDICTIONS_PATH}")

            return results

        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            raise

    def predict_real_time(self, sensor_values: Dict[str, float],
                          scaler: Any = None) -> Dict[str, Any]:

        try:
            logger.info("Making real-time prediction...")

            df = pd.DataFrame([sensor_values])
            prediction = self.predict(df, scaler)

            result = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'prediction': float(prediction[0]),
                'sensor_values': sensor_values,
                'confidence': 0.95
            }

            logger.info(f"Real-time prediction: {result['prediction']:.2f}")
            return result

        except Exception as e:
            logger.error(f"Error in real-time prediction: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        try:
            info = {
                'model_path': str(self.model_path),
                'use_onnx': self.use_onnx,
                'onnx_available': ONNX_AVAILABLE,
                'model_loaded': self.model is not None or self.ort_session is not None
            }

            if self.use_onnx and self.ort_session is not None:
                info.update({
                    'input_name': self.input_name,
                    'output_name': self.output_name,
                    'input_shape': self.ort_session.get_inputs()[0].shape,
                    'output_shape': self.ort_session.get_outputs()[0].shape
                })

            elif self.model is not None:
                info.update({
                    'model_type': type(self.model).__name__,
                    'feature_columns': self.feature_columns
                })

            return info

        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            raise
