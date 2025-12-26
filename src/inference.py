import numpy as np
import onnxruntime as ort
from pathlib import Path
from src.config import Config
from src.logger import logger


class ModelInference:

    def __init__(self, model_path: Path = None):
        self.model_path = model_path or Config.MODEL_ONNX_PATH
        self.model = None
        self.ort_session = None
        self.input_name = None
        self.output_name = None

        logger.info(f"Initialized ModelInference with ONNX")


    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self.ort_session = ort.InferenceSession(str(self.model_path))
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        logger.info(f"ONNX model loaded. Input: {self.input_name}, Output: {self.output_name}")


    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.ort_session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if x.ndim == 1:
            x = x.reshape(1, -1)

        predictions = self.ort_session.run([self.output_name], {self.input_name: x})[0]
        return predictions
