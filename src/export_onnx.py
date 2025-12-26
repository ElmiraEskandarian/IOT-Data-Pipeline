import numpy as np
from pathlib import Path
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from src.config import Config
from src.logger import logger


class ONNXExporter:

    def __init__(self, onnx_path: Path | None = None):
        self.onnx_path = onnx_path or Config.MODEL_ONNX_PATH
        logger.info("Initialized ONNXExporter on onnx path")

    def run_onnx_export(self, training_results: dict, preprocessing_results: dict) -> dict:

        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: ONNX EXPORT")
        logger.info("=" * 60)

        try:
            X_sample = preprocessing_results['X_train'].to_numpy().astype(np.float32)
            n_features = X_sample.shape[1]
            initial_type = [("float_input", FloatTensorType([None, n_features]))]

            logger.info("Exporting model to ONNX...")
            onnx_model = convert_sklearn(training_results['model'], initial_types=initial_type)

            with open(self.onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            logger.info(f"ONNX model saved to {self.onnx_path}")

            return {"onnx_path": str(self.onnx_path)}

        except Exception as e:
            logger.error(f"ONNX export failed: {str(e)}")
            raise