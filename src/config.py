from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = BASE_DIR / "models"
    OUTPUTS_DIR = BASE_DIR / "outputs"
    PLOTS_DIR = OUTPUTS_DIR / "plots"

    for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    KAGGLE_WEATHER_DATASET_URL = "https://www.kaggle.com/datasets/muthuj7/weather-dataset"

    SYNTHETIC_DATA_PATH = RAW_DATA_DIR / "synthetic_data.csv"
    REAL_DATA_PATH = RAW_DATA_DIR / "weatherHistory.csv"
    CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "clean_data.csv"
    MODEL_PKL_PATH = MODELS_DIR / "temperature_model.pkl"
    MODEL_ONNX_PATH = MODELS_DIR / "temperature_model.onnx"
    PREDICTIONS_PATH = OUTPUTS_DIR / "predictions.csv"

    SENSOR_COUNT = 5
    TIME_PERIODS = 1000
    FREQUENCY = "H"
    START_DATE = "2025-01-01"

    TEMPERATURE_PARAMS = {
        'base': 20,
        'amplitude': 5,
        'noise_std': 0.5
    }

    HUMIDITY_PARAMS = {
        'base': 50,
        'amplitude': 10,
        'noise_std': 1
    }

    PRESSURE_PARAMS = {
        'base': 1013,
        'amplitude': 20,
        'noise_std': 2
    }

    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    MODEL_TYPE = "random_forest"

    @classmethod
    def get_all_paths(cls):
        return {
            'synthetic_data': cls.SYNTHETIC_DATA_PATH,
            'real_data': cls.REAL_DATA_PATH,
            'clean_data': cls.CLEAN_DATA_PATH,
            'model_pkl': cls.MODEL_PKL_PATH,
            'model_onnx': cls.MODEL_ONNX_PATH,
            'predictions': cls.PREDICTIONS_PATH,
            'plots_dir': cls.PLOTS_DIR
        }