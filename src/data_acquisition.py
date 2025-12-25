import pandas as pd
import numpy as np
from typing import Dict

from src.utils.kaggle_utils import download_kaggle_dataset
from src.config import Config
from src.logger import logger


class DataAcquisition:

    def __init__(self):
        self.config = Config
        logger.info("Initialized DataAcquisition module")

    def generate_synthetic_data(self) -> pd.DataFrame:
        try:
            logger.info("Generating synthetic sensor data...")

            time_index = pd.date_range(
                start=self.config.START_DATE,
                periods=self.config.TIME_PERIODS,
                freq=self.config.FREQUENCY
            )

            data = {"Time": time_index}

            for i in range(self.config.SENSOR_COUNT):

                temp_phase = np.random.uniform(0, 2 * np.pi)
                temperature = (
                        self.config.TEMPERATURE_PARAMS['base'] +
                        self.config.TEMPERATURE_PARAMS['amplitude'] * np.sin(
                    np.linspace(0 + temp_phase, 10 + temp_phase, self.config.TIME_PERIODS)
                ) +
                        np.random.normal(0, self.config.TEMPERATURE_PARAMS['noise_std'], self.config.TIME_PERIODS)
                )

                humidity_phase = np.random.uniform(0, 2 * np.pi)
                humidity = (
                        self.config.HUMIDITY_PARAMS['base'] +
                        self.config.HUMIDITY_PARAMS['amplitude'] * np.cos(
                    np.linspace(0 + humidity_phase, 10 + humidity_phase, self.config.TIME_PERIODS)
                ) +
                        np.random.normal(0, self.config.HUMIDITY_PARAMS['noise_std'], self.config.TIME_PERIODS)
                )

                pressure_phase = np.random.uniform(0, 2 * np.pi)
                pressure = (
                        self.config.PRESSURE_PARAMS['base'] +
                        self.config.PRESSURE_PARAMS['amplitude'] * np.sin(
                    np.linspace(0 + pressure_phase, 5 + pressure_phase, self.config.TIME_PERIODS)
                ) +
                        np.random.normal(0, self.config.PRESSURE_PARAMS['noise_std'], self.config.TIME_PERIODS)
                )

                data[f"Temperature_Sensor_{i + 1}"] = temperature
                data[f"Humidity_Sensor_{i + 1}"] = humidity
                data[f"Pressure_Sensor_{i + 1}"] = pressure

            df = pd.DataFrame(data)

            mask = np.random.random(df.shape) < 0.05
            df = df.mask(mask)

            numeric_cols = df.select_dtypes(include=np.number).columns
            outlier_mask = np.random.random(df[numeric_cols].shape) < 0.01
            df[numeric_cols] = df[numeric_cols].mask(
                outlier_mask, df[numeric_cols] * np.random.uniform(1.5, 3)
            )

            logger.info(f"Generated synthetic data with shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error generating synthetic data: {str(e)}")
            raise

    def get_real_data(self, force_download: bool = False) -> pd.DataFrame:
        try:
            if not force_download and self.config.REAL_DATA_PATH.exists():
                logger.info("Loading existing real weather data...")
                return pd.read_csv(self.config.REAL_DATA_PATH)

            download_kaggle_dataset(self.config.KAGGLE_WEATHER_DATASET_URL, self.config.REAL_DATA_PATH)

            df = pd.read_csv(self.config.REAL_DATA_PATH)
            logger.info(f"Real data saved to {self.config.REAL_DATA_PATH}")

            return df

        except Exception as e:
            logger.error(f"Error getting real data: {str(e)}")
            raise

    def get_all_data(self) -> Dict[str, pd.DataFrame]:

        try:
            synthetic_data = self.generate_synthetic_data()
            synthetic_data.to_csv(self.config.SYNTHETIC_DATA_PATH, index=False)
            logger.info(f"Synthetic data saved to {self.config.SYNTHETIC_DATA_PATH}")

            real_data = self.get_real_data()

            return {
                'synthetic': synthetic_data,
                'real': real_data
            }

        except Exception as e:
            logger.error(f"Error in get_all_data: {str(e)}")
            raise
