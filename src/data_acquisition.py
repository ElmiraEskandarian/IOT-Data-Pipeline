import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.utils.kaggle_utils import download_kaggle_dataset
from src.config import Config
from src.logger import logger


class DataAcquisition:

    def __init__(self):
        self.config = Config
        logger.info("Initialized DataAcquisition module")

    def generate_synthetic_data(self) -> pd.DataFrame:
        try:
            logger.info("Generating simple synthetic weather data...")

            # Create time range
            n_rows = 10000  # Similar to your real data size
            start_date = datetime(2025, 1, 1)
            times = [start_date + timedelta(hours=i) for i in range(n_rows)]

            # Generate simple patterns for each column
            hours = np.arange(n_rows)

            # Temperature (C) - simple daily pattern
            temperature = 10 + 10 * np.sin(2 * np.pi * (hours % 24) / 24) + np.random.normal(0, 2, n_rows)

            # Apparent Temperature (C) - similar to temperature with slight variation
            apparent_temp = temperature + np.random.normal(0, 1, n_rows)

            # Humidity (0-1 scale)
            humidity = 0.5 + 0.3 * np.sin(2 * np.pi * (hours % 24) / 24 - np.pi / 4) + np.random.normal(0, 0.1, n_rows)
            humidity = np.clip(humidity, 0, 1)

            # Wind Speed (km/h)
            wind_speed = 10 + 5 * np.random.random(n_rows)

            # Wind Bearing (degrees)
            wind_bearing = np.random.uniform(0, 360, n_rows)

            # Visibility (km)
            visibility = 10 + 5 * np.random.random(n_rows)

            # Pressure (millibars)
            pressure = 1013 + 10 * np.random.random(n_rows)

            # Simple categorical columns
            summaries = np.random.choice(
                ['Partly Cloudy', 'Clear', 'Overcast', 'Rain', 'Snow'],
                size=n_rows,
                p=[0.3, 0.3, 0.2, 0.15, 0.05]
            )

            # Precip Type based on summary
            precip_type = []
            for summary in summaries:
                if summary == 'Rain':
                    precip_type.append('rain')
                elif summary == 'Snow':
                    precip_type.append('snow')
                else:
                    precip_type.append(None)

            # Daily Summary - simple pattern
            daily_summaries = []
            for i in range(n_rows):
                hour = hours[i] % 24
                if hour < 6 or hour > 20:
                    daily_summaries.append('Clear throughout the night.')
                else:
                    daily_summaries.append('Partly cloudy throughout the day.')

            # Create DataFrame with EXACT same columns as real data
            df = pd.DataFrame({
                'Formatted Date': times,
                'Summary': summaries,
                'Precip Type': precip_type,
                'Temperature (C)': temperature,
                'Apparent Temperature (C)': apparent_temp,
                'Humidity': humidity,
                'Wind Speed (km/h)': wind_speed,
                'Wind Bearing (degrees)': wind_bearing,
                'Visibility (km)': visibility,
                'Loud Cover': np.zeros(n_rows),  # Always 0
                'Pressure (millibars)': pressure,
                'Daily Summary': daily_summaries
            })

            # Add a tiny bit of missing values (like real data)
            mask = np.random.random(df.shape) < 0.01
            df = df.mask(mask)

            logger.info(f"Generated synthetic data with shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")

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
