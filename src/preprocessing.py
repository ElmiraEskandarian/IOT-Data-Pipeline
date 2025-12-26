import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.config import Config
from src.logger import logger


class DataPreprocessor:

    def __init__(self):
        self.config = Config
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'Temperature (C)'
        logger.info("Initialized DataPreprocessor")

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting preprocessing")

        df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
        df['hour'] = df['Formatted Date'].dt.hour
        df['dayofweek'] = df['Formatted Date'].dt.dayofweek
        df['month'] = df['Formatted Date'].dt.month

        df = df.drop(columns=['Daily Summary', 'Formatted Date'])
        df = df.dropna()
        df = pd.get_dummies(df, columns=['Summary', 'Precip Type'], drop_first=True)

        logger.info(f"After preprocessing shape: {df.shape}")
        return df


    def prepare_data_for_training(self, df: pd.DataFrame) -> Tuple:
        logger.info("Preparing data for training")

        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")

        data = df.drop(columns=[self.target_column])
        labels = df[self.target_column]

        self.feature_columns = data.columns.tolist()

        data_train, data_test, label_train, label_test = train_test_split(
            data,
            labels,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE
        )

        data_train[self.feature_columns] = self.scaler.fit_transform(data_train)
        data_test[self.feature_columns] = self.scaler.transform(data_test)

        logger.info(f"Train shape: {data_train.shape}, Test shape: {data_test.shape}")

        return data_train, data_test, label_train, label_test, self.scaler


    def run_full_preprocessing(self, data_path: Path) -> Dict[str, Any]:
        logger.info("Running full preprocessing pipeline")

        df = pd.read_csv(data_path)
        df = self.preprocess(df)

        data_train, data_test, label_train, label_test, scaler = self.prepare_data_for_training(df)

        clean_df = pd.concat([
            data_train.assign(Target=label_train.values),
            data_test.assign(Target=label_test.values)
        ])

        clean_df.to_csv(self.config.CLEAN_DATA_PATH, index=False)
        logger.info(f"Clean data saved to {self.config.CLEAN_DATA_PATH}")

        return {
            'X_train': data_train,
            'X_test': data_test,
            'y_train': label_train,
            'y_test': label_test,
            'scaler': scaler,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
