import pandas as pd
from pathlib import Path
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

from src.config import Config

warnings.filterwarnings('ignore')

from src.logger import logger


class DataPreprocessor:

    def __init__(self):
        self.config = Config
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = None
        logger.info("Initialized DataPreprocessor module")

    def load_data(self, data_path: Path) -> pd.DataFrame:
        try:
            logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path, parse_dates=['Time'])
            logger.info(f"Data loaded with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def handle_missing_values(self, df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        try:
            logger.info("Handling missing values...")
            missing_before = df.isnull().sum().sum()

            if method == 'interpolate':
                df = df.interpolate(method='linear', limit_direction='both')

                df = df.ffill().bfill()

            elif method == 'mean':
                df = df.fillna(df.mean())

            elif method == 'drop':
                df = df.dropna()

            missing_after = df.isnull().sum().sum()
            logger.info(f"Missing values: {missing_before} -> {missing_after}")

            return df

        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise


    def remove_outliers(self, df: pd.DataFrame, columns: List[str],
                        method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        try:
            logger.info("Removing outliers...")
            outliers_count = 0

            df_clean = df.copy()

            for col in columns:
                if col not in df.columns or col == 'Time':
                    continue

                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR

                    df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
                    df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])

                    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    outliers_count += outliers

                elif method == 'zscore':
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    outliers = (z_scores > threshold).sum()
                    outliers_count += outliers

                    median_val = df[col].median()
                    df_clean[col] = np.where(z_scores > threshold, median_val, df_clean[col])

            logger.info(f"Outliers handled: {outliers_count} total")
            return df_clean

        except Exception as e:
            logger.error(f"Error removing outliers: {str(e)}")
            raise


    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Creating features...")

            df_features = df.copy()

            if 'Time' in df.columns:
                df_features['hour'] = df_features['Time'].dt.hour
                df_features['day_of_week'] = df_features['Time'].dt.dayofweek
                df_features['month'] = df_features['Time'].dt.month
                df_features['day_of_year'] = df_features['Time'].dt.dayofyear

                df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
                df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
                df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
                df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)

            numeric_cols = df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                if col != 'Time':
                    for lag in [1, 2, 3, 6, 12]:
                        df_features[f'{col}_lag_{lag}'] = df_features[col].shift(lag)

                    df_features[f'{col}_rolling_mean_3'] = df_features[col].rolling(window=3).mean()
                    df_features[f'{col}_rolling_std_3'] = df_features[col].rolling(window=3).std()

                    df_features[f'{col}_diff_1'] = df_features[col].diff()

            df_features = df_features.dropna()

            logger.info(f"Features created. New shape: {df_features.shape}")
            return df_features

        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            raise


    def scale_features(self, df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, Any]:
        try:
            logger.info("Scaling features...")

            existing_cols = [col for col in feature_cols if col in df.columns]
            scaled_features = self.scaler.fit_transform(df[existing_cols])

            df_scaled = df.copy()
            df_scaled[existing_cols] = scaled_features

            logger.info(f"Features scaled: {len(existing_cols)} columns")
            return df_scaled, self.scaler

        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            raise


    def prepare_data_for_training(self, df: pd.DataFrame, target_col: str = 'Temperature_Sensor_1') -> Tuple:
        try:
            logger.info(f"Preparing data for training with target: {target_col}")

            df = self.handle_missing_values(df)

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Time' in numeric_cols:
                numeric_cols.remove('Time')

            df = self.remove_outliers(df, numeric_cols)

            df = self.create_features(df)

            feature_cols = [col for col in df.columns
                            if col not in ['Time', target_col]
                            and not col.startswith('Unnamed')]

            if target_col not in df.columns:
                logger.warning(f"Target column {target_col} not found. Using first temperature sensor.")
                temp_cols = [col for col in df.columns if 'Temperature' in col]
                if temp_cols:
                    target_col = temp_cols[0]
                else:
                    target_col = df.columns[1]

            x = df[feature_cols]
            y = df[target_col]


            x_train, x_test, y_train, y_test = train_test_split(
                x, y,
                test_size=self.config.TEST_SIZE,
                random_state=self.config.RANDOM_STATE,
                shuffle=False
            )

            x_scaled, scaler = self.scale_features(x, feature_cols)

            self.feature_columns = feature_cols
            self.target_column = target_col

            logger.info(f"Data prepared. Shapes - X_train: {x_train.shape}, X_test: {x_test.shape}")

            return x_train, x_test, y_train, y_test, scaler

        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise


    def run_full_preprocessing(self, data_path: Path) -> Dict[str, Any]:
        try:
            logger.info("Starting full preprocessing pipeline...")

            df = self.load_data(data_path)

            X_train, X_test, y_train, y_test, scaler = self.prepare_data_for_training(df)

            clean_df = pd.concat([
                pd.DataFrame(X_train, columns=self.feature_columns).assign(Target=y_train.values),
                pd.DataFrame(X_test, columns=self.feature_columns).assign(Target=y_test.values)
            ])

            clean_df.to_csv(self.config.CLEAN_DATA_PATH, index=False)
            logger.info(f"Clean data saved to {self.config.CLEAN_DATA_PATH}")

            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'scaler': scaler,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column
            }

        except Exception as e:
            logger.error(f"Error in full preprocessing: {str(e)}")
            raise
