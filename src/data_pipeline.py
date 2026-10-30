"""
Data pipeline for TrafficFlow AI.
Handles dataset loading, validation, anomaly cleaning, feature engineering,
and sequence generation for LSTM neural networks.
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List
from sklearn.preprocessing import MinMaxScaler
import joblib

from src.config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, SCALER_PATH,
    METADATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN, SEQUENCE_LENGTH
)
from src.database import log_event

def load_raw_data() -> pd.DataFrame:
    """Loads raw CSV dataset."""
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    return df

def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates data integrity, schema consistency, null values,
    and checks for sensor anomalies.
    """
    required_cols = ['date_time', 'traffic_volume', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main']
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    if missing_cols:
        return {
            "is_valid": False,
            "error": f"Missing required columns: {missing_cols}",
            "missing_cols": missing_cols
        }

    total_rows = len(df)
    null_counts = df[required_cols].isnull().sum().to_dict()
    
    # Anomaly checks
    zero_kelvin_count = int((df['temp'] < 200).sum())
    negative_traffic_count = int((df['traffic_volume'] < 0).sum())
    extreme_rain_count = int((df['rain_1h'] > 100).sum())
    
    # Dates
    dates = pd.to_datetime(df['date_time'], errors='coerce')
    start_date = str(dates.min())
    end_date = str(dates.max())

    summary_stats = {
        "mean_volume": float(round(df['traffic_volume'].mean(), 1)),
        "median_volume": float(round(df['traffic_volume'].median(), 1)),
        "std_volume": float(round(df['traffic_volume'].std(), 1)),
        "min_volume": float(df['traffic_volume'].min()),
        "max_volume": float(df['traffic_volume'].max()),
        "total_observations": total_rows,
        "date_span": f"{start_date[:10]} to {end_date[:10]}"
    }

    validation_result = {
        "is_valid": True,
        "total_rows": total_rows,
        "start_date": start_date,
        "end_date": end_date,
        "required_columns_present": True,
        "null_counts": null_counts,
        "anomalies_detected": {
            "zero_kelvin_temp_sensors": zero_kelvin_count,
            "negative_traffic_values": negative_traffic_count,
            "extreme_precipitation_outliers": extreme_rain_count
        },
        "summary_stats": summary_stats
    }
    
    return validation_result

def clean_and_engineer_features(df: Optional[pd.DataFrame] = None, save_to_disk: bool = True) -> pd.DataFrame:
    """
    Cleans raw traffic dataset and engineers temporal & weather features.
    Saves clean features to disk.
    """
    if df is None:
        df = load_raw_data()
    
    df = df.copy()
    
    # 1. Parse Datetime and sort
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values('date_time').reset_index(drop=True)

    # 2. Fix Temperature Sensor Outliers (0 Kelvin anomalies)
    # Median Kelvin temp around 282K (~9 deg C)
    median_temp = df.loc[df['temp'] > 200, 'temp'].median()
    df.loc[df['temp'] < 200, 'temp'] = median_temp

    # Convert Kelvin to Celsius
    df['temp_c'] = np.round(df['temp'] - 273.15, 2)

    # 3. Clip extreme weather sensor noise
    df['rain_1h'] = df['rain_1h'].clip(0.0, 50.0)
    df['snow_1h'] = df['snow_1h'].clip(0.0, 20.0)
    df['clouds_all'] = df['clouds_all'].clip(0.0, 100.0)

    # 4. Temporal Features
    df['hour'] = df['date_time'].dt.hour
    df['dayofweek'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month

    # Cyclical encoding for Hour (24h period)
    df['hour_sin'] = np.round(np.sin(2 * np.pi * df['hour'] / 24.0), 4)
    df['hour_cos'] = np.round(np.cos(2 * np.pi * df['hour'] / 24.0), 4)

    # Cyclical encoding for Day of Week (7 day period)
    df['day_sin'] = np.round(np.sin(2 * np.pi * df['dayofweek'] / 7.0), 4)
    df['day_cos'] = np.round(np.cos(2 * np.pi * df['dayofweek'] / 7.0), 4)

    # Weekend and Holiday flags
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # Holiday: if not null and not 'None'
    df['is_holiday'] = ((df['holiday'].notnull()) & (df['holiday'].str.lower() != 'none')).astype(int)

    # Rush hour flag (Monday to Friday, 07:00-09:00 and 16:00-18:00)
    df['is_rush_hour'] = (
        (df['is_weekend'] == 0) & 
        (((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 16) & (df['hour'] <= 18)))
    ).astype(int)

    # 5. Handle duplicate hourly timestamps by aggregating (mean for continuous)
    agg_dict = {
        'traffic_volume': 'mean',
        'temp_c': 'mean',
        'rain_1h': 'mean',
        'snow_1h': 'mean',
        'clouds_all': 'mean',
        'hour_sin': 'first',
        'hour_cos': 'first',
        'day_sin': 'first',
        'day_cos': 'first',
        'is_weekend': 'first',
        'is_holiday': 'max',
        'is_rush_hour': 'first',
        'weather_main': 'first',
        'weather_description': 'first',
    }
    
    clean_df = df.groupby('date_time').agg(agg_dict).reset_index()
    clean_df = clean_df.sort_values('date_time').reset_index(drop=True)

    # Round continuous features for neatness
    clean_df['traffic_volume'] = clean_df['traffic_volume'].round(0).astype(int)
    clean_df['temp_c'] = clean_df['temp_c'].round(2)
    clean_df['rain_1h'] = clean_df['rain_1h'].round(2)
    clean_df['snow_1h'] = clean_df['snow_1h'].round(2)
    clean_df['clouds_all'] = clean_df['clouds_all'].round(1)

    if save_to_disk:
        clean_df.to_csv(PROCESSED_DATA_PATH, index=False)
        log_event("DATA", f"Cleaned dataset saved: {len(clean_df)} hourly records to {PROCESSED_DATA_PATH}")

    return clean_df

def load_processed_data() -> pd.DataFrame:
    """Loads cleaned and feature-engineered dataset."""
    if not os.path.exists(PROCESSED_DATA_PATH):
        clean_and_engineer_features(save_to_disk=True)
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df['date_time'] = pd.to_datetime(df['date_time'])
    return df

def create_lstm_sequences(
    data: np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts 2D scaled tabular array into 3D LSTM sliding window sequences:
    X shape: (num_samples, sequence_length, num_features)
    y shape: (num_samples,) predicting traffic_volume (index 0) at t+sequence_length.
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i : i + sequence_length])
        y.append(data[i + sequence_length, 0])  # traffic_volume is column 0
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def prepare_data_pipeline(
    df: Optional[pd.DataFrame] = None,
    sequence_length: int = SEQUENCE_LENGTH,
    feature_cols: Optional[List[str]] = None,
    val_split: float = 0.15,
    test_split: float = 0.15,
    save_artifacts: bool = True
) -> Dict[str, Any]:
    """
    End-to-end pipeline to produce Train, Validation, and Test 3D tensors for LSTM.
    Maintains strict chronological order for realistic time-series forecasting.
    """
    if df is None:
        df = load_processed_data()

    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    # Ensure target column is index 0
    if TARGET_COLUMN in feature_cols:
        feature_cols = [TARGET_COLUMN] + [c for c in feature_cols if c != TARGET_COLUMN]
    
    matrix = df[feature_cols].values.astype(np.float32)

    # Chronological Splits
    n = len(matrix)
    test_idx = int(n * (1.0 - test_split))
    val_idx = int(test_idx * (1.0 - val_split))

    train_raw = matrix[:val_idx]
    val_raw = matrix[val_idx:test_idx]
    test_raw = matrix[test_idx:]

    # Fit MinMaxScaler ONLY on training data to avoid data leakage
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    train_scaled = scaler.fit_transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)

    # Generate 3D sequences
    X_train, y_train = create_lstm_sequences(train_scaled, sequence_length)
    X_val, y_val = create_lstm_sequences(val_scaled, sequence_length)
    X_test, y_test = create_lstm_sequences(test_scaled, sequence_length)

    # Scaler min/max for the target column to allow easy inverse transformation
    target_min = float(scaler.data_min_[0])
    target_max = float(scaler.data_max_[0])

    if save_artifacts:
        joblib.dump(scaler, SCALER_PATH)
        metadata = {
            "feature_columns": feature_cols,
            "target_column": TARGET_COLUMN,
            "sequence_length": sequence_length,
            "target_min": target_min,
            "target_max": target_max,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
        }
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=4)
        log_event("PIPELINE", f"Saved scaler to {SCALER_PATH} and metadata to {METADATA_PATH}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "target_min": target_min,
        "target_max": target_max,
        "test_dates": df['date_time'].iloc[test_idx + sequence_length:].reset_index(drop=True)
    }
