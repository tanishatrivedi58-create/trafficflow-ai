"""
Deep Learning LSTM Architecture for TrafficFlow AI.
Builds, compiles, saves, and loads the multi-layer LSTM neural network.
"""
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.config import MODEL_PATH, SEQUENCE_LENGTH, FEATURE_COLUMNS, LEARNING_RATE
from src.database import log_event

def build_traffic_lstm(
    sequence_length: int = SEQUENCE_LENGTH,
    num_features: int = len(FEATURE_COLUMNS),
    learning_rate: float = LEARNING_RATE,
    lstm_units_1: int = 64,
    lstm_units_2: int = 32,
    dense_units: int = 16,
    dropout_rate: float = 0.2
) -> tf.keras.Model:
    """
    Constructs a stacked LSTM neural network for temporal traffic volume regression.
    """
    model = Sequential([
        Input(shape=(sequence_length, num_features)),
        LSTM(lstm_units_1, return_sequences=True, activation="tanh", recurrent_activation="sigmoid"),
        Dropout(dropout_rate),
        LSTM(lstm_units_2, return_sequences=False, activation="tanh", recurrent_activation="sigmoid"),
        Dropout(dropout_rate),
        Dense(dense_units, activation="relu"),
        Dense(1, activation="linear")
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="huber", metrics=["mae", "mse"])
    return model

def is_model_available() -> bool:
    """Checks whether the trained LSTM model file exists on disk."""
    return os.path.exists(MODEL_PATH)

def load_traffic_model(model_path: str = MODEL_PATH) -> tf.keras.Model:
    """Loads a pre-trained Keras model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    model = load_model(model_path)
    return model

def save_traffic_model(model: tf.keras.Model, model_path: str = MODEL_PATH):
    """Saves the Keras model to disk."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    log_event("MODEL", f"Model successfully saved to {model_path}")
