"""
Model Training and Fine-Tuning Pipeline for TrafficFlow AI.
Trains stacked LSTM neural networks, captures live training callbacks,
and records historical training runs in SQLite.
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from typing import Dict, Any, Optional, Callable

from src.config import (
    MODEL_PATH, HISTORY_PATH, SEQUENCE_LENGTH,
    BATCH_SIZE, DEFAULT_EPOCHS, LEARNING_RATE
)
from src.data_pipeline import load_processed_data, prepare_data_pipeline
from src.lstm_model import build_traffic_lstm, save_traffic_model
from src.evaluation import calculate_metrics
from src.database import insert_training_run, log_event

class StreamlitProgressCallback(Callback):
    """Custom Keras callback to send live epoch progress to Streamlit UI."""
    def __init__(self, total_epochs: int, update_fn: Optional[Callable[[int, int, float, float, str], None]] = None):
        super().__init__()
        self.total_epochs = total_epochs
        self.update_fn = update_fn

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = float(logs.get("loss", 0.0))
        val_loss = float(logs.get("val_loss", 0.0))
        epoch_num = epoch + 1
        msg = f"Epoch {epoch_num}/{self.total_epochs} completed - Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}"
        if self.update_fn:
            self.update_fn(epoch_num, self.total_epochs, train_loss, val_loss, msg)

def train_traffic_model(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = BATCH_SIZE,
    sequence_length: int = SEQUENCE_LENGTH,
    learning_rate: float = LEARNING_RATE,
    user_id: Optional[int] = None,
    username: str = "system",
    progress_callback: Optional[Callable[[int, int, float, float, str], None]] = None
) -> Dict[str, Any]:
    """
    Executes complete end-to-end training of the stacked LSTM model.
    Evaluates on holdout test set and saves all production artifacts.
    """
    log_event("TRAIN", f"Starting training: {epochs} epochs, batch_size={batch_size}, seq_len={sequence_length}", username=username)

    # 1. Prepare data splits
    pipeline_data = prepare_data_pipeline(
        sequence_length=sequence_length,
        save_artifacts=True
    )

    X_train = pipeline_data["X_train"]
    y_train = pipeline_data["y_train"]
    X_val = pipeline_data["X_val"]
    y_val = pipeline_data["y_val"]
    X_test = pipeline_data["X_test"]
    y_test = pipeline_data["y_test"]
    scaler = pipeline_data["scaler"]
    target_min = pipeline_data["target_min"]
    target_max = pipeline_data["target_max"]
    test_dates = pipeline_data["test_dates"]

    num_features = X_train.shape[2]

    # 2. Build LSTM Model
    model = build_traffic_lstm(
        sequence_length=sequence_length,
        num_features=num_features,
        learning_rate=learning_rate
    )

    # 3. Setup Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=0)
    ]

    if progress_callback is not None:
        callbacks.append(StreamlitProgressCallback(total_epochs=epochs, update_fn=progress_callback))

    # 4. Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # 5. Evaluate on Holdout Test Set
    test_pred_scaled = model.predict(X_test, verbose=0).flatten()
    
    # Invert target scaling: volume = scaled * (max - min) + min
    y_test_actual = y_test * (target_max - target_min) + target_min
    y_test_pred = test_pred_scaled * (target_max - target_min) + target_min
    y_test_pred = np.clip(y_test_pred, 0.0, None)

    # Compute comprehensive metrics
    test_metrics = calculate_metrics(y_test_actual, y_test_pred)

    # Final epoch losses
    final_train_loss = float(history.history["loss"][-1])
    final_val_loss = float(history.history["val_loss"][-1]) if "val_loss" in history.history else final_train_loss

    # 6. Save Model and Training History
    save_traffic_model(model, MODEL_PATH)

    history_dict = {
        "loss": [float(x) for x in history.history.get("loss", [])],
        "val_loss": [float(x) for x in history.history.get("val_loss", [])],
        "epochs_trained": len(history.history.get("loss", [])),
        "test_metrics": test_metrics,
        "batch_size": batch_size,
        "sequence_length": sequence_length
    }

    with open(HISTORY_PATH, "w") as f:
        json.dump(history_dict, f, indent=4)

    # 7. Record run in SQLite
    run_id = insert_training_run(
        user_id=user_id,
        username=username,
        epochs=epochs,
        batch_size=batch_size,
        sequence_length=sequence_length,
        train_loss=final_train_loss,
        val_loss=final_val_loss,
        test_rmse=test_metrics["rmse"],
        test_mae=test_metrics["mae"],
        test_r2=test_metrics["r2"],
        test_mape=test_metrics["mape"],
        status="COMPLETED"
    )

    log_event("TRAIN", f"Training completed. Run #{run_id} - R²: {test_metrics['r2']} - RMSE: {test_metrics['rmse']}", username=username)

    return {
        "model": model,
        "history": history_dict,
        "test_metrics": test_metrics,
        "y_test_actual": y_test_actual,
        "y_test_pred": y_test_pred,
        "test_dates": test_dates,
        "run_id": run_id
    }

def get_latest_training_history() -> Optional[Dict[str, Any]]:
    """Reads saved training history and evaluation metrics from disk."""
    if not os.path.exists(HISTORY_PATH):
        return None
    try:
        with open(HISTORY_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return None
