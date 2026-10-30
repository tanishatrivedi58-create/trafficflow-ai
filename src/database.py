"""
Database manager for TrafficFlow AI using SQLite.
Stores user accounts, prediction history, training runs, and audit logs.
"""
import sqlite3
import datetime
from typing import Optional, List, Dict, Any
from src.config import DB_PATH

def get_connection():
    """Returns a SQLite connection with row_factory enabled and WAL mode."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    """Initializes tables for users, predictions, training runs, and audit logs."""
    conn = get_connection()
    with conn:
        # Users Table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                full_name TEXT,
                role TEXT DEFAULT 'analyst',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            );
        """)

        # Prediction History Table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                username TEXT,
                forecast_target_time TEXT,
                predicted_volume REAL NOT NULL,
                actual_volume REAL,
                congestion_level TEXT NOT NULL,
                los_grade TEXT,
                est_speed_mph REAL,
                temp_c REAL,
                weather_desc TEXT,
                is_rush_hour INTEGER,
                source_mode TEXT DEFAULT 'dashboard_forecast',
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        """)

        # Model Training Runs Table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                username TEXT,
                epochs INTEGER,
                batch_size INTEGER,
                sequence_length INTEGER,
                train_loss REAL,
                val_loss REAL,
                test_rmse REAL,
                test_mae REAL,
                test_r2 REAL,
                test_mape REAL,
                status TEXT DEFAULT 'COMPLETED',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # System Activity / Audit Logs Table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level TEXT DEFAULT 'INFO',
                module TEXT,
                message TEXT NOT NULL,
                username TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
    conn.close()

def log_event(module: str, message: str, username: str = "system", level: str = "INFO"):
    """Inserts a structured system log entry."""
    try:
        conn = get_connection()
        with conn:
            conn.execute(
                "INSERT INTO system_logs (level, module, message, username) VALUES (?, ?, ?, ?)",
                (level, module, message, username)
            )
        conn.close()
    except Exception:
        pass

def insert_prediction(
    user_id: Optional[int],
    username: str,
    forecast_target_time: str,
    predicted_volume: float,
    actual_volume: Optional[float],
    congestion_level: str,
    los_grade: str,
    est_speed_mph: float,
    temp_c: float,
    weather_desc: str,
    is_rush_hour: int,
    source_mode: str = "dashboard_forecast",
    notes: str = ""
) -> int:
    """Saves a prediction observation to SQLite history."""
    conn = get_connection()
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO prediction_history (
                user_id, username, forecast_target_time, predicted_volume,
                actual_volume, congestion_level, los_grade, est_speed_mph,
                temp_c, weather_desc, is_rush_hour, source_mode, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, username, forecast_target_time, predicted_volume,
                actual_volume, congestion_level, los_grade, est_speed_mph,
                temp_c, weather_desc, is_rush_hour, source_mode, notes
            )
        )
        record_id = cursor.lastrowid
    conn.close()
    return record_id

def get_prediction_history(limit: int = 100, username: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves prediction history records sorted descending by timestamp."""
    conn = get_connection()
    if username:
        cursor = conn.execute(
            "SELECT * FROM prediction_history WHERE username = ? ORDER BY id DESC LIMIT ?",
            (username, limit)
        )
    else:
        cursor = conn.execute(
            "SELECT * FROM prediction_history ORDER BY id DESC LIMIT ?",
            (limit,)
        )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def insert_training_run(
    user_id: Optional[int],
    username: str,
    epochs: int,
    batch_size: int,
    sequence_length: int,
    train_loss: float,
    val_loss: float,
    test_rmse: float,
    test_mae: float,
    test_r2: float,
    test_mape: float,
    status: str = "COMPLETED"
) -> int:
    """Records a model retraining session."""
    conn = get_connection()
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO training_runs (
                user_id, username, epochs, batch_size, sequence_length,
                train_loss, val_loss, test_rmse, test_mae, test_r2, test_mape, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, username, epochs, batch_size, sequence_length,
                train_loss, val_loss, test_rmse, test_mae, test_r2, test_mape, status
            )
        )
        run_id = cursor.lastrowid
    conn.close()
    return run_id

def get_training_runs(limit: int = 20) -> List[Dict[str, Any]]:
    """Fetches recent model training runs."""
    conn = get_connection()
    cursor = conn.execute("SELECT * FROM training_runs ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_recent_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Fetches recent system logs."""
    conn = get_connection()
    cursor = conn.execute("SELECT * FROM system_logs ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def clear_prediction_history(username: Optional[str] = None):
    """Clears prediction history records."""
    conn = get_connection()
    with conn:
        if username:
            conn.execute("DELETE FROM prediction_history WHERE username = ?", (username,))
        else:
            conn.execute("DELETE FROM prediction_history")
    conn.close()
