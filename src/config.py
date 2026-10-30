"""
TrafficFlow AI - Configuration and Environment Settings
"""
import os

# Set environment variables for TensorFlow and Keras BEFORE any ML imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERAS_DIR = os.path.join(BASE_DIR, ".keras")
os.makedirs(KERAS_DIR, exist_ok=True)
os.environ["KERAS_HOME"] = KERAS_DIR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# File and Directory Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "Metro_Interstate_Traffic_Volume.csv")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "traffic_features.csv")
DB_PATH = os.path.join(DATA_DIR, "trafficflow.db")

MODEL_PATH = os.path.join(MODELS_DIR, "lstm_traffic_model.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "feature_metadata.json")
HISTORY_PATH = os.path.join(MODELS_DIR, "training_history.json")

# Model Architecture and Training Defaults
SEQUENCE_LENGTH = 12  # 12 hours lookback window
BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
LEARNING_RATE = 0.001
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Engineered Features for LSTM
FEATURE_COLUMNS = [
    "traffic_volume",
    "temp_c",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "is_weekend",
    "is_holiday",
    "is_rush_hour",
]
TARGET_COLUMN = "traffic_volume"

# Highway and Congestion Benchmarks (MBA Decision Science Framework)
HIGHWAY_CAPACITY = 7200  # Vehicles/hour nominal max capacity (Interstate-94 segment)
FREE_FLOW_SPEED = 65.0   # mph

CONGESTION_LEVELS = {
    "LOW": {
        "label": "Free Flow",
        "min": 0,
        "max": 2000,
        "color": "#10B981",  # Emerald Green
        "los": "LOS A / B",
        "avg_speed_mph": 62.0,
        "delay_factor": 1.0,
        "severity": "Low",
        "description": "Traffic moves smoothly near free-flow speeds. Minimal driver tension."
    },
    "MODERATE": {
        "label": "Moderate Flow",
        "min": 2000,
        "max": 3800,
        "color": "#3B82F6",  # Indigo/Blue
        "los": "LOS C",
        "avg_speed_mph": 50.0,
        "delay_factor": 1.25,
        "severity": "Moderate",
        "description": "Steady traffic with intermittent deceleration. Manageable flow rate."
    },
    "HEAVY": {
        "label": "Heavy Congestion",
        "min": 3800,
        "max": 5200,
        "color": "#F59E0B",  # Amber
        "los": "LOS D / E",
        "avg_speed_mph": 32.0,
        "delay_factor": 1.85,
        "severity": "High",
        "description": "High vehicle density, unstable speeds, significant stop-and-go delays."
    },
    "SEVERE": {
        "label": "Severe Gridlock",
        "min": 5200,
        "max": 10000,
        "color": "#EF4444",  # Crimson Red
        "los": "LOS F",
        "avg_speed_mph": 16.0,
        "delay_factor": 3.40,
        "severity": "Critical",
        "description": "System operating above practical capacity. Breakdown of flow with severe queues."
    }
}

# Economic & Environmental Cost Parameters
HOURLY_TIME_VALUE_USD = 18.50       # Average passenger value of time per vehicle-hour delay
EXCESS_FUEL_BURN_GAL_HR = 0.65      # Idling fuel consumption during heavy congestion
CO2_KG_PER_GALLON = 8.887           # Standard EPA conversion factor

def get_congestion_info(volume: float) -> dict:
    """Returns congestion tier and MBA traffic metrics for a given volume."""
    vol = float(volume)
    if vol < 2000:
        tier = "LOW"
    elif vol < 3800:
        tier = "MODERATE"
    elif vol < 5200:
        tier = "HEAVY"
    else:
        tier = "SEVERE"
    info = CONGESTION_LEVELS[tier].copy()
    info["tier"] = tier
    info["volume"] = vol
    info["capacity_utilization"] = min(100.0, round((vol / HIGHWAY_CAPACITY) * 100.0, 1))
    return info
