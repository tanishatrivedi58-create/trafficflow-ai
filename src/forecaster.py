"""
Forecasting engine and MBA What-If Simulator for TrafficFlow AI.
Generates real-time next-hour forecasts, rolling 24-hour trajectories with
confidence intervals, and economic impact simulations.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, List, Optional, Tuple
import datetime

from src.config import (
    MODEL_PATH, SCALER_PATH, METADATA_PATH, HISTORY_PATH,
    SEQUENCE_LENGTH, FEATURE_COLUMNS, TARGET_COLUMN,
    HOURLY_TIME_VALUE_USD, EXCESS_FUEL_BURN_GAL_HR, CO2_KG_PER_GALLON,
    get_congestion_info
)
from src.lstm_model import load_traffic_model
from src.data_pipeline import load_processed_data

def get_artifacts() -> Tuple[Any, Any, Dict[str, Any]]:
    """Loads model, scaler, and metadata."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Model or scaler artifacts are missing. Please initialize or train the model.")
    
    model = load_traffic_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return model, scaler, metadata

def predict_next_hour(
    recent_sequence_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Generates a forecast for the immediate next hour (t+1) based on the last 12 hours.
    """
    model, scaler, metadata = get_artifacts()
    feature_cols = metadata["feature_columns"]
    seq_len = metadata.get("sequence_length", SEQUENCE_LENGTH)
    target_min = metadata["target_min"]
    target_max = metadata["target_max"]

    if recent_sequence_df is None:
        df = load_processed_data()
        recent_sequence_df = df.tail(seq_len)

    if len(recent_sequence_df) < seq_len:
        raise ValueError(f"Need at least {seq_len} rows of history to predict, got {len(recent_sequence_df)}")

    # Extract features and scale
    input_features = recent_sequence_df[feature_cols].tail(seq_len).values.astype(np.float32)
    input_scaled = scaler.transform(input_features)
    
    # Reshape for LSTM: (1, seq_len, num_features)
    X_input = np.expand_dims(input_scaled, axis=0)

    # Inference
    pred_scaled = float(model.predict(X_input, verbose=0)[0, 0])
    pred_volume = float(np.clip(pred_scaled * (target_max - target_min) + target_min, 0.0, None))
    pred_volume = round(pred_volume, 0)

    # Congestion metadata
    congestion_info = get_congestion_info(pred_volume)

    # Calculate last observed volume
    last_observed_volume = float(recent_sequence_df['traffic_volume'].iloc[-1])
    last_timestamp = recent_sequence_df['date_time'].iloc[-1]
    
    if isinstance(last_timestamp, str):
        last_dt = pd.to_datetime(last_timestamp)
    else:
        last_dt = last_timestamp
    next_timestamp = last_dt + pd.Timedelta(hours=1)

    delta_volume = round(pred_volume - last_observed_volume, 0)
    delta_pct = round((delta_volume / max(1.0, last_observed_volume)) * 100.0, 1)

    return {
        "forecast_timestamp": str(next_timestamp),
        "last_observed_timestamp": str(last_dt),
        "predicted_volume": int(pred_volume),
        "last_observed_volume": int(last_observed_volume),
        "delta_volume": int(delta_volume),
        "delta_pct": delta_pct,
        "congestion_tier": congestion_info["tier"],
        "congestion_label": congestion_info["label"],
        "congestion_color": congestion_info["color"],
        "los": congestion_info["los"],
        "avg_speed_mph": congestion_info["avg_speed_mph"],
        "capacity_utilization": congestion_info["capacity_utilization"],
        "severity": congestion_info["severity"],
        "description": congestion_info["description"]
    }

def predict_rolling_horizon(
    recent_sequence_df: Optional[pd.DataFrame] = None,
    horizon_hours: int = 24
) -> Dict[str, Any]:
    """
    Generates an autoregressive multi-step rolling forecast for future hours (e.g. 24h)
    with 95% confidence intervals derived from model test RMSE.
    """
    model, scaler, metadata = get_artifacts()
    feature_cols = metadata["feature_columns"]
    seq_len = metadata.get("sequence_length", SEQUENCE_LENGTH)
    target_min = metadata["target_min"]
    target_max = metadata["target_max"]

    # Retrieve RMSE from history for uncertainty bounds
    rmse = 450.0  # sensible default
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r") as f:
                hist = json.load(f)
                rmse = float(hist.get("test_metrics", {}).get("rmse", 450.0))
        except Exception:
            pass

    if recent_sequence_df is None:
        df = load_processed_data()
        recent_sequence_df = df.tail(seq_len)

    current_window = recent_sequence_df[feature_cols].tail(seq_len).values.copy().astype(np.float32)
    last_timestamp = pd.to_datetime(recent_sequence_df['date_time'].iloc[-1])

    future_dates = []
    future_forecasts = []
    lower_bounds = []
    upper_bounds = []

    running_dt = last_timestamp

    for step in range(1, horizon_hours + 1):
        running_dt += pd.Timedelta(hours=1)
        future_dates.append(running_dt.strftime("%Y-%m-%d %H:00"))

        # Scale current window
        window_scaled = scaler.transform(current_window)
        X_step = np.expand_dims(window_scaled, axis=0)

        # Predict next volume
        pred_scaled = float(model.predict(X_step, verbose=0)[0, 0])
        pred_vol = float(np.clip(pred_scaled * (target_max - target_min) + target_min, 0.0, None))
        pred_vol = round(pred_vol, 0)
        future_forecasts.append(int(pred_vol))

        # Expanding uncertainty band over rolling horizon: sigma_t = RMSE * sqrt(1 + 0.05 * step)
        step_sigma = rmse * np.sqrt(1.0 + 0.05 * step)
        lower_bounds.append(int(max(0.0, pred_vol - 1.96 * step_sigma)))
        upper_bounds.append(int(min(7500.0, pred_vol + 1.96 * step_sigma)))

        # Autoregressively roll window forward:
        # Build next row features
        hour = running_dt.hour
        dayofweek = running_dt.dayofweek
        hour_sin = np.round(np.sin(2 * np.pi * hour / 24.0), 4)
        hour_cos = np.round(np.cos(2 * np.pi * hour / 24.0), 4)
        day_sin = np.round(np.sin(2 * np.pi * dayofweek / 7.0), 4)
        day_cos = np.round(np.cos(2 * np.pi * dayofweek / 7.0), 4)
        is_weekend = 1 if dayofweek >= 5 else 0
        is_rush_hour = 1 if (is_weekend == 0 and ((7 <= hour <= 9) or (16 <= hour <= 18))) else 0
        
        # Keep recent average weather
        recent_temp = float(current_window[-1, 1])
        recent_rain = float(current_window[-1, 2])
        recent_snow = float(current_window[-1, 3])
        recent_clouds = float(current_window[-1, 4])
        is_holiday = float(current_window[-1, 10])

        next_row = np.array([
            pred_vol, recent_temp, recent_rain, recent_snow, recent_clouds,
            hour_sin, hour_cos, day_sin, day_cos, is_weekend, is_holiday, is_rush_hour
        ], dtype=np.float32)

        # Shift window: drop oldest, append next_row
        current_window = np.vstack([current_window[1:], next_row])

    # Past context
    past_dates = [pd.to_datetime(d).strftime("%Y-%m-%d %H:00") for d in recent_sequence_df['date_time'].tail(seq_len)]
    past_values = [int(v) for v in recent_sequence_df['traffic_volume'].tail(seq_len)]

    return {
        "past_dates": past_dates,
        "past_values": past_values,
        "future_dates": future_dates,
        "future_forecast": future_forecasts,
        "lower_bound": lower_bounds,
        "upper_bound": upper_bounds,
        "peak_predicted_volume": max(future_forecasts),
        "peak_hour": future_dates[future_forecasts.index(max(future_forecasts))]
    }

def simulate_what_if_scenario(
    temp_c: float,
    rain_mm: float,
    snow_mm: float,
    clouds_pct: float,
    hour: int,
    day_of_week: int,
    is_holiday: bool
) -> Dict[str, Any]:
    """
    Managerial Decision Simulator (MBA Operations Research).
    Simulates traffic conditions under specific weather, temporal, and holiday stressors.
    Calculates congestion level, delay index, and economic/environmental costs.
    """
    model, scaler, metadata = get_artifacts()
    feature_cols = metadata["feature_columns"]
    seq_len = metadata.get("sequence_length", SEQUENCE_LENGTH)
    target_min = metadata["target_min"]
    target_max = metadata["target_max"]

    df = load_processed_data()
    base_window = df[feature_cols].tail(seq_len).values.copy().astype(np.float32)

    # Compute cyclical terms
    hour_sin = np.round(np.sin(2 * np.pi * hour / 24.0), 4)
    hour_cos = np.round(np.cos(2 * np.pi * hour / 24.0), 4)
    day_sin = np.round(np.sin(2 * np.pi * day_of_week / 7.0), 4)
    day_cos = np.round(np.cos(2 * np.pi * day_of_week / 7.0), 4)
    is_weekend = 1 if day_of_week >= 5 else 0
    is_rush_hour = 1 if (is_weekend == 0 and ((7 <= hour <= 9) or (16 <= hour <= 18))) else 0
    holiday_flag = 1 if is_holiday else 0

    # Build hypothetical target row
    sim_row = np.array([
        float(base_window[-1, 0]),  # previous volume
        float(temp_c),
        float(rain_mm),
        float(snow_mm),
        float(clouds_pct),
        hour_sin, hour_cos, day_sin, day_cos,
        is_weekend, holiday_flag, is_rush_hour
    ], dtype=np.float32)

    # Inject into simulation sequence
    sim_sequence = np.vstack([base_window[1:], sim_row])
    sim_scaled = scaler.transform(sim_sequence)
    X_sim = np.expand_dims(sim_scaled, axis=0)

    # Infer
    pred_scaled = float(model.predict(X_sim, verbose=0)[0, 0])
    sim_volume = float(np.clip(pred_scaled * (target_max - target_min) + target_min, 0.0, None))
    sim_volume = round(sim_volume, 0)

    # Congestion & Operational Economics
    info = get_congestion_info(sim_volume)
    delay_factor = info["delay_factor"]
    avg_speed = info["avg_speed_mph"]

    # Baseline comparison (normal conditions at same hour)
    baseline_speed = 60.0
    excess_delay_hours = max(0.0, (delay_factor - 1.0) * sim_volume)
    economic_loss_usd = round(excess_delay_hours * HOURLY_TIME_VALUE_USD, 2)
    fuel_wasted_gal = round(excess_delay_hours * EXCESS_FUEL_BURN_GAL_HR, 2)
    co2_kg = round(fuel_wasted_gal * CO2_KG_PER_GALLON, 2)

    # MBA Mitigation Recommendations
    recommendations = []
    if info["tier"] == "SEVERE":
        recommendations.append("Activate Variable Message Signs (VMS) with alternate corridor diversions.")
        recommendations.append("Implement dynamic toll pricing / congestion surcharge (+35%).")
        recommendations.append("Pre-position rapid incident-clearance tow trucks at interchanges.")
        recommendations.append("Surge municipal public transit frequency by 25%.")
    elif info["tier"] == "HEAVY":
        recommendations.append("Activate highway ramp metering to meter inflow rates.")
        recommendations.append("Encourage employer flex-hour commuting programs.")
        recommendations.append("Broadcast advisory travel time warnings via 511 traffic apps.")
    elif info["tier"] == "MODERATE":
        recommendations.append("Maintain standard ramp metering and monitor incident queues.")
    else:
        recommendations.append("Free flow operations. Maintain scheduled preventative corridor maintenance.")

    if rain_mm > 10.0:
        recommendations.append("Adverse Wet Weather: Post advisory speed limit reductions (-10 mph) for hydroplaning risk.")
    if snow_mm > 2.0 or temp_c < -5.0:
        recommendations.append("Adverse Winter Weather: Dispatch DOT salt/plow spreaders to prevent bridge freezing.")

    return {
        "simulated_volume": int(sim_volume),
        "congestion_tier": info["tier"],
        "congestion_label": info["label"],
        "congestion_color": info["color"],
        "los": info["los"],
        "avg_speed_mph": avg_speed,
        "capacity_utilization": info["capacity_utilization"],
        "excess_delay_hours": round(excess_delay_hours, 1),
        "economic_loss_usd": economic_loss_usd,
        "fuel_wasted_gal": fuel_wasted_gal,
        "co2_kg": co2_kg,
        "recommendations": recommendations
    }
