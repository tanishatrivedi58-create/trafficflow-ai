"""
Evaluation and visualization module for TrafficFlow AI.
Computes industry-standard regression metrics and produces interactive Plotly charts.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

from src.config import CONGESTION_LEVELS, HIGHWAY_CAPACITY

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes rigorous evaluation metrics: RMSE, MAE, R², MAPE, and Explained Variance.
    """
    y_t = np.asarray(y_true, dtype=np.float64).flatten()
    y_p = np.asarray(y_pred, dtype=np.float64).flatten()

    # Clip negative predictions to 0
    y_p = np.clip(y_p, 0.0, None)

    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mae = float(mean_absolute_error(y_t, y_p))
    r2 = float(r2_score(y_t, y_p))
    ev = float(explained_variance_score(y_t, y_p))

    # MAPE computed on non-trivial volumes (> 100 vehicles/hr) to avoid division distortion
    mask = y_t > 100
    if np.sum(mask) > 0:
        mape = float(np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100.0)
    else:
        mape = 0.0

    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2": round(r2, 4),
        "mape": round(mape, 2),
        "explained_variance": round(ev, 4)
    }

def create_actual_vs_predicted_plot(
    dates: List[Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Test Set: Actual vs. LSTM Predicted Traffic Volume (Interstate-94)"
) -> go.Figure:
    """Creates interactive Plotly line comparison chart."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=y_true,
        mode="lines",
        name="Actual Volume (Observed)",
        line=dict(color="#38BDF8", width=2.5),
        opacity=0.85
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=y_pred,
        mode="lines",
        name="LSTM Predicted",
        line=dict(color="#F43F5E", width=2, dash="dot"),
        opacity=0.9
    ))

    # Add Congestion Reference Lines
    fig.add_hline(
        y=5200, line_dash="dash", line_color="#EF4444",
        annotation_text="Severe Gridlock (>5200 vph)", annotation_position="top left",
        annotation_font=dict(color="#EF4444", size=10)
    )
    fig.add_hline(
        y=3800, line_dash="dash", line_color="#F59E0B",
        annotation_text="Heavy Congestion (>3800 vph)", annotation_position="top left",
        annotation_font=dict(color="#F59E0B", size=10)
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#F8FAFC")),
        xaxis=dict(title="Date & Time", showgrid=True, gridcolor="#334155", color="#94A3B8"),
        yaxis=dict(title="Traffic Volume (Vehicles/Hour)", showgrid=True, gridcolor="#334155", color="#94A3B8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#F8FAFC")),
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def create_residual_distribution_plot(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Plots histogram and KDE-like analysis of model prediction residuals (errors)."""
    residuals = np.asarray(y_true).flatten() - np.asarray(y_pred).flatten()
    mean_res = float(np.mean(residuals))
    std_res = float(np.std(residuals))

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=50,
        name="Residuals (Error)",
        marker=dict(color="#8B5CF6", line=dict(color="#6D28D9", width=1)),
        opacity=0.75
    ))

    fig.add_vline(
        x=mean_res, line_dash="dash", line_color="#10B981",
        annotation_text=f"Mean Error: {mean_res:+.1f}", annotation_position="top right",
        annotation_font=dict(color="#10B981")
    )

    fig.update_layout(
        title=dict(text=f"Residual Error Distribution (Std Dev: {std_res:.1f} vph)", font=dict(size=16, color="#F8FAFC")),
        xaxis=dict(title="Residual Error (Actual - Predicted)", showgrid=True, gridcolor="#334155", color="#94A3B8"),
        yaxis=dict(title="Frequency", showgrid=True, gridcolor="#334155", color="#94A3B8"),
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def create_scatter_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, r2: float) -> go.Figure:
    """Generates scatter plot comparing actual versus predicted with 45-degree reference line."""
    sample_size = min(3000, len(y_true))
    idx = np.random.choice(len(y_true), sample_size, replace=False) if len(y_true) > 3000 else np.arange(len(y_true))
    
    yt_sample = np.asarray(y_true)[idx]
    yp_sample = np.asarray(y_pred)[idx]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yt_sample,
        y=yp_sample,
        mode="markers",
        name="Observations",
        marker=dict(size=4, color="#38BDF8", opacity=0.45)
    ))

    max_val = max(float(np.max(yt_sample)), float(np.max(yp_sample)))
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode="lines",
        name="Ideal Fit (y = x)",
        line=dict(color="#10B981", width=2, dash="dash")
    ))

    fig.update_layout(
        title=dict(text=f"Regression Fit (R² = {r2:.4f})", font=dict(size=16, color="#F8FAFC")),
        xaxis=dict(title="Actual Traffic Volume (vph)", showgrid=True, gridcolor="#334155", color="#94A3B8"),
        yaxis=dict(title="Predicted Traffic Volume (vph)", showgrid=True, gridcolor="#334155", color="#94A3B8"),
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def create_training_loss_plot(history: Dict[str, List[float]]) -> go.Figure:
    """Plots training and validation loss curves across epochs."""
    fig = go.Figure()
    epochs = list(range(1, len(history.get("loss", [])) + 1))

    fig.add_trace(go.Scatter(
        x=epochs,
        y=history.get("loss", []),
        mode="lines+markers",
        name="Training Loss (Huber)",
        line=dict(color="#3B82F6", width=2.5)
    ))

    if "val_loss" in history:
        fig.add_trace(go.Scatter(
            x=epochs,
            y=history.get("val_loss", []),
            mode="lines+markers",
            name="Validation Loss",
            line=dict(color="#F59E0B", width=2.5, dash="dash")
        ))

    fig.update_layout(
        title=dict(text="Neural Network Convergence (Epoch Loss)", font=dict(size=16, color="#F8FAFC")),
        xaxis=dict(title="Epoch", showgrid=True, gridcolor="#334155", color="#94A3B8", tickmode="linear", dtick=1),
        yaxis=dict(title="Loss Value", showgrid=True, gridcolor="#334155", color="#94A3B8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#F8FAFC")),
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def create_forecast_horizon_plot(
    past_dates: List[Any],
    past_values: List[float],
    future_dates: List[Any],
    future_forecast: List[float],
    lower_bound: Optional[List[float]] = None,
    upper_bound: Optional[List[float]] = None
) -> go.Figure:
    """
    Renders 24-hour future forecast trajectory with past context and 95% confidence band.
    """
    fig = go.Figure()

    # Past Context
    fig.add_trace(go.Scatter(
        x=past_dates,
        y=past_values,
        mode="lines+markers",
        name="Observed History",
        line=dict(color="#94A3B8", width=2),
        marker=dict(size=5)
    ))

    # Confidence Interval Upper Bound
    if upper_bound is not None:
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=upper_bound,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))

    # Confidence Interval Lower Bound filled to upper
    if lower_bound is not None and upper_bound is not None:
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=lower_bound,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(59, 130, 246, 0.2)",
            name="95% Confidence Band",
            hoverinfo="skip"
        ))

    # Future Forecast Point Line
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_forecast,
        mode="lines+markers",
        name="AI Forecast (t+1 ... t+24)",
        line=dict(color="#3B82F6", width=3),
        marker=dict(size=7, color="#60A5FA")
    ))

    # Horizontal Congestion Thresholds
    fig.add_hline(y=5200, line_dash="dot", line_color="#EF4444", annotation_text="Gridlock (5200 vph)", annotation_font=dict(color="#EF4444", size=10))
    fig.add_hline(y=3800, line_dash="dot", line_color="#F59E0B", annotation_text="Heavy (3800 vph)", annotation_font=dict(color="#F59E0B", size=10))
    fig.add_hline(y=2000, line_dash="dot", line_color="#10B981", annotation_text="Moderate (2000 vph)", annotation_font=dict(color="#10B981", size=10))

    fig.update_layout(
        title=dict(text="Rolling 24-Hour Congestion Forecast with Uncertainty Interval", font=dict(size=16, color="#F8FAFC")),
        xaxis=dict(title="Timeline", showgrid=True, gridcolor="#334155", color="#94A3B8"),
        yaxis=dict(title="Traffic Volume (vph)", showgrid=True, gridcolor="#334155", color="#94A3B8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#F8FAFC")),
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def create_congestion_gauge(volume: float, title: str = "Forecasted Congestion Intensity") -> go.Figure:
    """Renders an intuitive speedometer gauge for executive decision makers."""
    vol = float(np.clip(volume, 0, 7200))
    
    if vol < 2000:
        bar_color = "#10B981"
    elif vol < 3800:
        bar_color = "#3B82F6"
    elif vol < 5200:
        bar_color = "#F59E0B"
    else:
        bar_color = "#EF4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=vol,
        title={'text': title, 'font': {'size': 15, 'color': '#F8FAFC'}},
        number={'suffix': " vph", 'font': {'size': 24, 'color': '#F8FAFC'}},
        gauge={
            'axis': {'range': [0, 7200], 'tickwidth': 1, 'tickcolor': "#94A3B8", 'tickfont': {'color': "#94A3B8"}},
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': "rgba(30, 41, 59, 0.8)",
            'borderwidth': 1,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 2000], 'color': "rgba(16, 185, 129, 0.25)"},
                {'range': [2000, 3800], 'color': "rgba(59, 130, 246, 0.25)"},
                {'range': [3800, 5200], 'color': "rgba(245, 158, 11, 0.25)"},
                {'range': [5200, 7200], 'color': "rgba(239, 68, 68, 0.3)"}
            ],
            'threshold': {
                'line': {'color': "#EF4444", 'width': 3},
                'thickness': 0.8,
                'value': 5200
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC"),
        height=220,
        margin=dict(l=25, r=25, t=35, b=10)
    )
    return fig

def create_diurnal_traffic_profile(df: pd.DataFrame) -> go.Figure:
    """Renders diurnal 24-hour average traffic profiles comparing Weekdays and Weekends."""
    weekday_df = df[df['is_weekend'] == 0].groupby('hour')['traffic_volume'].mean().reset_index()
    weekend_df = df[df['is_weekend'] == 1].groupby('hour')['traffic_volume'].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekday_df['hour'],
        y=weekday_df['traffic_volume'],
        mode="lines+markers",
        name="Weekday (Mon - Fri)",
        line=dict(color="#38BDF8", width=3),
        marker=dict(size=6)
    ))
    fig.add_trace(go.Scatter(
        x=weekend_df['hour'],
        y=weekend_df['traffic_volume'],
        mode="lines+markers",
        name="Weekend (Sat - Sun)",
        line=dict(color="#A855F7", width=3, dash="dash"),
        marker=dict(size=6)
    ))

    # Highlight typical morning & evening peaks
    fig.add_vrect(x0=7, x1=9, fillcolor="rgba(245, 158, 11, 0.15)", layer="below", line_width=0, annotation_text="AM Peak", annotation_position="top left", annotation_font=dict(color="#F59E0B", size=10))
    fig.add_vrect(x0=16, x1=18, fillcolor="rgba(239, 68, 68, 0.15)", layer="below", line_width=0, annotation_text="PM Peak", annotation_position="top left", annotation_font=dict(color="#EF4444", size=10))

    fig.update_layout(
        title=dict(text="Average Hourly Diurnal Traffic Profiles (Commute vs. Weekend)", font=dict(size=16, color="#F8FAFC")),
        xaxis=dict(title="Hour of Day (0 - 23)", dtick=2, showgrid=True, gridcolor="#334155", color="#94A3B8"),
        yaxis=dict(title="Average Traffic Volume (vph)", showgrid=True, gridcolor="#334155", color="#94A3B8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#F8FAFC")),
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def create_congestion_heatmap(df: pd.DataFrame) -> go.Figure:
    """Generates day-of-week by hour congestion intensity heatmap."""
    pivot = df.pivot_table(
        index='dayofweek',
        columns='hour',
        values='traffic_volume',
        aggfunc='mean'
    ).round(0)

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot.index = [day_names[i] for i in pivot.index]

    fig = px.imshow(
        pivot,
        labels=dict(x="Hour of Day", y="Day of Week", color="Mean Volume (vph)"),
        x=list(range(24)),
        y=day_names,
        color_continuous_scale="Plasma",
        aspect="auto"
    )

    fig.update_layout(
        title=dict(text="Weekly Congestion Heatmap Matrix (I-94 Corridor)", font=dict(size=16, color="#F8FAFC")),
        xaxis=dict(title="Hour of Day", dtick=2, color="#94A3B8"),
        yaxis=dict(title="Day of Week", color="#94A3B8"),
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        coloraxis_colorbar=dict(title="Volume (vph)", tickfont=dict(color="#94A3B8"), titlefont=dict(color="#F8FAFC")),
        margin=dict(l=60, r=40, t=60, b=40)
    )
    return fig
