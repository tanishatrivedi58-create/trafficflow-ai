"""
UI styling and reusable components for TrafficFlow AI.
Provides CSS styling, KPI metric cards, badges, and zero-state placeholders.
"""
import streamlit as st
from typing import Dict, Any, Optional

def inject_custom_css():
    """Injects modern dark glassmorphic styling into the Streamlit application."""
    custom_css = """
    <style>
        /* Base typography & theme */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        code, pre {
            font-family: 'JetBrains Mono', monospace;
        }

        /* App container background */
        .stApp {
            background: radial-gradient(circle at 15% 15%, #131d36 0%, #0B1120 100%);
            color: #F8FAFC;
        }

        /* Glassmorphic cards */
        .glass-card {
            background: rgba(30, 41, 59, 0.65);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        
        .glass-card:hover {
            border-color: rgba(59, 130, 246, 0.35);
            box-shadow: 0 12px 36px rgba(14, 165, 233, 0.15);
        }

        /* Metric Hero Card */
        .kpi-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 18px 20px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }

        .kpi-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, #38BDF8, #818CF8);
        }

        .kpi-title {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94A3B8;
            font-weight: 600;
            margin-bottom: 6px;
        }

        .kpi-value {
            font-size: 2.1rem;
            font-weight: 800;
            color: #F8FAFC;
            letter-spacing: -0.02em;
            line-height: 1.1;
        }

        .kpi-sub {
            font-size: 0.8rem;
            color: #64748B;
            margin-top: 6px;
        }

        /* Congestion badges */
        .badge-free-flow {
            background: rgba(16, 185, 129, 0.15);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 20px;
            padding: 4px 12px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
        }

        .badge-moderate {
            background: rgba(59, 130, 246, 0.15);
            color: #60A5FA;
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 20px;
            padding: 4px 12px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
        }

        .badge-heavy {
            background: rgba(245, 158, 11, 0.15);
            color: #FBBF24;
            border: 1px solid rgba(245, 158, 11, 0.3);
            border-radius: 20px;
            padding: 4px 12px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
        }

        .badge-severe {
            background: rgba(239, 68, 68, 0.15);
            color: #F87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 20px;
            padding: 4px 12px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
        }

        /* Hero Header */
        .app-header {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
            border: 1px solid rgba(59, 130, 246, 0.2);
            border-radius: 18px;
            padding: 24px 32px;
            margin-bottom: 24px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: rgba(15, 23, 42, 0.5);
            padding: 6px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .stTabs [data-baseweb="tab"] {
            height: 42px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 8px;
            color: #94A3B8;
            font-weight: 600;
            font-size: 0.9rem;
            padding: 0 18px;
            border: none;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
        }

        /* Buttons */
        .stButton>button {
            border-radius: 10px;
            font-weight: 600;
            letter-spacing: 0.02em;
            padding: 8px 20px;
            transition: all 0.2s ease;
        }

        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.3);
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def render_app_header(user: Optional[Dict[str, Any]] = None):
    """Renders the top branding header with user status."""
    st.markdown("""
    <div class="app-header">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="background: linear-gradient(135deg, #3B82F6, #8B5CF6); width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 22px; box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);">
                        🚦
                    </div>
                    <div>
                        <h1 style="margin: 0; font-size: 1.85rem; font-weight: 800; background: linear-gradient(90deg, #F8FAFC 0%, #93C5FD 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                            TrafficFlow AI
                        </h1>
                        <p style="margin: 2px 0 0 0; color: #94A3B8; font-size: 0.88rem; font-weight: 500;">
                            Smart Traffic Congestion Forecasting & Decision Support System • Interstate-94 Corridor
                        </p>
                    </div>
                </div>
            </div>
            <div style="text-align: right; margin-top: 10px;">
                <span style="background: rgba(59, 130, 246, 0.15); color: #93C5FD; padding: 4px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; border: 1px solid rgba(59, 130, 246, 0.3);">
                    Deep Learning LSTM Engine v2.0
                </span>
                <div style="margin-top: 6px; font-size: 0.8rem; color: #64748B;">
                    MBA Data Science & AI Portfolio Platform
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_zero_state_dashboard():
    """
    Renders the dashboard with all metrics initialized to zero before authentication.
    Fulfills Requirement 3: 'Before authentication, all dashboard metrics are initialized to zero.'
    """
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.5); border: 1px dashed rgba(148, 163, 184, 0.25); border-radius: 14px; padding: 16px 20px; margin-bottom: 24px; display: flex; align-items: center; justify-content: space-between;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.5rem;">🔒</span>
            <div>
                <strong style="color: #F8FAFC; font-size: 0.95rem;">Authentication Required to Stream Live Intelligence</strong>
                <p style="margin: 0; color: #94A3B8; font-size: 0.82rem;">Dashboard metrics are currently zero-initialized. Please authenticate to initialize dataset telemetry and activate real-time LSTM forecasting.</p>
            </div>
        </div>
        <span style="background: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 12px; padding: 4px 10px; font-size: 0.78rem; font-weight: 600;">
            SYSTEM LOCKED
        </span>
    </div>
    """, unsafe_allow_html=True)

    # 4 Zero-Initialized Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Current Observed Volume</div>
            <div class="kpi-value" style="color: #94A3B8;">0 <span style="font-size: 1rem; font-weight: 500;">vph</span></div>
            <div class="kpi-sub">Sensor baseline uninitialized</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Next-Hour Forecast (t+1)</div>
            <div class="kpi-value" style="color: #94A3B8;">0 <span style="font-size: 1rem; font-weight: 500;">vph</span></div>
            <div class="kpi-sub">Awaiting model authorization</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Travel Delay Factor</div>
            <div class="kpi-value" style="color: #94A3B8;">0.00 <span style="font-size: 1rem; font-weight: 500;">x</span></div>
            <div class="kpi-sub">Level of Service: Locked</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Model Accuracy (R²)</div>
            <div class="kpi-value" style="color: #94A3B8;">0.0000</div>
            <div class="kpi-sub">Weights offline</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 40px 20px;">
        <div style="font-size: 2.8rem; margin-bottom: 12px;">📊 ➡️ 🚀</div>
        <h3 style="color: #F8FAFC; margin-bottom: 8px; font-weight: 700;">Welcome to TrafficFlow AI System</h3>
        <p style="color: #94A3B8; max-width: 650px; margin: 0 auto 20px auto; font-size: 0.95rem; line-height: 1.6;">
            A production-ready deep learning platform integrating real Minnesota Department of Transportation 
            Interstate-94 traffic data, recurrent LSTM neural networks, real-time hourly forecasting, 
            and MBA managerial decision scenario simulations.
        </p>
        <p style="color: #38BDF8; font-size: 0.88rem; font-weight: 600;">
            Use the login or registration panel below to unlock real-time intelligence.
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_kpi_cards(
    observed_volume: int,
    predicted_volume: int,
    delta_volume: int,
    delta_pct: float,
    congestion_info: Dict[str, Any],
    r2_score: float
):
    """Renders populated live executive KPI cards."""
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Current Observed Volume</div>
            <div class="kpi-value">{observed_volume:,} <span style="font-size: 1rem; font-weight: 500; color: #94A3B8;">vph</span></div>
            <div class="kpi-sub">Baseline from I-94 Sensor</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        color = congestion_info["color"]
        delta_sign = "+" if delta_volume > 0 else ""
        delta_color = "#EF4444" if delta_volume > 0 else "#10B981"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Next-Hour Forecast (t+1)</div>
            <div class="kpi-value" style="color: {color};">{predicted_volume:,} <span style="font-size: 1rem; font-weight: 500; color: #94A3B8;">vph</span></div>
            <div class="kpi-sub" style="color: {delta_color}; font-weight: 600;">
                {delta_sign}{delta_volume:,} vph ({delta_sign}{delta_pct}%) vs now
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        delay = congestion_info.get("delay_factor", 1.0)
        speed = congestion_info.get("avg_speed_mph", 60.0)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Congestion & Speed Index</div>
            <div class="kpi-value">{speed:.0f} <span style="font-size: 1rem; font-weight: 500; color: #94A3B8;">mph</span></div>
            <div class="kpi-sub">Delay factor: {delay:.2f}x ({congestion_info['los']})</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Model Accuracy (Test R²)</div>
            <div class="kpi-value" style="color: #38BDF8;">{r2_score:.4f}</div>
            <div class="kpi-sub">Deep Learning LSTM (Huber Loss)</div>
        </div>
        """, unsafe_allow_html=True)

def render_congestion_badge(tier: str) -> str:
    """Returns HTML for styled congestion badge."""
    t = tier.upper()
    if t == "LOW":
        return '<span class="badge-free-flow">🟢 FREE FLOW (LOS A/B)</span>'
    elif t == "MODERATE":
        return '<span class="badge-moderate">🔵 MODERATE (LOS C)</span>'
    elif t == "HEAVY":
        return '<span class="badge-heavy">🟡 HEAVY CONGESTION (LOS D/E)</span>'
    else:
        return '<span class="badge-severe">🔴 SEVERE GRIDLOCK (LOS F)</span>'
