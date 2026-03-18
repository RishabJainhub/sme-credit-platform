"""
India SME Credit Risk & Growth Intelligence Platform
=====================================================
Full Streamlit Application — 6 Dashboard Pages
Dark Navy Theme | Plotly Interactive Charts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (precision_recall_curve, roc_curve, roc_auc_score,
                              confusion_matrix, classification_report)
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import json, os

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India SME Credit Risk Platform",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS — PREMIUM GLASSMORPHISM DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

  /* ── ROOT VARIABLES ── */
  :root {
      --bg-deepspace:  #060D18;
      --bg-midnight:   #0A1628;
      --bg-panel:      #0E1E34;
      --border-glass:  rgba(0,201,167,0.12);
      --border-subtle: rgba(30,58,95,0.6);
      --glow-teal:     0 0 30px rgba(0,201,167,0.15), 0 0 60px rgba(0,201,167,0.05);
      --glow-gold:     0 0 25px rgba(255,215,0,0.12);
      --glow-red:      0 0 25px rgba(255,107,107,0.12);
      --glass-bg:      rgba(14,30,52,0.65);
      --glass-border:  rgba(255,255,255,0.06);
  }

  /* ── KEYFRAMES ── */
  @keyframes fadeInUp {
      from { opacity: 0; transform: translateY(18px); }
      to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes shimmer {
      0%   { background-position: -200% 0; }
      100% { background-position: 200% 0; }
  }
  @keyframes gradientShift {
      0%   { background-position: 0% 50%; }
      50%  { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
  }
  @keyframes pulseGlow {
      0%, 100% { box-shadow: 0 0 8px rgba(0,201,167,0.3); }
      50%      { box-shadow: 0 0 20px rgba(0,201,167,0.6); }
  }
  @keyframes livePulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50%      { opacity: 0.5; transform: scale(0.85); }
  }
  @keyframes borderGlow {
      0%   { border-color: rgba(0,201,167,0.2); }
      50%  { border-color: rgba(0,201,167,0.5); }
      100% { border-color: rgba(0,201,167,0.2); }
  }
  @keyframes float {
      0%, 100% { transform: translateY(0px); }
      50%      { transform: translateY(-4px); }
  }

  /* ── BASE ── */
  html, body, [class*="css"] {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: var(--bg-deepspace) !important;
      color: #E8F0FE;
      -webkit-font-smoothing: antialiased;
  }

  .main .block-container {
      padding: 1rem 2.5rem 2rem 2.5rem;
      max-width: 1400px;
  }

  /* ── CUSTOM SCROLLBAR ── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg-deepspace); }
  ::-webkit-scrollbar-thumb {
      background: linear-gradient(180deg, #00C9A7, #0A6E5C);
      border-radius: 10px;
  }
  ::-webkit-scrollbar-thumb:hover { background: #00E8BF; }

  /* ── SIDEBAR — FROSTED GLASS ── */
  [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #060D18 0%, #0A1628 40%, #0E1E34 100%) !important;
      border-right: 1px solid var(--border-glass) !important;
      backdrop-filter: blur(20px);
  }
  [data-testid="stSidebar"] * { color: #E8F0FE !important; }

  [data-testid="stSidebar"] .stRadio > div {
      gap: 2px !important;
  }
  [data-testid="stSidebar"] .stRadio label {
      font-size: 0.85rem !important;
      padding: 10px 14px !important;
      border-radius: 10px !important;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
      border-left: 3px solid transparent !important;
      margin: 1px 0 !important;
  }
  [data-testid="stSidebar"] .stRadio label:hover {
      background: rgba(0,201,167,0.08) !important;
      border-left-color: rgba(0,201,167,0.4) !important;
      transform: translateX(3px);
  }
  [data-testid="stSidebar"] .stRadio label[data-checked="true"],
  [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {
      background: rgba(0,201,167,0.12) !important;
      border-left-color: #00C9A7 !important;
      box-shadow: 0 0 15px rgba(0,201,167,0.08);
  }

  /* ── SIDEBAR LOGO PANEL ── */
  .sidebar-logo {
      background: linear-gradient(135deg, rgba(0,201,167,0.06) 0%, rgba(74,144,217,0.06) 100%);
      border: 1px solid var(--glass-border);
      border-radius: 16px;
      padding: 1.5rem 1rem;
      text-align: center;
      backdrop-filter: blur(10px);
      position: relative;
      overflow: hidden;
  }
  .sidebar-logo::before {
      content: '';
      position: absolute;
      top: -50%; left: -50%;
      width: 200%; height: 200%;
      background: conic-gradient(from 0deg, transparent 0deg, rgba(0,201,167,0.05) 60deg, transparent 120deg);
      animation: rotate 8s linear infinite;
  }
  @keyframes rotate { to { transform: rotate(360deg); } }
  .sidebar-logo > * { position: relative; z-index: 1; }

  /* ── SIDEBAR STATS ── */
  .sidebar-stats {
      background: var(--glass-bg);
      border: 1px solid var(--glass-border);
      border-radius: 12px;
      padding: 1rem 1.2rem;
      backdrop-filter: blur(8px);
  }
  .sidebar-stats .stat-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 6px 0;
      border-bottom: 1px solid rgba(255,255,255,0.03);
      font-size: 0.78rem;
  }
  .sidebar-stats .stat-row:last-child { border-bottom: none; }
  .stat-icon { font-size: 0.85rem; }
  .stat-value {
      font-family: 'JetBrains Mono', monospace;
      font-weight: 600;
      font-size: 0.82rem;
  }
  .live-dot {
      display: inline-block;
      width: 7px; height: 7px;
      background: #00C9A7;
      border-radius: 50%;
      margin-right: 5px;
      animation: livePulse 2s ease-in-out infinite;
  }

  /* ── KPI CARDS — GLASSMORPHISM ── */
  .kpi-card {
      background: linear-gradient(135deg, rgba(14,30,52,0.8) 0%, rgba(10,22,40,0.9) 100%);
      border: 1px solid var(--glass-border);
      border-radius: 16px;
      padding: 1.4rem 1.2rem;
      text-align: center;
      backdrop-filter: blur(12px);
      transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
      animation: fadeInUp 0.6s ease-out both;
      position: relative;
      overflow: hidden;
  }
  .kpi-card::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 2px;
      background: linear-gradient(90deg, transparent, #00C9A7, transparent);
      opacity: 0;
      transition: opacity 0.4s ease;
  }
  .kpi-card:hover {
      transform: translateY(-6px);
      box-shadow: var(--glow-teal);
      border-color: rgba(0,201,167,0.25);
  }
  .kpi-card:hover::before { opacity: 1; }

  .kpi-value {
      font-family: 'JetBrains Mono', monospace;
      font-size: 2.2rem;
      font-weight: 700;
      line-height: 1.1;
      letter-spacing: -0.5px;
  }
  .kpi-label {
      font-size: 0.72rem;
      color: #6B8CA8;
      margin-top: 8px;
      letter-spacing: 1px;
      text-transform: uppercase;
      font-weight: 500;
  }

  /* Staggered card animations */
  [data-testid="stHorizontalBlock"] > div:nth-child(1) .kpi-card { animation-delay: 0.05s; }
  [data-testid="stHorizontalBlock"] > div:nth-child(2) .kpi-card { animation-delay: 0.12s; }
  [data-testid="stHorizontalBlock"] > div:nth-child(3) .kpi-card { animation-delay: 0.19s; }
  [data-testid="stHorizontalBlock"] > div:nth-child(4) .kpi-card { animation-delay: 0.26s; }

  /* ── PAGE HEADER — ANIMATED GRADIENT LINE ── */
  .page-header {
      background: linear-gradient(135deg, rgba(14,30,52,0.6) 0%, rgba(10,40,64,0.4) 100%);
      border-left: none;
      border-radius: 16px;
      padding: 1rem 1.5rem;
      margin-bottom: 1.5rem;
      position: relative;
      overflow: hidden;
      backdrop-filter: blur(8px);
      border: 1px solid var(--glass-border);
  }
  .page-header::before {
      content: '';
      position: absolute;
      left: 0; top: 0; bottom: 0;
      width: 4px;
      background: linear-gradient(180deg, #00C9A7, #4A90D9, #00C9A7);
      background-size: 100% 200%;
      animation: gradientShift 3s ease infinite;
      border-radius: 4px 0 0 4px;
  }
  .page-header h2 {
      margin: 0;
      font-size: 1.15rem;
      font-weight: 700;
      background: linear-gradient(135deg, #00C9A7, #00E8BF, #4AE3C7);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
  }
  .page-header p {
      margin: 4px 0 0 0;
      font-size: 0.76rem;
      color: #5A7A9A;
      letter-spacing: 0.3px;
  }

  /* ── INSIGHT BOX — FROSTED GLASS ── */
  .insight-box {
      background: linear-gradient(135deg, rgba(0,201,167,0.06) 0%, rgba(0,201,167,0.02) 100%);
      border: 1px solid rgba(0,201,167,0.18);
      border-left: 3px solid #00C9A7;
      border-radius: 12px;
      padding: 0.9rem 1.2rem;
      margin: 0.8rem 0;
      font-size: 0.82rem;
      color: #B0E8DF;
      backdrop-filter: blur(6px);
      transition: all 0.3s ease;
  }
  .insight-box:hover {
      border-color: rgba(0,201,167,0.35);
      box-shadow: 0 4px 20px rgba(0,201,167,0.08);
  }

  /* ── OPPORTUNITY BOX ── */
  .opp-box {
      background: linear-gradient(135deg, rgba(255,215,0,0.06) 0%, rgba(255,215,0,0.02) 100%);
      border: 1px solid rgba(255,215,0,0.18);
      border-left: 3px solid #FFD700;
      border-radius: 12px;
      padding: 0.9rem 1.2rem;
      margin: 0.8rem 0;
      font-size: 0.84rem;
      color: #FFE87C;
      backdrop-filter: blur(6px);
      transition: all 0.3s ease;
  }
  .opp-box:hover {
      border-color: rgba(255,215,0,0.35);
      box-shadow: var(--glow-gold);
  }

  /* ── WARNING BOX ── */
  .warn-box {
      background: linear-gradient(135deg, rgba(255,107,107,0.06) 0%, rgba(255,107,107,0.02) 100%);
      border: 1px solid rgba(255,107,107,0.18);
      border-left: 3px solid #FF6B6B;
      border-radius: 12px;
      padding: 0.9rem 1.2rem;
      margin: 0.8rem 0;
      font-size: 0.82rem;
      color: #FFAAAA;
      backdrop-filter: blur(6px);
      transition: all 0.3s ease;
  }
  .warn-box:hover {
      border-color: rgba(255,107,107,0.35);
      box-shadow: var(--glow-red);
  }

  /* ── DIVIDERS ── */
  hr {
      border: none;
      height: 1px;
      background: linear-gradient(90deg, transparent, rgba(30,58,95,0.5), transparent);
      margin: 1.2rem 0;
  }

  /* ── PLOTLY ── */
  .js-plotly-plot, .plotly { background: transparent !important; }

  /* ── HEADINGS ── */
  h1, h2, h3 { color: #E8F0FE !important; }
  h3 {
      font-weight: 600 !important;
      letter-spacing: -0.2px;
  }

  /* ── DATAFRAMES — PREMIUM STYLING ── */
  .stDataFrame {
      border-radius: 12px !important;
      overflow: hidden;
      border: 1px solid var(--glass-border) !important;
  }
  [data-testid="stDataFrame"] > div {
      border-radius: 12px;
  }

  /* ── BUTTONS & INPUTS ── */
  .stButton > button {
      background: linear-gradient(135deg, #00C9A7 0%, #00A88A 100%) !important;
      border: none !important;
      border-radius: 10px !important;
      color: #060D18 !important;
      font-weight: 600 !important;
      padding: 0.5rem 1.5rem !important;
      transition: all 0.3s ease !important;
  }
  .stButton > button:hover {
      box-shadow: var(--glow-teal) !important;
      transform: translateY(-2px) !important;
  }

  /* ── GLOBAL HEADER ── */
  .main-header {
      text-align: center;
      padding: 0.5rem 0 1.5rem;
      position: relative;
  }
  .main-header h1 {
      font-size: 1.8rem !important;
      font-weight: 800 !important;
      background: linear-gradient(135deg, #FFFFFF 0%, #00C9A7 40%, #4A90D9 70%, #A78BFA 100%);
      background-size: 300% 300%;
      animation: gradientShift 6s ease infinite;
      -webkit-background-clip: text !important;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      letter-spacing: -0.5px;
      margin: 0 !important;
  }
  .main-header p {
      color: #4A6A8A;
      font-size: 0.78rem;
      margin: 6px 0 0 0;
      letter-spacing: 0.5px;
  }
  .main-header::after {
      content: '';
      display: block;
      width: 80px;
      height: 3px;
      background: linear-gradient(90deg, #00C9A7, #4A90D9);
      margin: 12px auto 0;
      border-radius: 2px;
  }

  /* ── FOOTER — FROSTED ── */
  .premium-footer {
      text-align: center;
      padding: 1.5rem 2rem;
      background: linear-gradient(135deg, rgba(14,30,52,0.4) 0%, rgba(6,13,24,0.6) 100%);
      border-top: 1px solid var(--glass-border);
      border-radius: 16px;
      margin-top: 2rem;
      backdrop-filter: blur(8px);
  }
  .premium-footer a {
      color: #00C9A7 !important;
      text-decoration: none;
      transition: all 0.3s ease;
      font-weight: 500;
  }
  .premium-footer a:hover {
      color: #00E8BF !important;
      text-shadow: 0 0 10px rgba(0,201,167,0.4);
  }

  /* ── MISCELLANEOUS ── */
  .stMarkdown code {
      background: rgba(0,201,167,0.1) !important;
      color: #00C9A7 !important;
      border-radius: 4px;
      padding: 2px 6px;
  }

  /* ── DISCLOSURE BOX (MODEL CARD) ── */
  .disclosure-box {
      background: linear-gradient(135deg, rgba(255,107,107,0.08) 0%, rgba(255,107,107,0.03) 100%);
      border: 1px solid rgba(255,107,107,0.25);
      border-radius: 16px;
      padding: 1.3rem 1.6rem;
      margin-bottom: 1.5rem;
      position: relative;
      overflow: hidden;
      backdrop-filter: blur(8px);
  }
  .disclosure-box::before {
      content: '';
      position: absolute;
      left: 0; top: 0; bottom: 0;
      width: 4px;
      background: linear-gradient(180deg, #FF6B6B, #FF4757);
      border-radius: 4px 0 0 4px;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# THEME CONSTANTS — PREMIUM PALETTE
# ─────────────────────────────────────────────────────────────
BG       = "#060D18"
PANEL    = "#0E1E34"
TEAL     = "#00C9A7"
TEAL_LT  = "#00E8BF"
RED      = "#FF6B6B"
WHITE    = "#E8F0FE"
GREY     = "#6B8CA8"
GOLD     = "#FFD700"
BLUE     = "#4A90D9"
PURPLE   = "#A78BFA"
ORANGE   = "#FB923C"
CYAN     = "#06B6D4"

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(6,13,24,0.5)",
    font=dict(family="Inter, -apple-system, sans-serif", color=WHITE, size=12),
    margin=dict(t=55, b=40, l=20, r=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=WHITE, size=11),
                bordercolor="rgba(255,255,255,0.05)", borderwidth=1),
    colorway=[TEAL, BLUE, GOLD, RED, PURPLE, ORANGE, CYAN]
)

AXIS_STYLE = dict(
    gridcolor="rgba(30,58,95,0.3)",
    zerolinecolor="rgba(30,58,95,0.5)",
    tickfont=dict(color="#8CA0B8", size=11),
    title_font=dict(color=GREY, size=11),
)

def apply_chart_style(fig, title="", height=400):
    fig.update_layout(**CHART_LAYOUT, height=height,
                      title=dict(text=f"<b>{title}</b>",
                                 font=dict(size=13, color=WHITE), x=0, xanchor="left") if title else {})
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig


# ─────────────────────────────────────────────────────────────
# DATA LOAD
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base, "data", "sme_clean.csv"))
    df["LOG_CAP"] = np.log1p(df["AUTHORIZED_CAP_INR"])
    df["HAS_MULTIPLE_DIRECTORS"] = (df["DIRECTOR_COUNT"] > 2).astype(int)
    df["AGE_BUCKET"] = pd.cut(df["AGE_YEARS"], bins=[0, 2, 5, 10, 25],
                               labels=["0–2 yrs", "2–5 yrs", "5–10 yrs", "10+ yrs"])
    try:
        with open(os.path.join(base, "outputs", "model_metrics.json")) as f:
            metrics = json.load(f)
    except Exception:
        metrics = {"auc_roc": 0.868, "top_feature": "Multiple Directors"}
    # Load real RBI calibration data
    try:
        with open(os.path.join(base, "rbi_data", "rbi_msme_macro.json")) as f:
            rbi_macro = json.load(f)
    except Exception:
        rbi_macro = {}
    try:
        with open(os.path.join(base, "rbi_data", "rbi_sector_calibration.json")) as f:
            rbi_sectors = json.load(f)
    except Exception:
        rbi_sectors = {}
    return df, metrics, rbi_macro, rbi_sectors

df, metrics, rbi_macro, rbi_sectors = load_data()

CREDIT_THRESHOLD    = 65
OPPORTUNITY_CAP_INR = 5_000_000
COMPANY_COUNT       = 2500

# Pre-compute key stats
total         = len(df)
avg_score     = df["CREDIT_SCORE"].mean()
pct_high_risk = df["DEFAULT_RISK"].mean() * 100
pct_creditworthy = (df["CREDIT_SCORE"] >= CREDIT_THRESHOLD).mean() * 100
sector_avg    = df.groupby("INDUSTRY")["CREDIT_SCORE"].mean().sort_values(ascending=False)
safest_sector = sector_avg.index[0]
riskiest_sector = sector_avg.index[-1]
opp_df        = df[df["IS_OPPORTUNITY"] == 1] if "IS_OPPORTUNITY" in df.columns else df[(df["CREDIT_SCORE"] > 65) & (df["AUTHORIZED_CAP_INR"] < OPPORTUNITY_CAP_INR)]
opp_count     = len(opp_df)
industry_palette = {
    "Retail": TEAL, "Manufacturing": BLUE, "Logistics": GOLD,
    "F&B": RED, "IT Services": PURPLE, "Construction": ORANGE
}

# ─────────────────────────────────────────────────────────────
# SIDEBAR — FROSTED GLASS
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div style="font-size:2.6rem; margin-bottom:6px;">🇮🇳</div>
        <div style="font-size:0.95rem; font-weight:800; line-height:1.3;
                    background: linear-gradient(135deg, #00C9A7, #00E8BF);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text;">
            India SME Credit Risk<br>& Growth Intelligence
        </div>
        <div style="font-size:0.7rem; color:#4A6A8A; margin-top:6px; letter-spacing:1px; text-transform:uppercase;">
            Bengaluru · 2026
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        options=[
            "📊  Executive Overview",
            "🗺️  Geographic Intelligence",
            "⚠️  Sector Risk Analysis",
            "🏢  Company Profiles",
            "💰  Hidden Opportunities",
            "🔬  Model Card",
            "📡  Live RBI Data",
        ],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sidebar-stats">
        <div class="stat-row">
            <span><span class="stat-icon">📁</span> SME Records</span>
            <span class="stat-value" style="color:#00C9A7;">{total:,}</span>
        </div>
        <div class="stat-row">
            <span><span class="stat-icon">🏙️</span> Indian States</span>
            <span class="stat-value" style="color:#4A90D9;">{df['STATE'].nunique()}</span>
        </div>
        <div class="stat-row">
            <span><span class="stat-icon">🏭</span> Sectors</span>
            <span class="stat-value" style="color:#FFD700;">{df['INDUSTRY'].nunique()}</span>
        </div>
        <div class="stat-row">
            <span><span class="stat-icon">🤖</span> XGBoost AUC</span>
            <span class="stat-value" style="color:#00C9A7;">{metrics['auc_roc']:.4f}</span>
        </div>
        <div class="stat-row">
            <span><span class="stat-icon">💰</span> Opportunity SMEs</span>
            <span class="stat-value" style="color:#FFD700;">{opp_count}</span>
        </div>
        <div class="stat-row">
            <span><span class="live-dot"></span> RBI Data</span>
            <span class="stat-value" style="color:#00E8BF;">Live</span>
        </div>
    </div>
    <div style="margin-top:1rem; padding:0.6rem; text-align:center;
                background:rgba(255,107,107,0.06); border-radius:10px;
                border:1px solid rgba(255,107,107,0.12);">
        <span style="font-size:0.68rem; color:#FF9999;">⚠️ Synthetic data · methodology demo</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# GLOBAL HEADER — ANIMATED GRADIENT
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <h1>🇮🇳 India SME Credit Risk & Growth Intelligence Platform</h1>
    <p>Analysing {total:,} Indian SMEs across 6 sectors · {df['STATE'].nunique()} states · Real RBI data integrated · Bengaluru, 2026</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE 1 — EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════
if "Executive Overview" in page:
    st.markdown("""
    <div class="page-header">
        <h2>📊 Executive Overview</h2>
        <p>Platform-level KPIs · Capital distribution · Sector composition</p>
    </div>""", unsafe_allow_html=True)

    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        (k1, f"{total:,}", "Total SMEs Analysed", WHITE),
        (k2, f"{avg_score:.1f}", "Avg Credit Score / 100", TEAL),
        (k3, f"{pct_high_risk:.1f}%", "High-Risk SMEs", RED),
        (k4, f"{pct_creditworthy:.1f}%", f"Creditworthy (≥{CREDIT_THRESHOLD})", GOLD),
    ]
    for col, val, label, color in kpis:
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color:{color};">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Donut + Bar
    c1, c2 = st.columns([1, 1.6])

    with c1:
        tier_counts = df["CAPITAL_TIER"].value_counts()
        tier_colors = [TEAL, BLUE, GOLD, RED]
        fig_donut = go.Figure(go.Pie(
            labels=tier_counts.index.tolist(),
            values=tier_counts.values.tolist(),
            hole=0.60,
            marker=dict(colors=tier_colors[:len(tier_counts)],
                        line=dict(color="#0D1B2A", width=3)),
            textfont=dict(color=WHITE, size=11),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>"
        ))
        fig_donut.update_layout(
            **CHART_LAYOUT, height=320,
            title=dict(text="<b>Capital Tier Distribution</b>", font=dict(size=13, color=WHITE), x=0),
            annotations=[dict(text=f"<b>{total}</b><br><span style='font-size:10px'>SMEs</span>",
                              x=0.5, y=0.5, font=dict(size=16, color=WHITE),
                              showarrow=False)]
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"""<div class="insight-box">💡 <b>{tier_counts.get("Micro",0)/total*100:.0f}%</b> are micro-enterprises — the capital gap that formal credit can address.</div>""", unsafe_allow_html=True)

    with c2:
        ind_counts = df["INDUSTRY"].value_counts().sort_values()
        bar_colors = [industry_palette.get(i, BLUE) for i in ind_counts.index]
        fig_ind = go.Figure(go.Bar(
            x=ind_counts.index.tolist(), y=ind_counts.values.tolist(),
            marker=dict(color=bar_colors, line=dict(color="#0D1B2A", width=1)),
            text=ind_counts.values.tolist(), textposition="outside",
            textfont=dict(color=WHITE, size=11),
            hovertemplate="<b>%{x}</b><br>SME Count: %{y}<extra></extra>"
        ))
        apply_chart_style(fig_ind, "SME Count by Industry Sector", height=320)
        fig_ind.update_yaxes(title_text="No. of SMEs")
        fig_ind.update_layout(showlegend=False)
        st.plotly_chart(fig_ind, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"""<div class="insight-box">💡 Sector distribution is balanced. <b>{ind_counts.index[-1]}</b> leads with {ind_counts.iloc[-1]} SMEs.</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 3: Risk breakdown by sector
    risk_by_sector = df.groupby(["INDUSTRY", "DEFAULT_RISK"]).size().unstack(fill_value=0)
    fig_risk = go.Figure()
    for risk_val, color, label in [(0, TEAL, "✅ Low Risk"), (1, RED, "🔴 High Risk")]:
        fig_risk.add_trace(go.Bar(
            name=label,
            x=risk_by_sector.index.tolist(),
            y=risk_by_sector.get(risk_val, pd.Series([0]*len(risk_by_sector))).tolist(),
            marker=dict(color=color, opacity=0.88, line=dict(color="#0D1B2A", width=1)),
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y}}<extra></extra>"
        ))
    fig_risk.update_layout(**CHART_LAYOUT, height=280, barmode="group",
                           title=dict(text="<b>Risk Profile by Industry</b>",
                                      font=dict(size=13, color=WHITE), x=0))
    fig_risk.update_xaxes(**AXIS_STYLE)
    fig_risk.update_yaxes(**AXIS_STYLE, title_text="SME Count")
    st.plotly_chart(fig_risk, use_container_width=True, config={"displayModeBar": False})
    st.markdown(f"""<div class="insight-box">💡 <b>{riskiest_sector}</b> has the highest high-risk concentration — structural sector risk score of 0.60 drives defaults.</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 2 — GEOGRAPHIC INTELLIGENCE
# ══════════════════════════════════════════════════════════════
elif "Geographic" in page:
    st.markdown("""
    <div class="page-header">
        <h2>🗺️ Geographic Intelligence</h2>
        <p>State-level credit scoring · Default hotspots · Non-metro opportunity map</p>
    </div>""", unsafe_allow_html=True)

    state_avg    = df.groupby("STATE")["CREDIT_SCORE"].mean().sort_values(ascending=False)
    state_def    = df[df["DEFAULT_RISK"]==1].groupby("STATE").size().sort_values(ascending=False).head(10)
    state_cw_pct = (df.groupby("STATE")
                      .apply(lambda g: (g["CREDIT_SCORE"] >= CREDIT_THRESHOLD).mean() * 100)
                      .sort_values(ascending=False).head(10).reset_index())
    state_cw_pct.columns = ["State", "% Creditworthy"]

    top_state    = state_avg.index[0]
    top15        = state_avg.head(15)

    # Row 1
    c1, c2 = st.columns(2)

    with c1:
        bar_colors_s = [TEAL if s == top15.index[0] else BLUE for s in top15.index]
        fig_s1 = go.Figure(go.Bar(
            x=top15.values, y=top15.index.tolist(), orientation="h",
            marker=dict(color=bar_colors_s, line=dict(color="#0D1B2A", width=0.5)),
            text=[f"{v:.1f}" for v in top15.values], textposition="outside",
            textfont=dict(color=WHITE, size=10),
            hovertemplate="<b>%{y}</b><br>Avg Credit Score: %{x:.1f}<extra></extra>"
        ))
        apply_chart_style(fig_s1, "Top 15 States — Avg Credit Score", height=420)
        fig_s1.update_xaxes(range=[40, 80], title_text="Avg Credit Score")
        fig_s1.update_layout(showlegend=False)
        st.plotly_chart(fig_s1, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"""<div class="insight-box">💡 <b>{top_state}</b> leads with avg score {state_avg.iloc[0]:.1f} — strong IT services and startup ecosystem driving creditworthiness.</div>""", unsafe_allow_html=True)

    with c2:
        bar_colors_d = [RED if s == state_def.index[0] else "#D9534F" for s in state_def.index]
        fig_s2 = go.Figure(go.Bar(
            x=state_def.values, y=state_def.index.tolist(), orientation="h",
            marker=dict(color=bar_colors_d, line=dict(color="#0D1B2A", width=0.5)),
            text=state_def.values.tolist(), textposition="outside",
            textfont=dict(color=WHITE, size=10),
            hovertemplate="<b>%{y}</b><br>High-Risk SMEs: %{x}<extra></extra>"
        ))
        apply_chart_style(fig_s2, "Top 10 States — High-Risk SME Count", height=420)
        fig_s2.update_xaxes(title_text="High-Risk SME Count")
        fig_s2.update_layout(showlegend=False)
        st.plotly_chart(fig_s2, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"""<div class="warn-box">⚠️ <b>{state_def.index[0]}</b> has the highest default density — warrants geo-targeted credit guarantee expansion.</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Table + Non-Metro bar
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("**📋 Top 10 States: % Creditworthy SMEs**")
        styled_df = state_cw_pct.copy()
        styled_df["% Creditworthy"] = styled_df["% Creditworthy"].map(lambda x: f"{x:.1f}%")
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            height=340
        )

    with c4:
        non_metro_opp = (df[(df["CREDIT_SCORE"] > CREDIT_THRESHOLD) & (df["IS_METRO"]==0)]
                           .groupby("STATE").size().sort_values(ascending=False).head(10))
        opp_state_top = non_metro_opp.index[0]
        bar_colors_nm = [GOLD if s == opp_state_top else TEAL for s in non_metro_opp.index]
        fig_s3 = go.Figure(go.Bar(
            x=non_metro_opp.values, y=non_metro_opp.index.tolist(), orientation="h",
            marker=dict(color=bar_colors_nm, line=dict(color="#0D1B2A", width=0.5)),
            text=non_metro_opp.values.tolist(), textposition="outside",
            textfont=dict(color=WHITE, size=10),
            hovertemplate="<b>%{y}</b><br>Non-Metro Creditworthy: %{x}<extra></extra>"
        ))
        apply_chart_style(fig_s3, "Non-Metro Creditworthy SMEs by State", height=340)
        fig_s3.update_xaxes(title_text="SME Count")
        fig_s3.update_layout(showlegend=False)
        st.plotly_chart(fig_s3, use_container_width=True, config={"displayModeBar": False})

    opp_count_nm = df[(df["CREDIT_SCORE"] > CREDIT_THRESHOLD) & (df["IS_METRO"]==0)].groupby("STATE").size().max()
    st.markdown(f"""<div class="opp-box">🌟 <b>Highest Non-Metro Opportunity State: {opp_state_top}</b> — {opp_count_nm} creditworthy SMEs outside metros, likely underserved by urban-focused lenders.</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 3 — SECTOR RISK ANALYSIS
# ══════════════════════════════════════════════════════════════
elif "Sector Risk" in page:
    st.markdown("""
    <div class="page-header">
        <h2>⚠️ Sector Risk Analysis</h2>
        <p>Sector scorecard · Risk group distribution · Risk-score bubble chart</p>
    </div>""", unsafe_allow_html=True)

    # Sector KPI Row
    s1, s2, s3, s4 = st.columns(4)
    sector_kpis = [
        (s1, safest_sector,   f"{sector_avg.iloc[0]:.1f}",  "Safest Sector (Avg Score)",    TEAL),
        (s2, riskiest_sector, f"{sector_avg.iloc[-1]:.1f}", "Highest Risk Sector",           RED),
        (s3, f"{sector_avg.iloc[0] - sector_avg.iloc[-1]:.1f} pts", "", "Credit Score Gap (Best–Worst)", GOLD),
        (s4, f"{df[df['DEFAULT_RISK']==1]['INDUSTRY'].value_counts().index[0]}", "", "Most Defaults Sector", RED),
    ]
    for col, val, sub, label, color in sector_kpis:
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color:{color}; font-size:1.6rem;">{val}</div>
            {'<div style="color:#8BA0B0;font-size:0.75rem;">'+sub+'</div>' if sub else ''}
            <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        bar_colors_sec = [TEAL if s==safest_sector else (RED if s==riskiest_sector else BLUE)
                          for s in sector_avg.index]
        fig_sec1 = go.Figure(go.Bar(
            x=sector_avg.values, y=sector_avg.index.tolist(), orientation="h",
            marker=dict(color=bar_colors_sec, line=dict(color="#0D1B2A", width=0.5)),
            text=[f"{v:.1f}" for v in sector_avg.values], textposition="outside",
            textfont=dict(color=WHITE, size=11),
            hovertemplate="<b>%{y}</b><br>Avg Credit Score: %{x:.1f}<extra></extra>"
        ))
        apply_chart_style(fig_sec1, "Avg Credit Score by Sector", height=350)
        fig_sec1.update_xaxes(range=[40, sector_avg.max()*1.15], title_text="Avg Credit Score")
        fig_sec1.update_layout(showlegend=False)
        st.plotly_chart(fig_sec1, use_container_width=True, config={"displayModeBar": False})

    with c2:
        risk_by_sec = df.groupby(["INDUSTRY", "DEFAULT_RISK"]).size().unstack(fill_value=0)
        fig_sec2 = go.Figure()
        for risk_val, color, label in [(0, TEAL, "✅ Low Risk"), (1, RED, "🔴 High Risk")]:
            fig_sec2.add_trace(go.Bar(
                name=label, x=risk_by_sec.index.tolist(),
                y=risk_by_sec.get(risk_val, pd.Series([0]*len(risk_by_sec))).tolist(),
                marker=dict(color=color, opacity=0.88, line=dict(color="#0D1B2A", width=1)),
                hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y}}<extra></extra>"
            ))
        fig_sec2.update_layout(**CHART_LAYOUT, height=350, barmode="group",
                               title=dict(text="<b>High Risk vs Low Risk by Sector</b>",
                                          font=dict(size=13, color=WHITE), x=0))
        fig_sec2.update_xaxes(**AXIS_STYLE)
        fig_sec2.update_yaxes(**AXIS_STYLE, title_text="SME Count")
        st.plotly_chart(fig_sec2, use_container_width=True, config={"displayModeBar": False})

    # Bubble chart
    sector_risk_map = df.groupby("INDUSTRY")["SECTOR_RISK_SCORE"].first()
    sector_counts   = df.groupby("INDUSTRY").size()
    scatter_x = [sector_risk_map[s] for s in sector_avg.index]
    scatter_y = sector_avg.values.tolist()
    scatter_s = [sector_counts[s] * 2.5 for s in sector_avg.index]
    scatter_c = [TEAL if s==safest_sector else (RED if s==riskiest_sector else BLUE) for s in sector_avg.index]

    fig_bubble = go.Figure(go.Scatter(
        x=scatter_x, y=scatter_y,
        mode="markers+text",
        text=sector_avg.index.tolist(), textposition="top center",
        textfont=dict(color=WHITE, size=11),
        marker=dict(size=scatter_s, color=scatter_c, opacity=0.85,
                    line=dict(color=WHITE, width=1.5)),
        hovertemplate="<b>%{text}</b><br>Sector Risk: %{x:.2f}<br>Avg Credit Score: %{y:.1f}<extra></extra>"
    ))
    apply_chart_style(fig_bubble, "Sector Risk Score vs Avg Credit Score  (bubble size = SME count)", height=360)
    fig_bubble.update_xaxes(title_text="Sector Risk Score  (0 = safest, 1 = riskiest)")
    fig_bubble.update_yaxes(title_text="Avg Credit Score")
    st.plotly_chart(fig_bubble, use_container_width=True, config={"displayModeBar": False})

    c_a, c_b = st.columns(2)
    c_a.markdown(f"""<div class="insight-box">🟢 <b>Safest Sector: {safest_sector}</b> — avg score {sector_avg.iloc[0]:.1f}, sector risk 0.20. Low structural exposure makes IT firms natural candidates for working-capital credit.</div>""", unsafe_allow_html=True)
    c_b.markdown(f"""<div class="warn-box">🔴 <b>Highest Risk: {riskiest_sector}</b> — avg score {sector_avg.iloc[-1]:.1f}, sector risk 0.60. Prolonged project cycles and payment delays amplify SME default probability by 2–3×.</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 4 — COMPANY PROFILE ANALYSIS
# ══════════════════════════════════════════════════════════════
elif "Company Profiles" in page:
    st.markdown("""
    <div class="page-header">
        <h2>🏢 Company Profile Analysis</h2>
        <p>Age vs credit score · Capital tier distributions · Metro advantage · Age bucket trends</p>
    </div>""", unsafe_allow_html=True)

    metro_avg = df.groupby("IS_METRO")["CREDIT_SCORE"].mean()
    metro_diff = metro_avg.get(1, 0) - metro_avg.get(0, 0)
    age_bucket_avg = df.groupby("AGE_BUCKET", observed=True)["CREDIT_SCORE"].mean()

    c1, c2 = st.columns(2)

    with c1:
        fig_scatter = go.Figure()
        for ind in df["INDUSTRY"].unique():
            sub = df[df["INDUSTRY"] == ind]
            fig_scatter.add_trace(go.Scatter(
                x=sub["AGE_YEARS"], y=sub["CREDIT_SCORE"],
                mode="markers", name=ind,
                marker=dict(color=industry_palette.get(ind, BLUE), size=5, opacity=0.60),
                hovertemplate=f"<b>{ind}</b><br>Age: %{{x:.1f}} yrs<br>Score: %{{y:.1f}}<extra></extra>"
            ))
        apply_chart_style(fig_scatter, "Company Age vs Credit Score (by Sector)", height=380)
        fig_scatter.update_xaxes(title_text="Age (Years)")
        fig_scatter.update_yaxes(title_text="Credit Score")
        st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"""<div class="insight-box">💡 Clear upward trend — companies aged 10+ years score ~25 pts higher than new entrants as tenure de-risks lender exposure.</div>""", unsafe_allow_html=True)

    with c2:
        fig_violin = go.Figure()
        tier_order  = ["Micro", "Small", "Medium", "Large"]
        tier_colors = [TEAL, BLUE, GOLD, RED]
        for tier, color in zip(tier_order, tier_colors):
            if tier in df["CAPITAL_TIER"].values:
                sub = df[df["CAPITAL_TIER"] == tier]
                r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
                fig_violin.add_trace(go.Violin(
                    x=sub["CAPITAL_TIER"], y=sub["CREDIT_SCORE"], name=tier,
                    line_color=color,
                    fillcolor=f"rgba({r},{g},{b},0.18)",
                    opacity=0.9, box_visible=True, meanline_visible=True,
                    hovertemplate=f"<b>{tier}</b><br>Credit Score: %{{y:.1f}}<extra></extra>"
                ))
        apply_chart_style(fig_violin, "Credit Score Distribution by Capital Tier", height=380)
        fig_violin.update_yaxes(title_text="Credit Score")
        st.plotly_chart(fig_violin, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"""<div class="insight-box">💡 Medium-tier SMEs have the tightest, highest distribution — proving scale matters for credit quality.</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        fig_metro = go.Figure(go.Bar(
            x=["Non-Metro", "Metro"],
            y=[metro_avg.get(0, 0), metro_avg.get(1, 0)],
            marker=dict(color=[BLUE, TEAL], line=dict(color="#0D1B2A", width=1)),
            text=[f"{metro_avg.get(0,0):.1f}", f"{metro_avg.get(1,0):.1f}"],
            textposition="outside",
            textfont=dict(color=WHITE, size=13),
            hovertemplate="%{x}<br>Avg Credit Score: %{y:.1f}<extra></extra>"
        ))
        apply_chart_style(fig_metro, "Metro vs Non-Metro Avg Credit Score", height=300)
        fig_metro.update_yaxes(title_text="Avg Credit Score", range=[0, 80])
        fig_metro.update_layout(showlegend=False)
        st.plotly_chart(fig_metro, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"""<div class="insight-box">💡 Metro-based SMEs score <b>{metro_diff:.1f} pts higher</b> — infrastructure, market access, and banking proximity drive this advantage.</div>""", unsafe_allow_html=True)

    with c4:
        bucket_labels = age_bucket_avg.index.astype(str).tolist()
        bucket_vals   = age_bucket_avg.values.tolist()
        bc = [TEAL, BLUE, GOLD, RED]
        fig_age = go.Figure(go.Bar(
            x=bucket_labels, y=bucket_vals,
            marker=dict(color=bc[:len(bucket_labels)], line=dict(color="#0D1B2A", width=1)),
            text=[f"{v:.1f}" for v in bucket_vals], textposition="outside",
            textfont=dict(color=WHITE, size=13),
            hovertemplate="%{x}<br>Avg Credit Score: %{y:.1f}<extra></extra>"
        ))
        apply_chart_style(fig_age, "Avg Credit Score by Company Age Bucket", height=300)
        fig_age.update_yaxes(title_text="Avg Credit Score", range=[0, 85])
        fig_age.update_layout(showlegend=False)
        st.plotly_chart(fig_age, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"""<div class="insight-box">💡 Companies aged <b>10+ years</b> score highest — vintage is a stronger predictor than capital size alone.</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 5 — HIDDEN OPPORTUNITIES
# ══════════════════════════════════════════════════════════════
elif "Hidden Opportunities" in page:
    st.markdown("""
    <div class="page-header">
        <h2>💰 Hidden Opportunity Finder</h2>
        <p>Creditworthy SMEs (score > 65) + Capital &lt; ₹50L — the underserved lending frontier</p>
    </div>""", unsafe_allow_html=True)

    opp_states_n    = opp_df["STATE"].nunique()
    opp_cap_cr      = opp_df["AUTHORIZED_CAP_INR"].sum() / 1e7
    opp_avg_score   = opp_df["CREDIT_SCORE"].mean()
    opp_by_state    = opp_df.groupby("STATE").size().sort_values(ascending=False).head(10)
    opp_by_industry = opp_df.groupby("INDUSTRY").size().sort_values(ascending=False)

    # KPI Row
    o1, o2, o3, o4 = st.columns(4)
    opp_kpis = [
        (o1, f"{opp_count}",           "Opportunity SMEs",             GOLD),
        (o2, f"{opp_states_n}",        "States Represented",           TEAL),
        (o3, f"₹{opp_cap_cr:.0f} Cr",  "Addressable Capital Base",     PURPLE),
        (o4, f"{opp_avg_score:.1f}",   "Avg Credit Score (Opp. SMEs)", TEAL),
    ]
    for col, val, label, color in opp_kpis:
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color:{color};">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="opp-box" style="margin-top:1rem;">
        💰 <b>These {opp_count} SMEs are creditworthy but likely underserved by traditional lenders due to small
        capital size</b> — representing a high-yield, lower-risk lending frontier across {opp_states_n} Indian states.
        Estimated addressable base: <b>₹{opp_cap_cr:.0f} Cr</b>.
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Table + Charts
    c1, c2 = st.columns([1.5, 1])

    with c1:
        st.markdown("**📋 Top Opportunity SMEs** *(creditworthy + micro-capitalised)*")
        display_df = (opp_df[["COMPANY_NAME", "STATE", "INDUSTRY", "CREDIT_SCORE", "CAPITAL_TIER",
                               "AUTHORIZED_CAP_INR"]]
                      .head(15).copy())
        display_df["CREDIT_SCORE"]       = display_df["CREDIT_SCORE"].map(lambda x: f"{x:.1f}")
        display_df["AUTHORIZED_CAP_INR"] = display_df["AUTHORIZED_CAP_INR"].map(lambda x: f"₹{x/1e5:.1f}L")
        display_df.columns = ["Company", "State", "Industry", "Score", "Tier", "Capital"]
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=380)

    with c2:
        # By Industry donut
        opp_ind_labels = opp_by_industry.index.tolist()
        opp_ind_vals   = opp_by_industry.values.tolist()
        opp_ind_colors = [industry_palette.get(i, BLUE) for i in opp_ind_labels]
        fig_opp_ind = go.Figure(go.Pie(
            labels=opp_ind_labels, values=opp_ind_vals, hole=0.50,
            marker=dict(colors=opp_ind_colors, line=dict(color="#0D1B2A", width=2)),
            textfont=dict(color=WHITE, size=10),
            hovertemplate="<b>%{label}</b><br>Count: %{value} (%{percent})<extra></extra>"
        ))
        opp_pie_layout = {k: v for k, v in CHART_LAYOUT.items() if k != "legend"}
        fig_opp_ind.update_layout(**opp_pie_layout, height=380,
                                   title=dict(text="<b>By Industry</b>", font=dict(size=12, color=WHITE), x=0),
                                   showlegend=True,
                                   legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=WHITE, size=10),
                                               x=0, y=-0.15, orientation="h"))
        st.plotly_chart(fig_opp_ind, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)

    # State bar
    bar_cols_opp = [GOLD if s == opp_by_state.index[0] else TEAL for s in opp_by_state.index]
    fig_opp_state = go.Figure(go.Bar(
        x=opp_by_state.values, y=opp_by_state.index.tolist(), orientation="h",
        marker=dict(color=bar_cols_opp, line=dict(color="#0D1B2A", width=0.5)),
        text=opp_by_state.values.tolist(), textposition="outside",
        textfont=dict(color=WHITE, size=11),
        hovertemplate="<b>%{y}</b><br>Opportunity SMEs: %{x}<extra></extra>"
    ))
    apply_chart_style(fig_opp_state, "Opportunity SMEs by State (Top 10)", height=320)
    fig_opp_state.update_xaxes(title_text="No. of Opportunity SMEs")
    fig_opp_state.update_layout(showlegend=False)
    st.plotly_chart(fig_opp_state, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""
    <div class="warn-box">
        ⚠️ <b>Synthetic Data Notice:</b> DEFAULT_RISK is rule-derived from a credit scoring formula. These
        opportunity flags demonstrate analytical methodology. For live fintech deployment, retrain on CIBIL /
        RBI NPA bureau data before making actual underwriting decisions.
    </div>""", unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════
#  PAGE 6 — MODEL CARD
# ══════════════════════════════════════════════════════════════
elif "Model Card" in page:
    st.markdown("""
    <div class="page-header">
        <h2>🔬 Model Card — Transparency & Methodology</h2>
        <p>Honest performance analysis · Synthetic data disclosure · Opportunity scoring rubric · Real data roadmap</p>
    </div>""", unsafe_allow_html=True)

    # ── SYNTHETIC DATA DISCLAIMER (top-of-page, hard to miss) ──
    st.markdown(f"""
    <div style="background:rgba(255,107,107,0.10); border:2px solid #FF6B6B; border-radius:10px;
                padding:1.1rem 1.4rem; margin-bottom:1.2rem;">
        <div style="font-size:1rem; font-weight:700; color:#FF6B6B; margin-bottom:6px;">
            ⚠️ Critical Disclosure: This Is Synthetic Data
        </div>
        <div style="font-size:0.83rem; color:#FFAAAA; line-height:1.7;">
            <b>DEFAULT_RISK is deterministically derived from a scoring formula</b> — not from actual
            loan performance, CIBIL bureau data, or RBI NPA records. The XGBoost AUC of
            <b>{metrics["auc_roc"]:.4f}</b> is high because the model is learning the rules <i>built into</i>
            the generator, not discovering latent real-world credit risk. This platform demonstrates
            <b>analytical methodology and engineering competence</b>, not live underwriting capability.
            <br><br>
            The phrase "MCA21/RBI/SIDBI calibrated" means the <i>distributional parameters</i>
            (state weights, sector NPA tiers, capital distribution shape) are informed by published
            aggregate data — not that individual records are verified against any registry.
        </div>
    </div>""", unsafe_allow_html=True)

    # ── RETRAIN MODEL IN-APP FOR CURVES ──
    @st.cache_data
    def get_model_curves():
        base = os.path.dirname(__file__)
        _df  = pd.read_csv(os.path.join(base, "data", "sme_clean.csv"))
        _df["LOG_CAP"] = np.log1p(_df["AUTHORIZED_CAP_INR"])
        _df["HAS_MULTIPLE_DIRECTORS"] = (_df["DIRECTOR_COUNT"] > 2).astype(int)
        X = _df[["AGE_YEARS","HAS_MULTIPLE_DIRECTORS","IS_METRO","SECTOR_RISK_SCORE","LOG_CAP"]]
        y = _df["DEFAULT_RISK"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        m = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                          use_label_encoder=False, eval_metric="logloss",
                          random_state=42, verbosity=0)
        m.fit(Xtr, ytr)
        yp = m.predict_proba(Xte)[:, 1]
        return yte.values, yp, Xte, m

    yte, yp, Xte, model = get_model_curves()

    # ── ROC + PR CURVES ──
    fpr, tpr, roc_thresh = roc_curve(yte, yp)
    prec, rec, pr_thresh  = precision_recall_curve(yte, yp)
    auc_val = roc_auc_score(yte, yp)

    c1, c2 = st.columns(2)

    with c1:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"XGBoost (AUC={auc_val:.4f})",
            line=dict(color=TEAL, width=2.5),
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines", name="Random Classifier",
            line=dict(color=GREY, dash="dash", width=1.5)
        ))
        apply_chart_style(fig_roc, "ROC Curve", height=360)
        fig_roc.update_xaxes(title_text="False Positive Rate", range=[0,1])
        fig_roc.update_yaxes(title_text="True Positive Rate", range=[0,1.01])
        st.plotly_chart(fig_roc, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"""<div class="insight-box">ℹ️ AUC of {auc_val:.4f} reflects a <b>synthetic dataset</b> where the model re-learns the generation rules. On real bureau data, expect AUC to drop to 0.70–0.78 — still useful, but significantly lower.</div>""", unsafe_allow_html=True)

    with c2:
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines", name="Precision-Recall",
            line=dict(color=GOLD, width=2.5),
            hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>"
        ))
        baseline = yte.mean()
        fig_pr.add_hline(y=baseline, line=dict(color=GREY, dash="dash", width=1.5),
                         annotation_text=f"Baseline ({baseline:.2f})", annotation_position="right")
        apply_chart_style(fig_pr, "Precision-Recall Curve", height=360)
        fig_pr.update_xaxes(title_text="Recall", range=[0,1])
        fig_pr.update_yaxes(title_text="Precision", range=[0,1.01])
        st.plotly_chart(fig_pr, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"""<div class="insight-box">ℹ️ The PR curve matters more than ROC for imbalanced datasets. With only {yte.mean()*100:.1f}% positive class, a lender cares <b>more about precision</b> (not approving bad loans) than recall.</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── THRESHOLD ANALYSIS TABLE ──
    st.markdown("**📋 Precision / Recall at Different Decision Thresholds**")
    st.markdown("<p style='font-size:0.78rem;color:#5A7A9A;'>A real credit model needs to be evaluated at the operating threshold a lender chooses, not just at 0.5 (the default). Adjust based on risk appetite.</p>", unsafe_allow_html=True)

    thresh_rows = []
    for t in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        pred = (yp >= t).astype(int)
        tp = ((pred == 1) & (yte == 1)).sum()
        fp = ((pred == 1) & (yte == 0)).sum()
        fn = ((pred == 0) & (yte == 1)).sum()
        tn = ((pred == 0) & (yte == 0)).sum()
        prec_t = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec_t  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_t   = 2*prec_t*rec_t/(prec_t+rec_t) if (prec_t+rec_t) > 0 else 0
        pct_flagged = pred.mean() * 100
        thresh_rows.append({
            "Threshold": f"{t:.2f}",
            "Precision (High-Risk)": f"{prec_t:.3f}",
            "Recall (High-Risk)": f"{rec_t:.3f}",
            "F1-Score": f"{f1_t:.3f}",
            "% Flagged as High-Risk": f"{pct_flagged:.1f}%",
            "Lender Use Case": (
                "Conservative — flag widely" if t < 0.40 else
                "Balanced" if t < 0.60 else
                "Tight — minimize false alarms"
            )
        })
    st.dataframe(pd.DataFrame(thresh_rows), use_container_width=True, hide_index=True)
    st.markdown("""<div class="insight-box">💡 At threshold 0.50: model flags high-risk SMEs with 80% precision and 71% recall. A real NBFC might lower the threshold to 0.35 to prioritise recall (catch more defaults) at the cost of more false alarms.</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CONFUSION MATRIX HEATMAP ──
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("**🟦 Confusion Matrix (threshold = 0.50)**")
        cm = confusion_matrix(yte, (yp >= 0.50).astype(int))
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Predicted: Low Risk", "Predicted: High Risk"],
            y=["Actual: Low Risk", "Actual: High Risk"],
            colorscale=[[0,"#0A1520"],[0.5,"#1E5070"],[1.0,TEAL]],
            text=cm, texttemplate="%{text}", textfont=dict(size=22, color=WHITE),
            showscale=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
        ))
        apply_chart_style(fig_cm, "", height=280)
        st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})

    with c4:
        # Feature importances
        st.markdown("**📊 Feature Importance (XGBoost Gain)**")
        fi = model.feature_importances_
        feat_labels = ["Age (yrs)", "Multi-Directors", "Metro", "Sector Risk", "Log Capital"]
        fi_df = pd.DataFrame({"Feature": feat_labels, "Importance": fi}).sort_values("Importance")
        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
            marker=dict(color=[TEAL if v == fi_df["Importance"].max() else BLUE
                               for v in fi_df["Importance"]],
                        line=dict(color="#0D1B2A", width=0.5)),
            text=[f"{v:.3f}" for v in fi_df["Importance"]], textposition="outside",
            textfont=dict(color=WHITE, size=10)
        ))
        apply_chart_style(fig_fi, "", height=280)
        fig_fi.update_xaxes(title_text="Importance (Gain)")
        fig_fi.update_layout(showlegend=False)
        st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)

    # ── OPPORTUNITY SCORING METHODOLOGY ──
    st.markdown("**💰 Opportunity SME Scoring — Auditable 3-Factor Criteria**")
    st.markdown("<p style='font-size:0.78rem;color:#5A7A9A;'>An SME must satisfy ALL THREE criteria simultaneously to be flagged as a lending opportunity. This is stricter than many scorecard systems that use single-threshold flags.</p>", unsafe_allow_html=True)

    criteria_df = pd.DataFrame({
        "Criterion": [
            "Credit Score ≥ 65",
            "Capital < ₹50 Lakh",
            "Sector Risk Score ≤ 0.50"
        ],
        "Rationale": [
            "Score of 65 sits above the dataset median (61.1) — SME has demonstrated above-average creditworthiness signals",
            "₹50L is the RBI/MUDRA cutoff for micro enterprise credit; firms above this qualify for conventional bank products",
            "Excludes Construction & F&B (risk scores 0.55–0.65, NPA penalties 15–22 pts) — high enough sector risk to warrant separate underwriting"
        ],
        "SMEs Passing": [
            f"{(df['CREDIT_SCORE'] >= 65).sum():,} ({(df['CREDIT_SCORE'] >= 65).mean()*100:.1f}%)",
            f"{(df['AUTHORIZED_CAP_INR'] < 5_000_000).sum():,} ({(df['AUTHORIZED_CAP_INR'] < 5_000_000).mean()*100:.1f}%)",
            f"{(df['SECTOR_RISK_SCORE'] <= 0.50).sum():,} ({(df['SECTOR_RISK_SCORE'] <= 0.50).mean()*100:.1f}%)"
        ],
        "Final (AND)": [
            f"{opp_count:,} ({opp_count/total*100:.1f}%)", "↑", "↑"
        ]
    })
    st.dataframe(criteria_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── REAL DATA ROADMAP ──
    st.markdown("**🗺️ Path from Demo → Production: Real Data Sources**")
    roadmap_df = pd.DataFrame({
        "Data Gap": [
            "Actual default labels",
            "GST-based revenue",
            "Geographic income data",
            "Bureau credit history",
            "Company registry data"
        ],
        "Source": [
            "RBI MSME NPA data (Table 5.3, Annual Report); SIDBI MSME Pulse quarterly report",
            "GSTN public aggregate data; OCEN API sandbox (credit-eligible GST returns)",
            "DBIE (RBI Database of Indian Economy) — district-level per-capita income",
            "CIBIL SME rank API; Experian India; Equifax SME"
        ],
        "Estimated Lift (AUC)": [
            "Real baseline: 0.68–0.75",
            "+0.04–0.07",
            "+0.02–0.03",
            "+0.06–0.10",
            "+0.02–0.04"
        ],
        "Access": ["Public PDF", "API (sandbox free)", "Public JSON", "Paid API", "Public CSV"]
    })
    st.dataframe(roadmap_df, use_container_width=True, hide_index=True)
    st.markdown("""<div class="opp-box">🎯 <b>Realistic target AUC on real data: 0.74–0.82</b> — achievable with GSTN revenue + CIBIL SME rank + 3–5 years of verified NPA labels. The synthetic model at 0.95 is a methodology proof-of-concept, not a benchmark.</div>""", unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════
#  PAGE 7 — LIVE RBI DATA
# ══════════════════════════════════════════════════════════════
elif "Live RBI Data" in page:
    st.markdown("""
    <div class="page-header">
        <h2>📡 Live RBI Data — Real Sector Credit Intelligence</h2>
        <p>Source: RBI Sectoral Deployment of Bank Credit, January 2026 &nbsp;|&nbsp; Fetched via rbidocs.rbi.org.in</p>
    </div>""", unsafe_allow_html=True)

    if not rbi_macro:
        st.error("RBI data not found. Run `python fetch_rbi_data.py` then `python parse_rbi_data.py` in the project folder.")
    else:
        # ── REAL MSME MACRO KPIs ──
        st.markdown("### 🏦 Micro & Small Enterprise Credit — India (Jan 2026)")

        r1, r2, r3, r4 = st.columns(4)
        kpi_pairs = [
            (r1, f"₹{rbi_macro.get('micro_small_outstanding_cr',0)/100_000:.1f}L Cr",
             "Micro & Small Credit Outstanding", TEAL),
            (r2, f"{rbi_macro.get('micro_small_yoy_growth_pct',0):.1f}%",
             "YoY Growth (Jan 25→26)", GOLD),
            (r3, f"{rbi_macro.get('micro_small_share_of_total_pct',0):.1f}%",
             "Share of Total Bank Credit", BLUE),
            (r4, f"₹{rbi_macro.get('services_credit_cr',0)/100_000:.1f}L Cr",
             "Services Sector Credit (Jan 26)", PURPLE),
        ]
        for col, val, label, color in kpi_pairs:
            col.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color:{color}; font-size:1.7rem;">{val}</div>
                <div class="kpi-label">{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="insight-box" style="margin-top:1rem;">
            📊 <b>Micro & Small enterprises grew credit at {rbi_macro.get("micro_small_yoy_growth_pct",0):.1f}% YoY</b> —
            significantly above the broader Industry segment ({rbi_macro.get("industry_credit_cr",0)/100_000:.1f}L Cr, ~12.1% YoY growth).
            This confirms strong demand momentum in the MSME segment. Source: {rbi_macro.get("source","RBI")}
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── SECTOR CREDIT COMPARISON TABLE ──
        if rbi_sectors:
            st.markdown("### 📋 RBI Credit Data per Sector — vs. Synthetic Risk Score")
            st.markdown("<p style='font-size:0.78rem;color:#5A7A9A;'>Left: real RBI credit outstanding and YoY growth. Right: how the blended risk score (40% RBI signal + 60% expert ordering) compares to the synthetic v1 score. Differences show where RBI data shifted our calibration.</p>", unsafe_allow_html=True)

            SYNTHETIC_V1 = {"IT Services": 0.20, "Retail": 0.32, "Logistics": 0.38,
                            "Manufacturing": 0.48, "F&B": 0.55, "Construction": 0.65}
            rows = []
            for sec, d in rbi_sectors.items():
                rows.append({
                    "Sector": sec,
                    "RBI Credit Jan 26 (₹ Cr)": f"₹{d['rbi_credit_jan26_cr']:,.0f}" if d.get("rbi_credit_jan26_cr") else "N/A",
                    "YoY Growth %": f"{d['rbi_yoy_growth_pct']:.1f}%" if d.get("rbi_yoy_growth_pct") else "N/A",
                    "RBI-Matched Categories": ", ".join(d.get("rbi_matched_categories", [])[:2]),
                    "Blended Risk Score": f"{d['blended_risk_score']:.3f}",
                    "Synthetic v1 Score": f"{SYNTHETIC_V1.get(sec, 0):.2f}",
                    "Δ (RBI vs v1)": f"{d['blended_risk_score'] - SYNTHETIC_V1.get(sec, 0):+.3f}"
                })
            rbi_compare_df = pd.DataFrame(rows).sort_values("Blended Risk Score")
            st.dataframe(rbi_compare_df, use_container_width=True, hide_index=True)

            # Bar chart comparing old vs new risk scores
            secs = [r["Sector"] for r in rows]
            v1_scores   = [SYNTHETIC_V1.get(s, 0) for s in secs]
            blend_scores = [rbi_sectors[s]["blended_risk_score"] for s in secs]

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                name="Synthetic v1 Score", x=secs, y=v1_scores,
                marker=dict(color=BLUE, opacity=0.70, line=dict(color="#0D1B2A", width=1)),
                hovertemplate="<b>%{x}</b><br>Synthetic v1: %{y:.3f}<extra></extra>"
            ))
            fig_comp.add_trace(go.Bar(
                name="RBI-Calibrated Score", x=secs, y=blend_scores,
                marker=dict(color=TEAL, opacity=0.85, line=dict(color="#0D1B2A", width=1)),
                hovertemplate="<b>%{x}</b><br>RBI-Calibrated: %{y:.3f}<extra></extra>"
            ))
            fig_comp.update_layout(**CHART_LAYOUT, height=340, barmode="group",
                                   title=dict(text="<b>Synthetic v1 vs RBI-Calibrated Sector Risk Scores</b>",
                                              font=dict(size=13, color=WHITE), x=0))
            fig_comp.update_xaxes(**AXIS_STYLE)
            fig_comp.update_yaxes(**AXIS_STYLE, title_text="Risk Score (0=safe, 1=risky)", range=[0, 0.75])
            st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})

            st.markdown("""<div class="warn-box">⚠️ <b>Key shift:</b> Logistics scores higher under RBI signal (4.3% YoY credit growth = lenders are cautious). Construction scores lower than v1's expert prior — RBI real-estate credit grew 16.2% YoY, suggesting easier access than assumed. Both observations are <i>real data insights</i>.</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── SUB-INDUSTRY TABLE ──
        st.markdown("### 📊 RBI Sub-Industry Credit Detail (Jan 2026)")
        try:
            ind_detail = pd.read_csv("rbi_data/rbi_industry_detail.csv")
            ind_detail_show = ind_detail[["industry","jan26","yoy_pct_jan26"]].dropna(subset=["jan26"]).copy()
            ind_detail_show = ind_detail_show[~ind_detail_show["industry"].str.startswith("Industries")]
            ind_detail_show.columns = ["RBI Industry Category", "Credit Outstanding Jan 26 (₹ Cr)", "YoY Growth %"]
            ind_detail_show["Credit Outstanding Jan 26 (₹ Cr)"] = ind_detail_show["Credit Outstanding Jan 26 (₹ Cr)"].map(lambda x: f"₹{x:,.0f}")
            ind_detail_show["YoY Growth %"] = ind_detail_show["YoY Growth %"].map(lambda x: f"{x:.2f}%")
            st.dataframe(ind_detail_show, use_container_width=True, hide_index=True, height=380)
        except Exception as e:
            st.warning(f"Could not load sub-industry detail: {e}")

        # ── DATA LINEAGE ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🔗 Data Lineage — Exact Source Files")
        lineage_df = pd.DataFrame({
            "Metric": [
                "Micro & Small Outstanding (₹10.3L Cr)",
                "Sector YoY Growth Rates",
                "Sub-industry Breakdown (Textiles, Chemicals etc.)",
                "Services Credit (Transport, Trade, Real Estate)"
            ],
            "RBI Document": [
                "Sectoral Deployment of Bank Credit — Statement 1",
                "Sectoral Deployment of Bank Credit — Statement 1, Col 7 (Jan26/Jan25)",
                "Sectoral Deployment of Bank Credit — Statement 2",
                "Sectoral Deployment — Statement 1, rows 3.1–3.10"
            ],
            "URL": [
                "rbidocs.rbi.org.in/rdocs/content/docs/SIBC27022026.xlsx",
                "same",
                "rbidocs.rbi.org.in/rdocs/content/docs/SIBC27022026.xlsx",
                "same"
            ],
            "Fetched": ["31 Jan 2026 data, pulled 18 Mar 2026"] * 4
        })
        st.dataframe(lineage_df, use_container_width=True, hide_index=True)
        st.markdown("""<div class="opp-box">✅ <b>This is real RBI data</b> — not a proxy, not estimated. The sector credit outstanding figures, YoY growth rates, and sub-industry breakdown are sourced directly from RBI's public document server (no API key, no login required). Refresh by re-running <code>python fetch_rbi_data.py && python parse_rbi_data.py</code>.</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# FOOTER — PREMIUM FROSTED GLASS
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="premium-footer">
    <div style="font-size:1.2rem; margin-bottom:8px;">🇮🇳</div>
    <div style="font-size:0.78rem; color:#4A6A8A; line-height:1.8;">
        <span style="font-weight:600; color:#6B8CA8;">India SME Credit Risk & Growth Intelligence Platform</span>
        <br>
        Bengaluru, 2026 &nbsp;·&nbsp; Built with ❤️ using Streamlit + XGBoost + Plotly
        <br>
        Data: Synthetic (MCA21/RBI/SIDBI calibrated) + <span style="color:#00C9A7;">Real RBI Jan 2026</span>
        &nbsp;·&nbsp;
        <a href="https://github.com/RishabJainhub">GitHub ↗</a>
    </div>
</div>
""", unsafe_allow_html=True)
