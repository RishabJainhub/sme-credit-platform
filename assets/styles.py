import streamlit as st

def apply_custom_styles():
    """
    Applies the global CSS for the Magic UI 2.0 theme.
    Moved to a separate file to ensure GitHub language stats reflect Python dominance.
    """
    st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

  /* ── ROOT VARIABLES ── */
  :root {
      --bg-deep:      #0A0E1A;
      --bg-sidebar:   #0D1224;
      --accent-indigo: #6366F1;
      --accent-pink:   #EC4899;
      --success:      #10B981;
      --warning:      #F59E0B;
      --danger:       #EF4444;
      --glass-bg:     rgba(13, 18, 36, 0.52);
      --glass-border: rgba(255, 255, 255, 0.12);
  }

  /* ── MAGIC UI ANIMATIONS ── */
  @keyframes countUp {
      from { opacity: 0; transform: translateY(12px); }
      to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes shimmer {
      0%   { background-position: 200% 0; }
      100% { background-position: -200% 0; }
  }
  @keyframes borderSpin {
      to { --angle: 360deg; }
  }
  @keyframes gradientShift {
      0%   { background-position: 0% 50%; }
      50%  { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
  }
  @keyframes marquee {
      from { transform: translateX(0); }
      to   { transform: translateX(-50%); }
  }
  @property --angle {
    syntax: '<angle>';
    initial-value: 0deg;
    inherits: false;
  }

  /* ── BASE ── */
  html, body, [class*="css"] {
      font-family: 'Inter', sans-serif;
      background-color: var(--bg-deep) !important;
      color: #F1F5F9;
  }

  /* ── SHIMMER BUTTON ── */
  div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(90deg, #060D18 0%, #1a2535 40%, #060D18 100%);
    background-size: 200% 100%;
    border: 1px solid #00D4AA;
    border-radius: 8px;
    color: #00D4AA;
    font-weight: 600;
    animation: shimmer 2.5s linear infinite;
    transition: all 0.2s;
    padding: 0.6rem 2rem;
  }
  div[data-testid="stButton"] > button[kind="primary"]:hover {
    box-shadow: 0 0 20px rgba(0, 212, 170, 0.3);
    transform: translateY(-1px);
  }

  /* ── TABS (SUI STYLE) ── */
  .stTabs [data-baseweb="tab-list"] {
      gap: 24px;
      padding: 0 1rem;
      border-bottom: 2px solid var(--glass-border);
      background: transparent;
      margin-bottom: 2rem;
  }
  .stTabs [data-baseweb="tab"] {
      color: #94A3B8;
      font-weight: 500;
      font-size: 0.9rem;
      border: none;
      background: transparent;
      padding-bottom: 12px;
      transition: all 0.2s;
  }
  .stTabs [aria-selected="true"] {
      color: var(--accent-indigo) !important;
      border-bottom: 3px solid var(--accent-indigo) !important;
      font-weight: 700 !important;
      background: transparent !important;
  }

  /* ── PREV CSS REPLACED WITH CLEANER FOR MAGIC UI ── */
  .kpi-card {
      background: var(--glass-bg);
      backdrop-filter: blur(6px);
      border: 1px solid var(--glass-border);
      border-radius: 16px;
      padding: 1.5rem;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      position: relative;
      overflow: hidden;
  }
  .kpi-card:hover {
      border-color: rgba(99, 102, 241, 0.4);
      transform: translateY(-4px);
      box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
  }
  .dot-pattern {
    position: absolute; inset: 0; opacity: 0.05;
    background-image: radial-gradient(circle, #fff 1px, transparent 1px);
    background-size: 16px 16px;
  }

  /* ── INSIGHT BOX — FROSTED GLASS ── */
  .insight-box {
      background: linear-gradient(135deg, rgba(99,102,241,0.10) 0%, rgba(99,102,241,0.03) 100%);
      border: 1px solid rgba(99,102,241,0.20);
      border-left: 3px solid #6366F1;
      border-radius: 12px;
      padding: 0.9rem 1.2rem;
      margin: 0.8rem 0;
      font-size: 0.82rem;
      color: #CBD5E1;
      backdrop-filter: blur(6px);
      transition: all 0.3s ease;
  }
  .insight-box:hover {
      border-color: rgba(99,102,241,0.35);
      box-shadow: 0 4px 18px rgba(99,102,241,0.12);
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
      box-shadow: 0 0 20px rgba(255,215,0,0.2);
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
      box-shadow: 0 0 20px rgba(255,107,107,0.2);
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
      box-shadow: 0 0 20px rgba(0,201,167,0.4) !important;
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

  /* ── COMMAND CENTER GRID ── */
  .command-center {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    grid-template-rows: auto;
    gap: 1.25rem;
    margin-bottom: 2rem;
  }
  .bento-main { grid-column: span 2; grid-row: span 2; }
  .bento-side { grid-column: span 1; }
  .bento-bottom { grid-column: span 2; }

  /* ── LIVE SIGNAL FEED (TERMINAL STYLE) ── */
  .signal-feed {
    background: #020617;
    border: 1px solid #1E293B;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #00D4AA;
    position: relative;
    overflow: hidden;
    margin-top: 1.5rem;
  }
  .signal-feed::before {
    content: 'LIVE SIGNAL';
    display: block;
    font-size: 0.6rem;
    color: #475569;
    margin-bottom: 4px;
    letter-spacing: 0.1em;
  }
  .signal-line {
    white-space: normal;
    overflow: visible;
  }

  /* ── SECTOR OUTLOOK CARDS ── */
  .outlook-card {
    background: rgba(14,30,52,0.45);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1rem;
    transition: all 0.3s ease;
  }
  .outlook-card:hover {
    border-color: #4A90D9;
    box-shadow: 0 4px 25px rgba(74,144,217,0.1);
  }
  .bullish { color: #00D4AA; border-left: 3px solid #00D4AA; }
  .bearish { color: #FF4B4B; border-left: 3px solid #FF4B4B; }

  /* ── GLOBAL GRID BACKGROUND ── */
  .stApp {
    background-image: 
      linear-gradient(rgba(30,58,95,0.05) 1px, transparent 1px),
      linear-gradient(90deg, rgba(30,58,95,0.05) 1px, transparent 1px);
    background-size: 60px 60px;
  }

  /* ── RESPONSIVE LAYOUT ── */
  @media (max-width: 1024px) {
    .command-center {
      grid-template-columns: 1fr;
      gap: 0.9rem;
    }
    .kpi-card { padding: 1rem; }
    .main-header h1 { font-size: 1.45rem !important; }
  }

  @media (max-width: 768px) {
    .stTabs [data-baseweb="tab-list"] {
      gap: 10px;
      padding: 0 0.25rem;
      overflow-x: auto;
      white-space: nowrap;
    }
    .stTabs [data-baseweb="tab"] {
      font-size: 0.78rem;
      padding-bottom: 8px;
    }
  }
</style>
""", unsafe_allow_html=True)
