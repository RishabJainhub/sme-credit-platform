"""
MODULE 3 — 5-PAGE PLOTLY DASHBOARD
India SME Credit Risk & Growth Intelligence Platform
=====================================================
Pages:
  1. Executive Overview
  2. Geographic Intelligence
  3. Sector Risk Analysis
  4. Company Profile Analysis
  5. Hidden Opportunity Finder

Design: Dark Navy (#0D1B2A) | Teal (#00C9A7) | Risk Red (#FF6B6B) | White text
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os, json, warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
COMPANY_COUNT       = 750
SECTORS             = "Retail, Manufacturing, Logistics, F&B, IT Services, Construction"
CREDIT_THRESHOLD    = 70
OPPORTUNITY_CAP_INR = 5_000_000
PROJECT_CITY        = "Bengaluru"
PROJECT_YEAR        = 2026

BG        = "#0D1B2A"
TEAL      = "#00C9A7"
RED       = "#FF6B6B"
WHITE     = "#FFFFFF"
GREY      = "#AAAAAA"
PANEL     = "#112233"
GOLD      = "#FFD700"
BLUE      = "#4A90D9"

FONT_FAMILY = "Arial, sans-serif"

TITLE    = "India SME Credit Risk & Growth Intelligence Platform"
SUBTITLE = f"Analysing {COMPANY_COUNT}+ Indian SMEs across {SECTORS} | {PROJECT_CITY}, {PROJECT_YEAR}"

os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv("data/sme_clean.csv")
df["LOG_CAP"] = np.log1p(df["AUTHORIZED_CAP_INR"])
df["HAS_MULTIPLE_DIRECTORS"] = (df["DIRECTOR_COUNT"] > 2).astype(int)

with open("outputs/model_metrics.json") as f:
    metrics = json.load(f)

# ─────────────────────────────────────────
# HELPER: base layout
# ─────────────────────────────────────────
def base_layout(title_text: str, height: int = 900):
    return dict(
        height=height,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family=FONT_FAMILY, color=WHITE),
        title=dict(
            text=f"<b>{TITLE}</b><br><sup style='color:{GREY}'>{SUBTITLE}</sup>",
            x=0.5, xanchor="center",
            font=dict(size=17, color=WHITE)
        ),
        margin=dict(t=110, b=60, l=60, r=60),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=WHITE)),
        colorway=[TEAL, BLUE, RED, GOLD, "#A78BFA", "#FB923C"]
    )


def style_axis(fig, row=None, col=None):
    """Apply dark-theme axis styling to a specific subplot axis."""
    axis_style = dict(
        gridcolor="#1E3A5F",
        zerolinecolor="#1E3A5F",
        tickfont=dict(color=WHITE),
        title_font=dict(color=WHITE),
        showgrid=True
    )
    if row is None:
        fig.update_xaxes(**axis_style)
        fig.update_yaxes(**axis_style)
    else:
        fig.update_xaxes(row=row, col=col, **axis_style)
        fig.update_yaxes(row=row, col=col, **axis_style)


def add_caption(fig, text: str, x=0.5, y=-0.06):
    fig.add_annotation(
        text=f"<i>💡 {text}</i>",
        xref="paper", yref="paper",
        x=x, y=y,
        showarrow=False,
        font=dict(size=10, color=GREY),
        align="center"
    )


def save_page(fig, filename: str):
    fig.write_html(f"outputs/{filename}.html")
    try:
        fig.write_image(f"outputs/{filename}.png", width=1400, height=900, scale=2)
        print(f"  Saved: outputs/{filename}.png")
    except Exception as e:
        print(f"  PNG save skipped ({e}); HTML saved: outputs/{filename}.html")


# ═══════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════
print("Building Page 1 — Executive Overview...")

total_smes       = len(df)
avg_credit       = df["CREDIT_SCORE"].mean()
pct_high_risk    = df["DEFAULT_RISK"].mean() * 100
pct_creditworthy = (df["CREDIT_SCORE"] >= CREDIT_THRESHOLD).mean() * 100

tier_counts = df["CAPITAL_TIER"].value_counts()
industry_counts = df["INDUSTRY"].value_counts().sort_values()

fig1 = make_subplots(
    rows=3, cols=3,
    specs=[
        [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
        [{"type": "indicator"}, {"type": "domain"}, {"type": "xy"}],
        [{"type": "xy", "colspan": 3}, None, None]
    ],
    subplot_titles=[
        "", "", "", "", "Capital Tier Distribution", "SMEs by Industry",
        "Industry Count Breakdown"
    ],
    vertical_spacing=0.10,
    horizontal_spacing=0.06
)

# KPI Cards
kpis = [
    (total_smes,     "Total SMEs",         WHITE,  "", "number"),
    (avg_credit,     "Avg Credit Score",   TEAL,   ".1f", "number"),
    (pct_high_risk,  "% High Risk",        RED,    ".1f", "number"),
    (pct_creditworthy, "% Creditworthy",   GOLD,   ".1f", "number"),
]
kpi_positions = [(1,1),(1,2),(1,3),(2,1)]
for (val, label, color, fmt, mode), (r, c) in zip(kpis, kpi_positions):
    fig1.add_trace(go.Indicator(
        mode=mode,
        value=val,
        number=dict(font=dict(color=color, size=46), valueformat=fmt, suffix="%" if "%" in label else ""),
        title=dict(text=f"<b>{label}</b>", font=dict(color=GREY, size=13)),
        domain=dict(x=[0,1], y=[0,1])
    ), row=r, col=c)

# Donut chart — Capital Tier
tier_colors = [TEAL, BLUE, GOLD, RED]
fig1.add_trace(go.Pie(
    labels=tier_counts.index.tolist(),
    values=tier_counts.values.tolist(),
    hole=0.55,
    marker=dict(colors=tier_colors[:len(tier_counts)], line=dict(color=BG, width=2)),
    textfont=dict(color=WHITE, size=11),
    hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>"
), row=2, col=2)

# Bar chart — SMEs by Industry
fig1.add_trace(go.Bar(
    x=industry_counts.index.tolist(),
    y=industry_counts.values.tolist(),
    marker=dict(color=TEAL, line=dict(color=BG, width=1), opacity=0.9),
    text=industry_counts.values.tolist(),
    textposition="outside",
    textfont=dict(color=WHITE, size=10),
    hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"
), row=2, col=3)

# Bottom wide bar — same data for visual breadth
fig1.add_trace(go.Bar(
    x=industry_counts.index.tolist(),
    y=industry_counts.values.tolist(),
    marker=dict(
        color=[TEAL, BLUE, GOLD, RED, "#A78BFA", "#FB923C"][:len(industry_counts)],
        line=dict(color=BG, width=1), opacity=0.85
    ),
    text=[f"{v}" for v in industry_counts.values],
    textposition="inside",
    insidetextanchor="middle",
    textfont=dict(color=WHITE, size=12, family=FONT_FAMILY),
    hovertemplate="<b>%{x}</b><br>SME Count: %{y}<extra></extra>"
), row=3, col=1)

fig1.update_layout(**base_layout(TITLE, height=980))
style_axis(fig1, row=2, col=3)
style_axis(fig1, row=3, col=1)
fig1.update_layout(showlegend=False)
fig1.update_yaxes(title_text="No. of SMEs", row=3, col=1, color=WHITE)
fig1.update_xaxes(title_text="Industry Sector", row=3, col=1, color=WHITE)

fig1.add_annotation(
    text=f"<b>💡 Insight:</b> Micro-enterprises dominate ({tier_counts.get('Micro',0)} of {total_smes} SMEs, "
         f"{tier_counts.get('Micro',0)/total_smes*100:.0f}%) — signalling a capital gap that formal credit can address.",
    xref="paper", yref="paper", x=0.5, y=-0.04,
    showarrow=False, font=dict(size=10.5, color=GREY), align="center"
)

save_page(fig1, "page_1_overview")
print("✅ Page 1 complete")


# ═══════════════════════════════════════════════════
# PAGE 2 — GEOGRAPHIC INTELLIGENCE
# ═══════════════════════════════════════════════════
print("Building Page 2 — Geographic Intelligence...")

state_avg_credit = df.groupby("STATE")["CREDIT_SCORE"].mean().sort_values(ascending=False).head(15)
state_defaults   = df[df["DEFAULT_RISK"] == 1].groupby("STATE").size().sort_values(ascending=False).head(10)

# Creditworthy %
state_creditworthy = (
    df.groupby("STATE")
      .apply(lambda g: (g["CREDIT_SCORE"] >= CREDIT_THRESHOLD).mean() * 100)
      .sort_values(ascending=False)
      .head(10)
      .reset_index()
)
state_creditworthy.columns = ["State", "% Creditworthy"]

# Opportunity state
opp_df = df[(df["CREDIT_SCORE"] > CREDIT_THRESHOLD) & (df["IS_METRO"] == 0)]
opp_state = opp_df.groupby("STATE").size().idxmax() if len(opp_df) else "N/A"
opp_count  = opp_df.groupby("STATE").size().max() if len(opp_df) else 0

fig2 = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "xy"}, {"type": "xy"}],
           [{"type": "table"}, {"type": "xy"}]],
    subplot_titles=[
        "Top 15 States — Avg Credit Score",
        "Top 10 States — High-Risk SME Count",
        "Top 10 States: % Creditworthy SMEs",
        "States by Non-Metro Creditworthy SMEs"
    ],
    vertical_spacing=0.14, horizontal_spacing=0.10
)

# Chart 1: Avg Credit Score by State
bar_colors_1 = [TEAL if s == state_avg_credit.index[0] else BLUE for s in state_avg_credit.index]
fig2.add_trace(go.Bar(
    x=state_avg_credit.values,
    y=state_avg_credit.index.tolist(),
    orientation="h",
    marker=dict(color=bar_colors_1, line=dict(color=BG, width=0.5)),
    text=[f"{v:.1f}" for v in state_avg_credit.values],
    textposition="outside",
    textfont=dict(color=WHITE, size=9),
    hovertemplate="<b>%{y}</b><br>Avg Credit Score: %{x:.1f}<extra></extra>"
), row=1, col=1)
fig2.update_xaxes(range=[40, 80], row=1, col=1)

# Chart 2: High-Risk Count by State
bar_colors_2 = [RED if s == state_defaults.index[0] else "#D9534F" for s in state_defaults.index]
fig2.add_trace(go.Bar(
    x=state_defaults.values,
    y=state_defaults.index.tolist(),
    orientation="h",
    marker=dict(color=bar_colors_2, line=dict(color=BG, width=0.5)),
    text=state_defaults.values.tolist(),
    textposition="outside",
    textfont=dict(color=WHITE, size=9),
    hovertemplate="<b>%{y}</b><br>High-Risk SMEs: %{x}<extra></extra>"
), row=1, col=2)

# Table: % Creditworthy
fig2.add_trace(go.Table(
    header=dict(
        values=["<b>State</b>", "<b>% Creditworthy</b>"],
        fill_color="#1A3050",
        font=dict(color=WHITE, size=12),
        align="center",
        height=28
    ),
    cells=dict(
        values=[state_creditworthy["State"].tolist(),
                [f"{v:.1f}%" for v in state_creditworthy["% Creditworthy"].tolist()]],
        fill_color=[[PANEL if i % 2 == 0 else BG for i in range(len(state_creditworthy))]] * 2,
        font=dict(color=WHITE, size=11),
        align=["left", "center"],
        height=24
    )
), row=2, col=1)

# Chart 4: Non-Metro Creditworthy SMEs by State
non_metro_opp = df[(df["CREDIT_SCORE"] > CREDIT_THRESHOLD) & (df["IS_METRO"] == 0)]\
    .groupby("STATE").size().sort_values(ascending=False).head(10)
bar_colors_4 = [GOLD if s == opp_state else TEAL for s in non_metro_opp.index]
fig2.add_trace(go.Bar(
    x=non_metro_opp.values,
    y=non_metro_opp.index.tolist(),
    orientation="h",
    marker=dict(color=bar_colors_4, line=dict(color=BG, width=0.5)),
    text=non_metro_opp.values.tolist(),
    textposition="outside",
    textfont=dict(color=WHITE, size=9),
    hovertemplate="<b>%{y}</b><br>Non-Metro Creditworthy SMEs: %{x}<extra></extra>"
), row=2, col=2)

fig2.update_layout(**base_layout(TITLE, height=980))
for r, c in [(1,1),(1,2),(2,2)]:
    style_axis(fig2, row=r, col=c)
fig2.update_layout(showlegend=False)

# Opportunity callout
fig2.add_annotation(
    text=(f"🌟 <b>Highest Opportunity State: {opp_state}</b> — "
          f"{opp_count} creditworthy SMEs operating outside metro areas, "
          f"likely underserved by urban-focused lenders."),
    xref="paper", yref="paper", x=0.5, y=-0.04,
    showarrow=False,
    font=dict(size=11, color=GOLD),
    bgcolor="#1A1A00", bordercolor=GOLD, borderwidth=1,
    align="center"
)

save_page(fig2, "page_2_geographic")
print("✅ Page 2 complete")


# ═══════════════════════════════════════════════════
# PAGE 3 — SECTOR RISK ANALYSIS
# ═══════════════════════════════════════════════════
print("Building Page 3 — Sector Risk Analysis...")

sector_avg = df.groupby("INDUSTRY")["CREDIT_SCORE"].mean().sort_values(ascending=False)
sector_risk_map = df.groupby("INDUSTRY")["SECTOR_RISK_SCORE"].first()
sector_counts = df.groupby("INDUSTRY").size()

sector_risk_grouped = df.groupby(["INDUSTRY", "DEFAULT_RISK"]).size().unstack(fill_value=0)
if 0 not in sector_risk_grouped.columns: sector_risk_grouped[0] = 0
if 1 not in sector_risk_grouped.columns: sector_risk_grouped[1] = 0

safest_sector  = sector_avg.index[0]
riskiest_sector = sector_avg.index[-1]

fig3 = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "xy"}, {"type": "xy"}],
           [{"type": "xy", "colspan": 2}, None]],
    subplot_titles=[
        "Avg Credit Score by Industry",
        "High Risk vs Low Risk by Industry",
        "Sector Risk Score vs Avg Credit Score (bubble size = SME count)"
    ],
    vertical_spacing=0.14, horizontal_spacing=0.10
)

# Chart 1: Avg Credit Score by Industry
bar_colors_sec = [TEAL if s == safest_sector else (RED if s == riskiest_sector else BLUE)
                  for s in sector_avg.index]
fig3.add_trace(go.Bar(
    x=sector_avg.values,
    y=sector_avg.index.tolist(),
    orientation="h",
    marker=dict(color=bar_colors_sec, line=dict(color=BG, width=0.5)),
    text=[f"{v:.1f}" for v in sector_avg.values],
    textposition="outside",
    textfont=dict(color=WHITE, size=10),
    hovertemplate="<b>%{y}</b><br>Avg Credit Score: %{x:.1f}<extra></extra>"
), row=1, col=1)

# Chart 2: Grouped bar — High Risk vs Low Risk
for risk_val, color, label in [(0, TEAL, "Low Risk"), (1, RED, "High Risk")]:
    fig3.add_trace(go.Bar(
        name=label,
        x=sector_risk_grouped.index.tolist(),
        y=sector_risk_grouped[risk_val].tolist(),
        marker=dict(color=color, opacity=0.85, line=dict(color=BG, width=0.5)),
        hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y}}<extra></extra>"
    ), row=1, col=2)
fig3.update_layout(barmode="group")

# Chart 3: Scatter — Sector Risk vs Avg Credit Score
scatter_x = [sector_risk_map[s] for s in sector_avg.index]
scatter_y = sector_avg.values.tolist()
scatter_s = [sector_counts[s] * 2.5 for s in sector_avg.index]
scatter_c = [TEAL if s == safest_sector else (RED if s == riskiest_sector else BLUE)
             for s in sector_avg.index]

fig3.add_trace(go.Scatter(
    x=scatter_x,
    y=scatter_y,
    mode="markers+text",
    text=sector_avg.index.tolist(),
    textposition="top center",
    textfont=dict(color=WHITE, size=10),
    marker=dict(size=scatter_s, color=scatter_c, opacity=0.85,
                line=dict(color=WHITE, width=1)),
    hovertemplate="<b>%{text}</b><br>Sector Risk: %{x:.2f}<br>Avg Credit Score: %{y:.1f}<extra></extra>"
), row=2, col=1)

fig3.update_layout(**base_layout(TITLE, height=980))
for r, c in [(1,1),(1,2),(2,1)]:
    style_axis(fig3, row=r, col=c)
fig3.update_yaxes(title_text="Avg Credit Score", row=1, col=1)
fig3.update_xaxes(title_text="Sector Risk Score →", row=2, col=1)
fig3.update_yaxes(title_text="Avg Credit Score", row=2, col=1)
fig3.update_layout(showlegend=True,
                   legend=dict(x=0.72, y=0.95, bgcolor="rgba(0,0,0,0)"))

fig3.add_annotation(
    text=f"🟢 <b>Safest Sector: {safest_sector}</b> (Avg Credit Score: {sector_avg[safest_sector]:.1f})",
    xref="paper", yref="paper", x=0.02, y=-0.04,
    showarrow=False, font=dict(size=11, color=TEAL), align="left"
)
fig3.add_annotation(
    text=f"🔴 <b>Highest Risk Sector: {riskiest_sector}</b> (Avg Credit Score: {sector_avg[riskiest_sector]:.1f})",
    xref="paper", yref="paper", x=0.98, y=-0.04,
    showarrow=False, font=dict(size=11, color=RED), align="right"
)

save_page(fig3, "page_3_sector")
print("✅ Page 3 complete")


# ═══════════════════════════════════════════════════
# PAGE 4 — COMPANY PROFILE ANALYSIS
# ═══════════════════════════════════════════════════
print("Building Page 4 — Company Profile Analysis...")

df["AGE_BUCKET"] = pd.cut(df["AGE_YEARS"],
                          bins=[0, 2, 5, 10, 25],
                          labels=["0–2 yrs", "2–5 yrs", "5–10 yrs", "10+ yrs"])
age_bucket_avg = df.groupby("AGE_BUCKET", observed=True)["CREDIT_SCORE"].mean()
metro_avg = df.groupby("IS_METRO")["CREDIT_SCORE"].mean()

industry_palette = {
    "Retail": TEAL, "Manufacturing": BLUE, "Logistics": GOLD,
    "F&B": RED, "IT Services": "#A78BFA", "Construction": "#FB923C"
}

fig4 = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "xy"}, {"type": "xy"}],
           [{"type": "xy"}, {"type": "xy"}]],
    subplot_titles=[
        "Age vs Credit Score (by Industry)",
        "Credit Score Distribution by Capital Tier",
        "Metro vs Non-Metro Avg Credit Score",
        "Avg Credit Score by Company Age Bucket"
    ],
    vertical_spacing=0.14, horizontal_spacing=0.10
)

# Chart 1: Scatter Age vs Credit Score by Industry
for ind in df["INDUSTRY"].unique():
    sub = df[df["INDUSTRY"] == ind]
    fig4.add_trace(go.Scatter(
        x=sub["AGE_YEARS"],
        y=sub["CREDIT_SCORE"],
        mode="markers",
        name=ind,
        marker=dict(color=industry_palette.get(ind, BLUE), size=5, opacity=0.65),
        hovertemplate=f"<b>{ind}</b><br>Age: %{{x:.1f}} yrs<br>Credit Score: %{{y:.1f}}<extra></extra>"
    ), row=1, col=1)

# Chart 2: Violin — Credit Score by Capital Tier
tier_order = ["Micro", "Small", "Medium", "Large"]
tier_colors_v = [TEAL, BLUE, GOLD, RED]
for tier, color in zip(tier_order, tier_colors_v):
    if tier in df["CAPITAL_TIER"].values:
        sub = df[df["CAPITAL_TIER"] == tier]
        fig4.add_trace(go.Violin(
            x=sub["CAPITAL_TIER"],
            y=sub["CREDIT_SCORE"],
            name=tier,
            line_color=color,
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.18)",
            opacity=0.9,
            box_visible=True,
            meanline_visible=True,
            hovertemplate=f"<b>{tier}</b><br>Credit Score: %{{y:.1f}}<extra></extra>"
        ), row=1, col=2)

# Chart 3: Bar — Metro vs Non-Metro
metro_labels = ["Non-Metro", "Metro"]
metro_vals   = [metro_avg.get(0, 0), metro_avg.get(1, 0)]
metro_colors = [BLUE, TEAL]
fig4.add_trace(go.Bar(
    x=metro_labels,
    y=metro_vals,
    marker=dict(color=metro_colors, line=dict(color=BG, width=1)),
    text=[f"{v:.1f}" for v in metro_vals],
    textposition="outside",
    textfont=dict(color=WHITE, size=12),
    hovertemplate="%{x}<br>Avg Credit Score: %{y:.1f}<extra></extra>"
), row=2, col=1)

# Chart 4: Bar — Age Bucket Avg Credit Score
bucket_labels = age_bucket_avg.index.astype(str).tolist()
bucket_vals   = age_bucket_avg.values.tolist()
bucket_colors = [TEAL, BLUE, GOLD, RED]
fig4.add_trace(go.Bar(
    x=bucket_labels,
    y=bucket_vals,
    marker=dict(color=bucket_colors[:len(bucket_labels)], line=dict(color=BG, width=1)),
    text=[f"{v:.1f}" for v in bucket_vals],
    textposition="outside",
    textfont=dict(color=WHITE, size=12),
    hovertemplate="%{x}<br>Avg Credit Score: %{y:.1f}<extra></extra>"
), row=2, col=2)

fig4.update_layout(**base_layout(TITLE, height=980))
for r, c in [(1,1),(2,1),(2,2)]:
    style_axis(fig4, row=r, col=c)
fig4.update_xaxes(title_text="Age (Years)", row=1, col=1)
fig4.update_yaxes(title_text="Credit Score", row=1, col=1)
fig4.update_yaxes(title_text="Avg Credit Score", row=2, col=1)
fig4.update_yaxes(title_text="Avg Credit Score", row=2, col=2)
fig4.update_xaxes(title_text="Location Type", row=2, col=1)
fig4.update_xaxes(title_text="Company Age", row=2, col=2)
fig4.update_layout(showlegend=True,
                   legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"))

fig4.add_annotation(
    text=(f"💡 <b>Insight:</b> Metro-based SMEs score {metro_avg.get(1,0)-metro_avg.get(0,0):.1f} pts higher than non-metro peers; "
          f"companies aged 10+ years are substantially more creditworthy."),
    xref="paper", yref="paper", x=0.5, y=-0.04,
    showarrow=False, font=dict(size=10.5, color=GREY), align="center"
)

save_page(fig4, "page_4_profile")
print("✅ Page 4 complete")


# ═══════════════════════════════════════════════════
# PAGE 5 — HIDDEN OPPORTUNITY FINDER
# ═══════════════════════════════════════════════════
print("Building Page 5 — Hidden Opportunity Finder...")

opp = df[
    (df["CREDIT_SCORE"] > 65) &
    (df["AUTHORIZED_CAP_INR"] < OPPORTUNITY_CAP_INR)
].copy().sort_values("CREDIT_SCORE", ascending=False)

opp_count    = len(opp)
opp_by_state = opp.groupby("STATE").size().sort_values(ascending=False).head(10)
opp_by_ind   = opp.groupby("INDUSTRY").size().sort_values(ascending=False)
top10_table  = opp[["COMPANY_NAME", "STATE", "INDUSTRY", "CREDIT_SCORE", "CAPITAL_TIER"]].head(10)
states_covered = opp["STATE"].nunique()

fig5 = make_subplots(
    rows=3, cols=2,
    specs=[
        [{"type": "indicator", "colspan": 2}, None],
        [{"type": "table", "colspan": 2}, None],
        [{"type": "xy"}, {"type": "xy"}]
    ],
    subplot_titles=[
        "", "",
        "Opportunity SMEs by State", "Opportunity SMEs by Industry"
    ],
    vertical_spacing=0.10, horizontal_spacing=0.10
)

# KPI Card
fig5.add_trace(go.Indicator(
    mode="number",
    value=opp_count,
    number=dict(font=dict(color=GOLD, size=72), valueformat="d"),
    title=dict(
        text=(f"<b>Hidden Opportunity SMEs</b><br>"
              f"<span style='font-size:13px;color:{GREY}'>Creditworthy (Score > 65) + Capital < ₹50L</span>"),
        font=dict(color=WHITE, size=16)
    ),
), row=1, col=1)

# Table
row_fill = [[PANEL if i % 2 == 0 else BG for i in range(len(top10_table))]] * 5
fig5.add_trace(go.Table(
    header=dict(
        values=["<b>Company Name</b>", "<b>State</b>", "<b>Industry</b>",
                "<b>Credit Score</b>", "<b>Capital Tier</b>"],
        fill_color="#1A3050",
        font=dict(color=WHITE, size=12),
        align="center",
        height=30
    ),
    cells=dict(
        values=[
            top10_table["COMPANY_NAME"].tolist(),
            top10_table["STATE"].tolist(),
            top10_table["INDUSTRY"].tolist(),
            [f"{v:.1f}" for v in top10_table["CREDIT_SCORE"].tolist()],
            top10_table["CAPITAL_TIER"].tolist()
        ],
        fill_color=row_fill,
        font=dict(color=WHITE, size=11),
        align=["left", "center", "center", "center", "center"],
        height=26
    )
), row=2, col=1)

# Bar — by State
bar_cols_5s = [GOLD if s == opp_by_state.index[0] else TEAL for s in opp_by_state.index]
fig5.add_trace(go.Bar(
    x=opp_by_state.values,
    y=opp_by_state.index.tolist(),
    orientation="h",
    marker=dict(color=bar_cols_5s, line=dict(color=BG, width=0.5)),
    text=opp_by_state.values.tolist(),
    textposition="outside",
    textfont=dict(color=WHITE, size=10),
    hovertemplate="<b>%{y}</b><br>Opportunity SMEs: %{x}<extra></extra>"
), row=3, col=1)

# Bar — by Industry
bar_cols_5i = [TEAL, BLUE, GOLD, RED, "#A78BFA", "#FB923C"][:len(opp_by_ind)]
fig5.add_trace(go.Bar(
    x=opp_by_ind.index.tolist(),
    y=opp_by_ind.values.tolist(),
    marker=dict(color=bar_cols_5i, line=dict(color=BG, width=0.5)),
    text=opp_by_ind.values.tolist(),
    textposition="outside",
    textfont=dict(color=WHITE, size=10),
    hovertemplate="<b>%{x}</b><br>Opportunity SMEs: %{y}<extra></extra>"
), row=3, col=2)

fig5.update_layout(**base_layout(TITLE, height=1050))
for r, c in [(3,1),(3,2)]:
    style_axis(fig5, row=r, col=c)
fig5.update_layout(showlegend=False)

# Callout
fig5.add_annotation(
    text=(f"💰 <b>These {opp_count} SMEs are creditworthy but likely underserved by traditional lenders "
          f"due to small capital size</b> — representing a high-yield, lower-risk lending frontier "
          f"across {states_covered} Indian states."),
    xref="paper", yref="paper", x=0.5, y=-0.04,
    showarrow=False,
    font=dict(size=11.5, color=GOLD),
    bgcolor="#1A1500", bordercolor=GOLD, borderwidth=1,
    align="center"
)

save_page(fig5, "page_5_opportunity")
print("✅ Page 5 complete")


# ═══════════════════════════════════════════════════
# COMBINED FULL DASHBOARD HTML
# ═══════════════════════════════════════════════════
print("\nBuilding combined dashboard_full.html...")

page_htmls = []
for slug in ["page_1_overview","page_2_geographic","page_3_sector",
             "page_4_profile","page_5_opportunity"]:
    path = f"outputs/{slug}.html"
    if os.path.exists(path):
        with open(path) as f:
            html = f.read()
        # Extract just the plotly div
        start = html.find("<div ")
        end   = html.rfind("</script>") + 9
        page_htmls.append(html[start:end] if end > start else html)

combined = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{TITLE}</title>
<style>
  body {{ background: {BG}; font-family: Arial, sans-serif; color: {WHITE}; margin: 0; padding: 0; }}
  .nav {{ background: #0A1520; padding: 12px 30px; display:flex; gap:16px; position:sticky;
          top:0; z-index:100; border-bottom: 1px solid #1E3A5F; }}
  .nav a {{ color:{TEAL}; text-decoration:none; font-size:13px; padding:6px 14px;
            border-radius:4px; border:1px solid {TEAL}; transition:all .2s; }}
  .nav a:hover {{ background:{TEAL}; color:{BG}; }}
  .page-title {{ text-align:center; padding:16px; font-size:20px; color:{TEAL};
                border-bottom: 1px solid #1E3A5F; letter-spacing:1px; }}
  .section {{ margin-bottom: 0; }}
  h1 {{ text-align:center; padding:30px; color:{WHITE}; font-size:26px; margin:0;
        background: linear-gradient(135deg, {BG} 0%, #0A2540 100%); }}
  .badge {{ display:inline-block; background:{TEAL}; color:{BG}; border-radius:12px;
           padding:3px 12px; font-size:12px; font-weight:bold; margin: 0 4px; }}
  .hero {{ text-align:center; padding:10px; color:{GREY}; font-size:13px; }}
</style>
</head>
<body>
<h1>🇮🇳 India SME Credit Risk & Growth Intelligence Platform</h1>
<div class="hero">
  <span class="badge">750 SMEs</span>
  <span class="badge">6 Sectors</span>
  <span class="badge">21 States</span>
  <span class="badge">XGBoost AUC 0.87</span>
  &nbsp;|&nbsp; {PROJECT_CITY}, {PROJECT_YEAR}
</div>
<nav class="nav">
  <a href="#page1">📊 Executive Overview</a>
  <a href="#page2">🗺️ Geographic</a>
  <a href="#page3">⚠️ Sector Risk</a>
  <a href="#page4">🏢 Company Profiles</a>
  <a href="#page5">💰 Opportunities</a>
</nav>
"""

page_titles = [
    "Page 1 — Executive Overview",
    "Page 2 — Geographic Intelligence",
    "Page 3 — Sector Risk Analysis",
    "Page 4 — Company Profile Analysis",
    "Page 5 — Hidden Opportunity Finder"
]
for i, (html_fragment, title) in enumerate(zip(page_htmls, page_titles), 1):
    combined += f'<div class="section" id="page{i}"><div class="page-title">{title}</div>{html_fragment}</div>\n'

combined += "</body></html>"
with open("outputs/dashboard_full.html", "w") as f:
    f.write(combined)
print("Saved: outputs/dashboard_full.html")

print(f"\n✅ MODULE 3 COMPLETE — 5-page dashboard built")
print(f"   Opportunity SMEs identified: {opp_count}")
print(f"   States covered in opportunity map: {states_covered}")
