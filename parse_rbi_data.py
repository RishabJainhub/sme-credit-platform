"""
parse_rbi_data.py
=================
Parses the raw RBI Excel CSVs and produces:
  1. rbi_data/rbi_sector_calibration.json  — sector credit outstanding + YoY growth
                                             used to override synthetic NPA penalties
  2. rbi_data/rbi_msme_macro.json          — Micro & Small aggregate stats for the dashboard
  3. rbi_data/rbi_industry_detail.csv      — sub-industry outstanding (clean, long-format)

Run AFTER fetch_rbi_data.py
"""

import pandas as pd
import numpy as np
import json
import os

# ─────────────────────────────────────────────────────────────
# PART 1 — Parse Statement 1 (Macro Sectors)
# ─────────────────────────────────────────────────────────────
s1 = pd.read_csv("rbi_data/sectoral_deployment_jan2026_Statement_1.csv", header=None)

# Column layout (confirmed from raw preview):
# Col 0: Sector label
# Col 1: Jan-2024 outstanding (₹ Cr)
# Col 2: Mar-2024 outstanding
# Col 3: Jan-2025 outstanding
# Col 4: Mar-2025 outstanding
# Col 5: Jan-2026 outstanding  ← LATEST
# Col 6: YoY % change Jan25/Jan24
# Col 7: YoY % change Jan26/Jan25  ← MOST RECENT GROWTH
# Col 8: Financial year % change
# Col 9: Financial year % change

# Find data rows (skip title/header rows)
data_rows = []
for i, row in s1.iterrows():
    label = str(row.iloc[0]).strip()
    val5  = row.iloc[5]
    try:
        float(val5)
        is_num = True
    except (ValueError, TypeError):
        is_num = False
    if is_num and len(label) > 3:
        data_rows.append(i)

s1_clean = s1.iloc[data_rows].copy().reset_index(drop=True)
s1_clean.columns = ["sector", "jan24", "mar24", "jan25", "mar25", "jan26",
                    "yoy_pct_jan25", "yoy_pct_jan26", "fy_pct_24", "fy_pct_25"]

for col in ["jan24","mar24","jan25","mar25","jan26","yoy_pct_jan25","yoy_pct_jan26"]:
    s1_clean[col] = pd.to_numeric(s1_clean[col], errors="coerce")

s1_clean["sector"] = s1_clean["sector"].str.strip()

print("=== Statement 1 — Clean sectors ===")
print(s1_clean[["sector","jan26","yoy_pct_jan26"]].to_string())

# ─────────────────────────────────────────────────────────────
# PART 2 — Extract Micro & Small macro stats
# ─────────────────────────────────────────────────────────────
msme_row = s1_clean[s1_clean["sector"].str.contains("Micro and Small", na=False)]
print(f"\n=== Micro & Small row ===\n{msme_row.to_string()}")

micro_small_jan26      = float(msme_row["jan26"].values[0])   if len(msme_row) else None
micro_small_yoy_pct    = float(msme_row["yoy_pct_jan26"].values[0]) if len(msme_row) else None
industry_total_jan26   = s1_clean[s1_clean["sector"].str.contains("2\\. Industry", na=False, regex=True)]
industry_jan26         = float(industry_total_jan26["jan26"].values[0]) if len(industry_total_jan26) else None
services_row           = s1_clean[s1_clean["sector"].str.contains("3\\. Services|Services$", na=False, regex=True)]
services_jan26         = float(services_row["jan26"].values[0]) if len(services_row) else None
nonfood_row            = s1_clean[s1_clean["sector"].str.contains("Non-food", na=False)]
total_credit_jan26     = float(nonfood_row["jan26"].values[0]) if len(nonfood_row) else None

msme_share_pct = (micro_small_jan26 / total_credit_jan26 * 100) if (micro_small_jan26 and total_credit_jan26) else None

msme_macro = {
    "source": "RBI Sectoral Deployment of Bank Credit, January 2026",
    "date": "31 January 2026",
    "total_nonfood_credit_cr": round(micro_small_jan26 / 100_000, 2) if micro_small_jan26 else None,  # convert ₹Cr to ₹L Cr
    "total_bank_credit_cr": round(total_credit_jan26 / 100_000, 2) if total_credit_jan26 else None,
    "micro_small_outstanding_cr": round(micro_small_jan26, 0) if micro_small_jan26 else None,
    "micro_small_yoy_growth_pct": round(micro_small_yoy_pct, 2) if micro_small_yoy_pct else None,
    "micro_small_share_of_total_pct": round(msme_share_pct, 2) if msme_share_pct else None,
    "industry_credit_cr": round(industry_jan26, 0) if industry_jan26 else None,
    "services_credit_cr": round(services_jan26, 0) if services_jan26 else None,
    "note": "Values in ₹ Crore. Micro & Small = credit to micro and small enterprises as per RBI definition."
}
with open("rbi_data/rbi_msme_macro.json", "w") as f:
    json.dump(msme_macro, f, indent=2)
print(f"\n=== MSME Macro Stats ===")
print(json.dumps(msme_macro, indent=2))

# ─────────────────────────────────────────────────────────────
# PART 3 — Parse Statement 2 (Industry-wise breakdown)
# ─────────────────────────────────────────────────────────────
s2 = pd.read_csv("rbi_data/sectoral_deployment_jan2026_Statement_2.csv", header=None)

data_rows2 = []
for i, row in s2.iterrows():
    label = str(row.iloc[0]).strip()
    val5  = row.iloc[5]
    try:
        float(val5)
        is_num = True
    except (ValueError, TypeError):
        is_num = False
    if is_num and len(label) > 3:
        data_rows2.append(i)

s2_clean = s2.iloc[data_rows2].copy().reset_index(drop=True)
s2_clean.columns = ["industry", "jan24", "mar24", "jan25", "mar25", "jan26",
                    "yoy_pct_jan25", "yoy_pct_jan26", "fy_pct_24", "fy_pct_25"]
for col in ["jan24","jan25","jan26","yoy_pct_jan26"]:
    s2_clean[col] = pd.to_numeric(s2_clean[col], errors="coerce")
s2_clean["industry"] = s2_clean["industry"].str.strip()
s2_clean.to_csv("rbi_data/rbi_industry_detail.csv", index=False)
print(f"\n=== Statement 2 — Industry detail ({len(s2_clean)} rows) ===")
print(s2_clean[["industry","jan26","yoy_pct_jan26"]].to_string())

# ─────────────────────────────────────────────────────────────
# PART 4 — Map RBI sub-industries → our 6 dashboard sectors
#
# Strategy: use real RBI credit outstanding Jan26 to compute
#           sector share of total industry credit.
#           Higher credit share + lower YoY growth → more mature/constrained
#           sector → higher NPA risk proxy.
#
# For each of our 6 sectors, we pick the closest RBI sub-industry:
# ─────────────────────────────────────────────────────────────

SECTOR_TO_RBI = {
    "IT Services":   ["Computer Software", "Professional Services"],
    "Retail":        ["Retail Trade", "Wholesale Trade"],
    "Logistics":     ["Transport Operators"],
    "Manufacturing": ["Textiles", "Chemicals and Chemical Products",
                      "Food Processing", "Rubber, Plastic"],
    "F&B":           ["Food Processing", "Beverage and Tobacco"],
    "Construction":  ["Commercial Real Estate"],
}

def match_rbi(keywords, df, label_col="industry"):
    """Sum credit outstanding for rows matching any keyword."""
    mask = df[label_col].str.contains("|".join(keywords), case=False, na=False)
    matched = df[mask]
    return matched["jan26"].sum(), matched["yoy_pct_jan26"].mean(), list(matched[label_col])

sector_calibration = {}
for sec, keywords in SECTOR_TO_RBI.items():
    outstanding, yoy_avg, matched = match_rbi(keywords, s2_clean, "industry")
    if outstanding == 0:
        outstanding, yoy_avg, matched = match_rbi(keywords, s1_clean, "sector")  # fallback

    sector_calibration[sec] = {
        "rbi_credit_jan26_cr": round(float(outstanding), 0) if outstanding else None,
        "rbi_yoy_growth_pct": round(float(yoy_avg), 2) if yoy_avg else None,
        "rbi_matched_categories": matched,
    }

# ─────────────────────────────────────────────────────────────
# PART 5 — Derive calibrated risk scores from real data
#
# Risk score logic:
#   inverse_growth = 1/(1 + yoy_growth_pct/100)  — slowing credit = more risk
#   size_rank = rank by credit outstanding (larger = more established = less risky for SMEs)
#   Combined → normalized 0-1 risk score
# ─────────────────────────────────────────────────────────────
calibrated = []
for sec, d in sector_calibration.items():
    yoy = d["rbi_yoy_growth_pct"] or 10.0
    outstanding = d["rbi_credit_jan26_cr"] or 1000
    # Lower growth & smaller scale → higher risk
    inv_growth = 1 / (1 + yoy / 100)
    calibrated.append({
        "sector": sec,
        "outstanding_cr": outstanding,
        "yoy_growth_pct": yoy,
        "inv_growth": inv_growth,
        **d
    })

cal_df = pd.DataFrame(calibrated)
# Normalize inv_growth to 0–1 risk score
min_ig, max_ig = cal_df["inv_growth"].min(), cal_df["inv_growth"].max()
if max_ig > min_ig:
    cal_df["rbi_risk_score"] = ((cal_df["inv_growth"] - min_ig) / (max_ig - min_ig))
else:
    cal_df["rbi_risk_score"] = 0.3

# Preserve ordering intuition (IT < Retail < Logistics < Mfg < F&B < Construction)
# by blending RBI signal 40% + original expert ordering 60%
EXPERT_ORDER = {"IT Services": 0.20, "Retail": 0.32, "Logistics": 0.38,
                "Manufacturing": 0.48, "F&B": 0.55, "Construction": 0.65}
cal_df["expert_risk"] = cal_df["sector"].map(EXPERT_ORDER)
cal_df["blended_risk_score"] = (0.40 * cal_df["rbi_risk_score"] + 0.60 * cal_df["expert_risk"]).round(3)

print(f"\n=== Calibrated Risk Scores (blended: 40% RBI + 60% expert) ===")
print(cal_df[["sector","outstanding_cr","yoy_growth_pct","blended_risk_score"]].sort_values("blended_risk_score").to_string(index=False))

# Build final calibration dict
final = {}
for _, row in cal_df.iterrows():
    final[row["sector"]] = {
        "blended_risk_score": float(row["blended_risk_score"]),
        "rbi_credit_jan26_cr": row["rbi_credit_jan26_cr"],
        "rbi_yoy_growth_pct": row["rbi_yoy_growth_pct"],
        "rbi_matched_categories": row["rbi_matched_categories"],
        "source": "RBI Sectoral Deployment Jan 2026"
    }

with open("rbi_data/rbi_sector_calibration.json", "w") as f:
    json.dump(final, f, indent=2)

with open("rbi_data/rbi_msme_macro.json", "w") as f:
    json.dump(msme_macro, f, indent=2)

print("\n✅ Parser complete.")
print("  rbi_data/rbi_sector_calibration.json — sector risk scores (RBI-calibrated)")
print("  rbi_data/rbi_msme_macro.json          — MSME macro stats (₹Cr, YoY%)")
print("  rbi_data/rbi_industry_detail.csv       — sub-industry level data (clean)")
