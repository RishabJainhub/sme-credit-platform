"""
fetch_rbi_data.py
India SME Credit Risk Platform — Real RBI Data Fetcher
=======================================================
Downloads 3 RBI Excel files:
  1. Sectoral Deployment of Bank Credit (Jan 2026)
  2. Bulletin Table 15 — Major Sectors credit time series
  3. Bulletin Table 16 — Industry-wise breakdown

Then parses and saves clean CSVs ready for use in the Streamlit app.
"""

import requests
import pandas as pd
from io import BytesIO
import os, warnings
warnings.filterwarnings("ignore")

urls = {
    "sectoral_deployment_jan2026": "https://rbidocs.rbi.org.in/rdocs/content/docs/SIBC27022026.xlsx",
    "bulletin_table15_major_sectors": "https://rbidocs.rbi.org.in/rdocs/Bulletin/DOCs/15T_BUL200220261B212CA1AFC54B2AAEA3C2638E3215DB.XLSX",
    "bulletin_table16_industry_wise": "https://rbidocs.rbi.org.in/rdocs/Bulletin/DOCs/16T_BUL20022026EEA7CC7549F5485F81A56F4AA55AB47D.XLSX",
}

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://rbi.org.in",
    "Accept": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,*/*"
}

os.makedirs("rbi_data", exist_ok=True)

raw_sheets = {}

for name, url in urls.items():
    print(f"\nDownloading: {name}...")
    try:
        r = requests.get(url, headers=headers, timeout=30, verify=False)
        r.raise_for_status()
        xl = pd.ExcelFile(BytesIO(r.content))
        print(f"  Sheets: {xl.sheet_names}  |  File size: {len(r.content)/1024:.0f} KB")
        raw_sheets[name] = {}
        for sheet in xl.sheet_names:
            df = xl.parse(sheet, header=None)
            safe_sheet = sheet.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
            csv_path = f"rbi_data/{name}_{safe_sheet}.csv"
            df.to_csv(csv_path, index=False)
            raw_sheets[name][sheet] = df
            print(f"  Saved: {csv_path}  ({df.shape[0]} rows × {df.shape[1]} cols)")
        # Preview
        first = xl.sheet_names[0]
        print(f"\n  Preview (first 18 rows of '{first}'):")
        print(raw_sheets[name][first].head(18).to_string())
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n\nDone. All raw CSVs in ./rbi_data/")
print("Now running parser to extract clean sector NPA calibration data...\n")

# ─────────────────────────────────────────────────────────────
# PARSER — extract what we actually need
# ─────────────────────────────────────────────────────────────
# From Sectoral Deployment Statement I we need:
#   - Micro & Small credit outstanding (₹ Cr)
#   - % share and YoY growth by major sector
# This gives us REAL weights to calibrate sector risk scores.

def extract_sectoral(raw_sheets):
    key = "sectoral_deployment_jan2026"
    if key not in raw_sheets:
        print("  Sectoral deployment file not downloaded — skipping parser.")
        return None
    df = raw_sheets[key].get("Statement I") or list(raw_sheets[key].values())[0]
    # Find the row where real data starts (look for row containing "Agriculture")
    start_row = None
    for i, row in df.iterrows():
        row_str = " ".join(str(v) for v in row.values if pd.notna(v))
        if "Agriculture" in row_str or "Micro & Small" in row_str or "Industry" in row_str:
            start_row = max(0, i - 2)
            break
    if start_row is None:
        print("  Could not auto-detect data rows — saving raw only.")
        return df
    clean = df.iloc[start_row:].reset_index(drop=True)
    clean.to_csv("rbi_data/PARSED_sectoral_deployment.csv", index=False)
    print(f"  Parsed sectoral data saved: rbi_data/PARSED_sectoral_deployment.csv")
    print(f"  {len(clean)} rows extracted from row {start_row}\n")
    print(clean.head(20).to_string())
    return clean

parsed = extract_sectoral(raw_sheets)
