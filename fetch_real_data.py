import requests
import pandas as pd
import time
from io import StringIO
import os

# ---- CONFIG ----
API_KEY = "579b464db66ec23bdd00000146c81fc7acda4df3416d2d30b80b4946"
RESOURCE_ID = "603037492"  # MCA Company Master Data
BASE_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
DBIE_SDMX = "https://data.rbi.org.in/DBIE/api/v1/data"

TARGET_STATES = ["maharashtra"] # Test with one state

# Ensure data directory exists
os.makedirs("data/real", exist_ok=True)

# ---- MCA FETCH ----
def fetch_mca_sme_data(states: list, max_records_per_state: int = 50) -> pd.DataFrame:
    all_records = []
    for state in states:
        print(f"Fetching MCA: {state}")
        offset = 0
        while offset < max_records_per_state:
            params = {
                "api-key": API_KEY,
                "format": "json",
                "limit": 50,
                "offset": offset,
                "filters[COMPANY_STATUS]": "Active",
                "filters[REGISTERED_STATE]": state.lower(),
            }
            retries = 3
            success = False
            while retries > 0 and not success:
                try:
                    resp = requests.get(BASE_URL, params=params, timeout=60)
                    resp.raise_for_status()
                    data = resp.json()
                    records = data.get("records", [])
                    if not records:
                        success = True
                        break
                    all_records.extend(records)
                    offset += 50
                    success = True
                    print(f"  Fetched {len(records)} records for {state}")
                    time.sleep(2.0)
                except Exception as e:
                    print(f"  Retry {4-retries} for {state}: {e}")
                    retries -= 1
                    time.sleep(5)
            if not success: break
    df = pd.DataFrame(all_records)
    return df

# ---- RBI FETCH (SDMX) ----
def fetch_rbi_series(series_id: str, start: str = "2020-Q1", end: str = "2025-Q4") -> pd.DataFrame:
    print(f"Fetching RBI Series: {series_id}")
    url = f"{DBIE_SDMX}/{series_id}?startPeriod={start}&endPeriod={end}&format=csv"
    try:
        resp = requests.get(url, timeout=30)
        df = pd.read_csv(StringIO(resp.text))
        return df
    except Exception as e:
        print(f"  Error fetching RBI series {series_id}: {e}")
        return pd.DataFrame()

# ---- EXECUTION ----
if __name__ == "__main__":
    print("--- STARTING REAL DATA ACQUISITION ---")
    
    # 1. MCA Data
    raw_mca = fetch_mca_sme_data(states=TARGET_STATES, max_records_per_state=200)
    if not raw_mca.empty:
        raw_mca.to_csv("data/real/mca_sme_raw.csv", index=False)
        print(f"Saved {len(raw_mca):,} raw MCA records")
    else:
        print("MCA fetch failed or returned empty")

    # 2. RBI Data
    msme_credit = fetch_rbi_series("IN.A.BSAD1.Q21")
    if not msme_credit.empty:
        msme_credit.to_csv("data/real/rbi_msme_credit.csv", index=False)
        
    sectoral = fetch_rbi_series("IN.Q.SDBC.MSME")
    if not sectoral.empty:
        sectoral.to_csv("data/real/rbi_sectoral_credit.csv", index=False)
        
    npa = fetch_rbi_series("IN.A.BSAD.NPA.RATIO")
    if not npa.empty:
        npa.to_csv("data/real/rbi_npa_ratio.csv", index=False)

    print("--- DATA ACQUISITION COMPLETE ---")
