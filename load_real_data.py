import argparse
import hashlib
import json
import os
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests


DEFAULT_STATES = [
    "Andaman And Nicobar Islands",
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chandigarh",
    "Chhattisgarh",
    "Delhi",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jammu And Kashmir",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Ladakh",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Puducherry",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
]

# Keep the user-provided ID first, then a verified fallback discovered from the live catalog page.
RESOURCE_ID_CANDIDATES = [
    "603037492",
    "ec58dab7-d891-4abb-936e-d5d274a6ce9b",
]

MCA_BASE = "https://api.data.gov.in/resource/{resource_id}"
RBI_SDMX = "https://data.rbi.org.in/DBIE/api/v1/data"

RBI_SERIES = {
    "rbi_msme_credit.csv": "IN.A.BSAD1.Q21",
    "rbi_sectoral_credit.csv": "IN.Q.SDBC.MSME",
    "rbi_npa_ratio.csv": "IN.A.BSAD.NPA.RATIO",
}

INDUSTRY_PATTERNS = {
    "IT Services": [
        "TECH",
        "SOFTWARE",
        "INFOTECH",
        "DIGITAL",
        "SYSTEM",
        "CYBER",
        "DATA",
        "CLOUD",
        "SOLUTION",
    ],
    "Logistics": [
        "LOGISTICS",
        "TRANSPORT",
        "CARGO",
        "FREIGHT",
        "SUPPLY",
        "WAREHOUSE",
        "COURIER",
        "SHIPPING",
    ],
    "Construction": [
        "CONSTRUCTION",
        "BUILDERS",
        "INFRA",
        "REALTY",
        "DEVELOPER",
        "PROJECT",
        "CIVIL",
        "ENGINEERING",
    ],
    "Manufacturing": [
        "MANUFACTUR",
        "INDUSTR",
        "FABRICAT",
        "CHEM",
        "PHARMA",
        "TEXTILE",
        "MILLS",
        "STEEL",
        "METAL",
        "PLASTIC",
    ],
    "F&B": [
        "FOOD",
        "BEVERAGE",
        "DAIRY",
        "RESTAURANT",
        "HOTEL",
        "CAFE",
        "AGRO",
        "KITCHEN",
        "NUTRITION",
    ],
    "Retail": [
        "TRADERS",
        "TRADING",
        "RETAIL",
        "MART",
        "STORE",
        "BAZAAR",
        "WHOLESALE",
        "COMMERCE",
    ],
}

SECTOR_RISK_FALLBACK = {
    "IT Services": 0.20,
    "Retail": 0.32,
    "Logistics": 0.38,
    "Manufacturing": 0.48,
    "F&B": 0.55,
    "Construction": 0.65,
}

STATE_NPA_PROXY = {
    "Delhi": 0.04,
    "Karnataka": 0.05,
    "Maharashtra": 0.05,
    "Tamil Nadu": 0.06,
    "Telangana": 0.06,
    "Gujarat": 0.06,
    "West Bengal": 0.08,
    "Rajasthan": 0.08,
    "Uttar Pradesh": 0.09,
}

METRO_STATES = {"Delhi", "Maharashtra", "Karnataka", "Tamil Nadu", "Telangana"}


def stable_int(key: str, mod: int) -> int:
    digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % mod


def resolve_resource_id(api_key: str) -> str:
    for rid in RESOURCE_ID_CANDIDATES:
        url = MCA_BASE.format(resource_id=rid)
        params = {"api-key": api_key, "format": "json", "limit": 1}
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            continue

        if payload.get("status") == "ok" and payload.get("message") != "Meta not found":
            print(f"Using MCA resource id: {rid} ({payload.get('title', 'unknown')})")
            return rid

    raise RuntimeError("Could not resolve a working MCA resource id.")


def fetch_mca(api_key: str, states: list[str], max_records_per_state: int) -> pd.DataFrame:
    resource_id = resolve_resource_id(api_key)
    url = MCA_BASE.format(resource_id=resource_id)

    all_records = []
    for state in states:
        offset = 0
        print(f"Fetching MCA state={state}")
        while offset < max_records_per_state:
            params = {
                "api-key": api_key,
                "format": "json",
                "limit": 500,
                "offset": offset,
                "filters[company_status]": "Active",
                "filters[registered_state]": state,
            }

            payload = None
            for attempt in range(1, 4):
                try:
                    resp = requests.get(url, params=params, timeout=45)
                    resp.raise_for_status()
                    payload = resp.json()
                    break
                except Exception as exc:
                    if attempt == 3:
                        print(f"  Failed state={state} offset={offset}: {exc}")
                    time.sleep(1.0 * attempt)

            if payload is None:
                break

            records = payload.get("records", [])
            if not records:
                break

            all_records.extend(records)
            offset += len(records)
            print(f"  +{len(records)} rows (offset={offset})")
            time.sleep(0.25)

    df = pd.DataFrame(all_records)
    if df.empty:
        raise RuntimeError("MCA API returned no records for the requested states.")

    return df


def fetch_rbi_sdmx(start_period: str, end_period: str, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    status = {}

    for filename, series_id in RBI_SERIES.items():
        url = f"{RBI_SDMX}/{series_id}?startPeriod={start_period}&endPeriod={end_period}&format=csv"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            text = resp.text

            if "<html" in text.lower() or "page not found" in text.lower():
                status[filename] = "failed_html_response"
                continue

            df = pd.read_csv(StringIO(text))
            if df.empty:
                status[filename] = "failed_empty"
                continue

            df.to_csv(out_dir / filename, index=False)
            status[filename] = f"ok_rows_{len(df)}"
        except Exception as exc:
            status[filename] = f"failed_{type(exc).__name__}"

    with open(out_dir / "rbi_sdmx_status.json", "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    return status


def classify_industry(company_name: str) -> str:
    name = str(company_name).upper()
    for sector, patterns in INDUSTRY_PATTERNS.items():
        if any(p in name for p in patterns):
            return sector
    return "Retail"


def derive_director_count(cin: str, company_class: str) -> int:
    base = 2
    cls = str(company_class).strip().lower()
    if cls == "public":
        base = 3
    elif cls == "private":
        base = 2
    else:
        base = 1

    bump = stable_int(cin, 2)
    return int(np.clip(base + bump, 1, 4))


def derive_credit_score(row: pd.Series) -> int:
    score = 730

    age = float(row["AGE_YEARS"])
    if age < 2:
        score -= 120
    elif age < 5:
        score -= 70

    tier = row["CAPITAL_TIER"]
    if tier == "Micro":
        score -= 55
    elif tier == "Small":
        score -= 20

    if row["STATE_NPA_PROXY"] > 0.08:
        score -= 45

    score -= int(float(row["SECTOR_RISK_SCORE"]) * 110)

    if row["DIRECTOR_COUNT"] >= 3:
        score += 20
    if age > 10:
        score += 35

    score += stable_int(row["CIN"], 51) - 25
    return int(np.clip(score, 300, 850))


def build_dashboard_dataset(mca_raw: pd.DataFrame, rbi_sector_path: Path) -> pd.DataFrame:
    df = mca_raw.copy()

    df["CIN"] = df["corporate_identification_number"].astype(str)
    df["COMPANY_NAME"] = df["company_name"].astype(str).str.strip()
    df["STATE"] = df["registered_state"].astype(str).str.strip().str.title()

    df["PAID_UP_CAPITAL"] = pd.to_numeric(df["paidup_capital"], errors="coerce")
    auth_cap = pd.to_numeric(df["authorized_capital"], errors="coerce")
    df["AUTHORIZED_CAP_INR"] = auth_cap
    df["AUTHORIZED_CAP_INR"] = df["AUTHORIZED_CAP_INR"].where(
        df["AUTHORIZED_CAP_INR"].notna(),
        df["PAID_UP_CAPITAL"] * 1.25,
    )

    reg_date = pd.to_datetime(df["date_of_registration"], errors="coerce", utc=True)
    now = pd.Timestamp.utcnow()
    df["AGE_YEARS"] = ((now - reg_date).dt.days / 365.25).clip(lower=0)

    # Keep only usable SME rows.
    df = df[(df["PAID_UP_CAPITAL"].notna()) & (df["PAID_UP_CAPITAL"] > 0)]
    df = df[df["PAID_UP_CAPITAL"] <= 100_000_000].copy()

    df["CAPITAL_TIER"] = pd.cut(
        df["PAID_UP_CAPITAL"],
        bins=[0, 1_000_000, 20_000_000, 100_000_000],
        labels=["Micro", "Small", "Medium"],
        include_lowest=True,
    ).astype(str)

    df["AGE_BUCKET"] = pd.cut(
        df["AGE_YEARS"],
        bins=[0, 2, 5, 10, 100],
        labels=["0-2 yrs", "2-5 yrs", "5-10 yrs", "10+ yrs"],
        include_lowest=True,
    ).astype(str)

    df["INDUSTRY"] = df["COMPANY_NAME"].map(classify_industry)

    if rbi_sector_path.exists():
        with open(rbi_sector_path, "r", encoding="utf-8") as f:
            rbi_sector = json.load(f)
        sector_risk = {
            k: float(v.get("blended_risk_score", SECTOR_RISK_FALLBACK.get(k, 0.4)))
            for k, v in rbi_sector.items()
        }
    else:
        sector_risk = SECTOR_RISK_FALLBACK.copy()

    for sec, val in SECTOR_RISK_FALLBACK.items():
        sector_risk.setdefault(sec, val)

    df["SECTOR_RISK_SCORE"] = df["INDUSTRY"].map(sector_risk).fillna(0.40)
    df["STATE_NPA_PROXY"] = df["STATE"].map(STATE_NPA_PROXY).fillna(0.08)

    gap_map = {"Micro": 0.65, "Small": 0.45, "Medium": 0.25}
    df["CREDIT_GAP"] = df["CAPITAL_TIER"].map(gap_map).fillna(0.40)

    df["DIRECTOR_COUNT"] = [
        derive_director_count(cin, cclass)
        for cin, cclass in zip(df["CIN"], df["company_class"], strict=False)
    ]
    df["IS_METRO"] = df["STATE"].isin(METRO_STATES).astype(int)

    # User's default proxy (with explicit state NPA and credit gap proxies).
    default_score = (
        np.where(df["PAID_UP_CAPITAL"] < 1_000_000, 2, 0)
        + np.where(df["AGE_YEARS"] < 2, 2, 0)
        + np.where(df["STATE_NPA_PROXY"] > 0.08, 1, 0)
        + np.where(df["CREDIT_GAP"] > 0.6, 1, 0)
    )
    df["DEFAULT_RISK"] = (default_score >= 4).astype(int)

    df["CREDIT_SCORE"] = df.apply(derive_credit_score, axis=1)

    df["IS_OPPORTUNITY"] = (
        (df["CREDIT_SCORE"] >= 680)
        & (df["PAID_UP_CAPITAL"] <= 5_000_000)
        & (df["SECTOR_RISK_SCORE"] <= 0.50)
    ).astype(int)

    out_cols = [
        "COMPANY_NAME",
        "STATE",
        "INDUSTRY",
        "PAID_UP_CAPITAL",
        "AUTHORIZED_CAP_INR",
        "AGE_YEARS",
        "AGE_BUCKET",
        "CAPITAL_TIER",
        "DIRECTOR_COUNT",
        "STATE_NPA_PROXY",
        "CREDIT_GAP",
        "SECTOR_RISK_SCORE",
        "CREDIT_SCORE",
        "DEFAULT_RISK",
        "IS_METRO",
        "IS_OPPORTUNITY",
    ]

    final = df[out_cols].copy()
    final = final.dropna(subset=["COMPANY_NAME", "STATE", "PAID_UP_CAPITAL"]).reset_index(drop=True)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch MCA + (best-effort) RBI and build dashboard dataset.")
    parser.add_argument("--api-key", default=os.getenv("DATAGOV_API_KEY"))
    parser.add_argument("--max-records-per-state", type=int, default=5000)
    parser.add_argument("--start-period", default="2020-Q1")
    parser.add_argument("--end-period", default="2025-Q4")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Pass --api-key or set DATAGOV_API_KEY.")

    repo = Path(__file__).resolve().parent
    data_real_dir = repo / "data" / "real"
    data_real_dir.mkdir(parents=True, exist_ok=True)

    mca_raw = fetch_mca(
        api_key=args.api_key,
        states=DEFAULT_STATES,
        max_records_per_state=args.max_records_per_state,
    )
    mca_raw_path = data_real_dir / "mca_sme_raw.csv"
    mca_raw.to_csv(mca_raw_path, index=False)
    print(f"Saved raw MCA rows: {len(mca_raw):,} -> {mca_raw_path}")

    rbi_status = fetch_rbi_sdmx(
        start_period=args.start_period,
        end_period=args.end_period,
        out_dir=data_real_dir,
    )
    print("RBI SDMX status:", json.dumps(rbi_status, indent=2))

    final = build_dashboard_dataset(
        mca_raw=mca_raw,
        rbi_sector_path=repo / "rbi_data" / "rbi_sector_calibration.json",
    )

    final_path = repo / "data" / "sme_clean_real.csv"
    final.to_csv(final_path, index=False)

    print("\nFinal dataset summary")
    print(f"Rows: {len(final):,}")
    print(f"States: {final['STATE'].nunique()}")
    print(f"Default rate: {final['DEFAULT_RISK'].mean():.2%}")
    print(f"Opportunity count: {final['IS_OPPORTUNITY'].sum():,}")
    print(f"Saved dashboard dataset -> {final_path}")


if __name__ == "__main__":
    main()
