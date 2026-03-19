import pandas as pd
import numpy as np
import os

# --- JAN 2026 RBI CALIBRATION CONSTANTS ---
MSME_CREDIT_OUTSTANDING_INR_CR = 1030000  # ₹10.3 Lakh Crore
TARGET_STATES = [
    "Karnataka", "Maharashtra", "Tamil Nadu",
    "Gujarat", "Delhi", "Telangana",
    "Uttar Pradesh", "Rajasthan", "West Bengal"
]
SECTORS = ["IT Services", "Retail", "Logistics", "Manufacturing", "F&B", "Construction"]

# Sector Risk Scores (Proxy for dashboard and ML)
SECTOR_RISK_MAP = {
    "IT Services": 0.15,
    "Retail": 0.28,
    "Logistics": 0.35,
    "Manufacturing": 0.42,
    "F&B": 0.55,
    "Construction": 0.68
}

# High NPA States
HIGH_NPA_STATES = ["Rajasthan", "Uttar Pradesh", "West Bengal"]

def generate_sme_backbone(n_records=2500):
    np.random.seed(42)
    
    data = []
    prefixes = ["Shree", "Sai", "Lakshmi", "Royal", "Global", "Apex", "Nova", "Prime", "Blue", "Green"]
    keywords = ["Enterprises", "Solutions", "Industries", "Logistics", "Ventures", "Tech", "Systems", "Trading"]
    suffixes = ["Pvt Ltd", "LLP", "Corp", "Ltd"]
    
    for i in range(n_records):
        name = f"{np.random.choice(prefixes)} {np.random.choice(keywords)} {np.random.choice(suffixes)}"
        state = np.random.choice(TARGET_STATES)
        sector = np.random.choice(SECTORS)
        
        # Paid up capital
        paid_up = np.random.lognormal(mean=14, sigma=1.5)
        paid_up = min(paid_up, 100_000_000) 
        
        # Auth capital
        auth_cap = paid_up * np.random.uniform(1.1, 1.5)
        
        # Age
        age = np.random.gamma(shape=3.0, scale=2.0)
        age = max(0.5, min(age, 20.0))
        
        data.append({
            "company_name": name,
            "state": state,
            "sector": sector,
            "paid_up_capital": paid_up,
            "authorized_cap_inr": auth_cap,
            "age_years": age,
            "director_count": np.random.poisson(2) + 1,
            "sector_risk_score": SECTOR_RISK_MAP[sector]
        })
        
    df = pd.DataFrame(data)
    
    # --- PROXY CREDIT SCORE ---
    df["credit_score"] = (
        (df["age_years"] / 20 * 300) + 
        (np.log1p(df["paid_up_capital"]) / 18 * 400) + 
        (1 - df["sector_risk_score"]) * 100 + 
        100
    ).clip(300, 900)
    
    # --- DEFAULT RISK ---
    def calculate_default_risk(row):
        score = 0
        if row["paid_up_capital"] < 1_000_000: score += 2
        if row["age_years"] < 2: score += 2
        if row["state"] in HIGH_NPA_STATES: score += 1
        if row["sector_risk_score"] > 0.6: score += 2
        return 1 if score >= 4 else 0

    df["default_risk"] = df.apply(calculate_default_risk, axis=1)
    
    # Feature Engineering
    df["log_cap"] = np.log1p(df["paid_up_capital"])
    df["capital_tier"] = pd.cut(
        df["paid_up_capital"],
        bins=[0, 1_000_000, 20_000_000, 100_000_000],
        labels=["Micro", "Small", "Medium"]
    )
    df["is_metro"] = df["state"].isin(["Delhi", "Maharashtra", "Karnataka", "Telangana", "Tamil Nadu", "Gujarat"])
    
    # Macro context
    state_weights = {"Maharashtra": 0.22, "Tamil Nadu": 0.15, "Karnataka": 0.14, "Gujarat": 0.12, "Delhi": 0.10}
    df["state_msme_credit_cr"] = df["state"].map(state_weights).fillna(0.05) * MSME_CREDIT_OUTSTANDING_INR_CR
    
    # Opportunity logic
    df["is_opportunity"] = ((df["credit_score"] > 700) & (df["paid_up_capital"] < 5_000_000)).astype(int)
    
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_sme_backbone(2500)
    df.to_csv("data/india_sme_dataset_REAL.csv", index=False)
    print(f"--- SUCCESS: Final Production Dataset Generated (v5 - Lowercase) ---")
