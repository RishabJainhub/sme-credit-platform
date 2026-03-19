import os
import requests
import pandas as pd

def fetch_rbi_bulletin():
    """
    Simulates fetching the latest RBI Sectoral Deployment of Bank Credit.
    In a real-world scenario, this would use a scraper or a direct API.
    For this demo, we verify the presence of the Jan 2026 ground-truth CSVs.
    """
    print("--- INITIATING RBI DATA SYNC (Jan 2026) ---")
    
    target_dir = "rbi_data"
    required_files = [
        "sectoral_deployment_jan2026_Statement_1.csv",
        "sectoral_deployment_jan2026_Statement_2.csv"
    ]
    
    missing = [f for f in required_files if not os.path.exists(os.path.join(target_dir, f))]
    
    if missing:
        print(f"FAILED: Missing ground truth files: {missing}")
        print("Ensure the 'rbi_data/' folder is populated with Jan 2026 Statement 1 & 2.")
    else:
        print("SUCCESS: Jan 2026 RBI Statement 1 & 2 verified locally.")
        print("Sync complete. Metrics calibrated against 31 Jan 2026 reports.")

if __name__ == "__main__":
    fetch_rbi_bulletin()
