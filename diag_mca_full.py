import requests
import json

API_KEY = "579b464db66ec23bdd00000146c81fc7acda4df3416d2d30b80b4946"
RESOURCE_ID = "603037492"
url = f"https://api.data.gov.in/resource/{RESOURCE_ID}?api-key={API_KEY}&format=json"

try:
    resp = requests.get(url, timeout=30)
    print(f"Status Code: {resp.status_code}")
    data = resp.json()
    print("Metadata:", {k: v for k, v in data.items() if k != 'records' and k != 'field'})
    print("Total Records (count):", data.get('total', 'N/A'))
    print("Number of records in this response:", len(data.get('records', [])))
except Exception as e:
    print(f"Error: {e}")
