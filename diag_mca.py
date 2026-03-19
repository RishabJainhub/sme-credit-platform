import requests
import json

API_KEY = "579b464db66ec23bdd00000146c81fc7acda4df3416d2d30b80b4946"
RESOURCE_ID = "603037492"
url = f"https://api.data.gov.in/resource/{RESOURCE_ID}?api-key={API_KEY}&format=json&limit=10"

print(f"Requesting: {url}")
try:
    resp = requests.get(url, timeout=30)
    print(f"Status Code: {resp.status_code}")
    data = resp.json()
    print("Columns found:", [f['id'] for f in data.get('field', [])])
    print("Number of records:", len(data.get('records', [])))
    if data.get('records'):
        print("Sample record:", data['records'][0])
except Exception as e:
    print(f"Error: {e}")
