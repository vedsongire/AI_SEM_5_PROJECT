import sys
import math
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.ui.app import app

def test_api_optimize_float_strip():
    client = app.test_client()

    # Test payload
    payload = {
        "location": "Jalna, Maharashtra",
        "crop": "Onion",
        "quantity": 80.0
    }

    response = client.post("/api/optimize", json=payload)
    print(f"Status Code: {response.status_code}")
    data = response.get_json()
    print(f"Response Status: {data.get('status')}")
    if data.get("status") == "success":
        print(f"Hero Winner: {data.get('hero_winner', {}).get('market_name')}")
        print("PASS: /api/optimize executed successfully with float-safe strip!")
    else:
        print(f"Error Message: {data.get('message')}")
        assert False, f"Failed: {data.get('message')}"

if __name__ == "__main__":
    test_api_optimize_float_strip()
