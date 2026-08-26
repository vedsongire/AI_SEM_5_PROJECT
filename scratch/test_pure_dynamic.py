import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agents.planner import PlannerAgent
from src.agents.scout import ScoutAgent

print("Initializing Agents...")
planner = PlannerAgent()
scout = ScoutAgent()

print("\n--- 1. Testing Live Geocoding for 'Muzaffarnagar, Uttar Pradesh' ---")
loc = planner.geocode_farmer_location("Muzaffarnagar, Uttar Pradesh")
print("Resolved Location Payload:", loc)

print("\n--- 2. Fetching Mandi Records for UP - Wheat ---")
records = scout.fetch_live_mandi_prices(state=loc["state"], commodity="Wheat")
print(f"Scout returned {len(records)} records:")
for r in records:
    print(" -", r.get("market_name"), "| District:", r.get("district"), "| Modal Price:", r.get("modal_price"))

print("\n--- 3. Discovering Candidate Markets Dynamically ---")
candidates = planner.discover_candidate_markets(
    farmer_lat=loc["latitude"],
    farmer_lon=loc["longitude"],
    state=loc["state"],
    active_mandi_records=records,
    commodity="Wheat",
)
print("Discovered Candidates:")
for c in candidates:
    print(" -", c["role_label"], ":", c["market_name"], f"({c['haversine_distance_km']} km)")
