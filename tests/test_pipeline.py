"""Integration test suite for Kisan AI real multi-agent pipeline."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agents.planner import PlannerAgent
from src.agents.predictor import PredictorAgent
from src.agents.scout import ScoutAgent


def test_real_multi_agent_pipeline_up():
    """Verify that a farmer in Muzaffarnagar, Uttar Pradesh receives local UP APMCs (< 300 km), not Maharashtra."""
    scout = ScoutAgent()
    predictor = PredictorAgent()
    planner = PlannerAgent()

    # 1. Geocode farmer location query dynamically via live API
    loc_query = "Muzaffarnagar, Uttar Pradesh"
    loc_info = planner.geocode_farmer_location(loc_query)

    assert "Uttar Pradesh" in loc_info["state"] or loc_info["state"] == "Uttar Pradesh"
    assert loc_info["district"] == "Muzaffarnagar"
    assert loc_info["latitude"] > 0
    assert loc_info["longitude"] > 0

    farmer_lat = loc_info["latitude"]
    farmer_lon = loc_info["longitude"]
    state = loc_info["state"]
    commodity = "Wheat"
    quantity_quintals = 50.0

    # 2. Scout weather & mandi prices for UP
    weather = scout.fetch_weather(latitude=farmer_lat, longitude=farmer_lon)
    assert "temperature" in weather

    mandi_records = scout.fetch_live_mandi_prices(state=state, commodity=commodity)
    assert len(mandi_records) > 0

    ndvi_info = predictor.generate_ndvi_index(state=state, commodity=commodity)
    assert "ndvi" in ndvi_info

    # 3. Discover Candidate Markets dynamically for UP
    candidate_markets = planner.discover_candidate_markets(
        farmer_lat=farmer_lat,
        farmer_lon=farmer_lon,
        state=state,
        active_mandi_records=mandi_records,
        commodity=commodity,
    )
    assert len(candidate_markets) >= 1

    # CRITICAL VERIFICATION: Ensure nearest candidate market is in UP and < 300 km away
    nearest = candidate_markets[0]
    assert nearest["haversine_distance_km"] < 300.0, f"Distance too far: {nearest['haversine_distance_km']} km"
    assert "Uttar Pradesh" in nearest["state"] or nearest["state"] == "Uttar Pradesh", f"Wrong state: {nearest['state']}"

    # 4. Generate Predictions & Optimize Logistics
    market_predictions = {}
    for cm in candidate_markets:
        m_name = cm["market_name"]
        mandi_rec = {
            "state": state,
            "district": cm["district"],
            "market_name": m_name,
            "commodity": commodity,
            "variety": "Local",
            "arrival_date": "22/08/2026",
            "modal_price": 2400,
        }
        preds = predictor.predict_quantile_prices(
            mandi_record=mandi_rec, weather_data=weather, ndvi_data=ndvi_info
        )
        market_predictions[m_name] = preds

    recommendations = planner.optimize_logistics(
        farmer_lat=farmer_lat,
        farmer_lon=farmer_lon,
        predictions_by_market=market_predictions,
        quantity_quintals=quantity_quintals,
        state=state,
        target_markets=candidate_markets,
    )

    assert len(recommendations) >= 1
    winner = recommendations[0]
    assert winner["one_way_distance_km"] < 300.0, f"Winner distance too far: {winner['one_way_distance_km']} km"
    print(f"\n[OK] UP Pipeline Test Passed! Winner: {winner['market_name']} ({winner['one_way_distance_km']} km, Profit: Rs. {winner['net_expected_profit_inr']:,.2f})")


def test_real_multi_agent_pipeline_haryana():
    """Verify that a farmer in Karnal, Haryana receives local Haryana APMCs."""
    scout = ScoutAgent()
    predictor = PredictorAgent()
    planner = PlannerAgent()

    loc_info = planner.geocode_farmer_location("Karnal, Haryana")
    assert "Haryana" in loc_info["state"] or loc_info["state"] == "Haryana"

    candidates = planner.discover_candidate_markets(
        farmer_lat=loc_info["latitude"],
        farmer_lon=loc_info["longitude"],
        state=loc_info["state"],
        commodity="Wheat",
    )
    assert len(candidates) >= 1
    assert candidates[0]["haversine_distance_km"] < 300.0
    print(f"[OK] Haryana Pipeline Test Passed! Nearest: {candidates[0]['market_name']} ({candidates[0]['haversine_distance_km']} km)")


def test_real_multi_agent_pipeline_indore_mp():
    """Verify backend multi-agent execution for Indore, MP farmer selling 60 quintals Wheat."""
    scout = ScoutAgent()
    predictor = PredictorAgent()
    planner = PlannerAgent()

    loc_query = "Indore, Madhya Pradesh"
    commodity = "Wheat"
    quantity_quintals = 60.0

    print(f"\n--- Running Backend Pipeline for {loc_query} ({quantity_quintals} qt {commodity}) ---")
    loc_info = planner.geocode_farmer_location(loc_query)
    lat, lon = loc_info["latitude"], loc_info["longitude"]
    state = loc_info["state"]

    weather = scout.fetch_weather(latitude=lat, longitude=lon)
    mandi_records = scout.fetch_live_mandi_prices(state=state, commodity=commodity)
    candidate_markets = planner.discover_candidate_markets(
        farmer_lat=lat, farmer_lon=lon, state=state, active_mandi_records=mandi_records, commodity=commodity
    )

    market_predictions = {}
    for cm in candidate_markets:
        m_name = cm["market_name"]
        mandi_rec = {
            "state": state,
            "district": cm.get("district", state),
            "market_name": m_name,
            "commodity": commodity,
            "variety": "Local",
            "arrival_date": "26/08/2026",
            "modal_price": cm.get("modal_price", 2400),
        }
        preds = predictor.predict_quantile_prices(
            mandi_record=mandi_rec, weather_data=weather, ndvi_data={"ndvi": 0.65}
        )
        market_predictions[m_name] = preds

    recs = planner.optimize_logistics(
        farmer_lat=lat,
        farmer_lon=lon,
        predictions_by_market=market_predictions,
        quantity_quintals=quantity_quintals,
        state=state,
        target_markets=candidate_markets,
    )

    assert len(recs) >= 1
    winner = recs[0]
    print(f"[OK] Indore MP Pipeline Test Passed!")
    print(f"     [WINNER] Best Mandi: {winner['market_name']} ({winner['one_way_distance_km']:.1f} km)")
    print(f"     [VEHICLE] Assigned: {winner['vehicle_assigned']}")
    print(f"     [PROFIT] Net Expected Profit: Rs. {winner['net_expected_profit_inr']:,.2f}")


if __name__ == "__main__":
    test_real_multi_agent_pipeline_indore_mp()
    test_real_multi_agent_pipeline_up()
    test_real_multi_agent_pipeline_haryana()

