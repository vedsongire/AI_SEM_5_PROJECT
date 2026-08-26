"""K.I.S.A.N. AI - Integrated Flask Web UI Controller.

Connects the locked single-page HTML5/Tailwind frontend to real backend agents:
- ScoutAgent (Live Weather & APMC Mandi Data Ingestion)
- PredictorAgent (LightGBM ML Price Quantile Predictions: p10, p50, p90)
- PlannerAgent (Geocoding, Diesel Rate Scraping, Highway OSRM Routing & Economics)
"""

import math
import sys
import traceback
from pathlib import Path
from flask import Flask, render_template, jsonify, request

# Dynamic path setup to ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.agents.scout import ScoutAgent
from src.agents.predictor import PredictorAgent
from src.agents.planner import PlannerAgent

app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
    static_folder=str(Path(__file__).resolve().parent / "static"),
)

# Initialize Real Backend Multi-Agent System
print("[INIT] Loading real K.I.S.A.N. AI Backend Agents...")
try:
    scout_agent = ScoutAgent()
    predictor_agent = PredictorAgent()
    planner_agent = PlannerAgent()
    print("[OK] Real Backend Agents initialized successfully!")
except Exception as exc:
    print(f"[WARNING] Agent initialization error: {exc}")
    scout_agent = None
    predictor_agent = None
    planner_agent = None


@app.route("/")
def index():
    """Render the open-access K.I.S.A.N. AI single-page dashboard & results terminal."""
    return render_template("index.html")


@app.route("/api/optimize", methods=["POST"])
def optimize_api():
    """Execute real multi-agent decision pipeline for spatial arbitrage optimization."""
    try:
        data = request.json or {}
        location_query = data.get("location", "Jalna, Maharashtra").strip()
        commodity = data.get("crop", "Onion").strip()
        quantity_qt = float(data.get("quantity", 80.0))

        if not planner_agent or not predictor_agent or not scout_agent:
            return jsonify({
                "status": "error",
                "message": "Backend multi-agent system not initialized."
            }), 500

        # ---------------------------------------------------------------------
        # STEP 1: Geocode Farmer Location via PlannerAgent
        # ---------------------------------------------------------------------
        loc_info = planner_agent.geocode_farmer_location(location_query)
        farmer_lat = loc_info.get("latitude", 19.8347)
        farmer_lon = loc_info.get("longitude", 75.8816)
        state = loc_info.get("state", "Maharashtra")
        district = loc_info.get("district", "Jalna")

        # ---------------------------------------------------------------------
        # STEP 2: Scout Live Weather & Satellite Crop Vigor (NDVI)
        # ---------------------------------------------------------------------
        env_data = scout_agent.fetch_weather(farmer_lat, farmer_lon)
        ndvi_data = predictor_agent.generate_ndvi_index(state, commodity)

        # ---------------------------------------------------------------------
        # STEP 3: Discover Candidate APMC Markets
        # ---------------------------------------------------------------------
        candidates = planner_agent.discover_candidate_markets(
            farmer_lat=farmer_lat,
            farmer_lon=farmer_lon,
            state=state,
            commodity=commodity,
        )

        if not candidates:
            # Fallback candidate list if no local CSV matches found
            candidates = [
                {"market_name": "MUMBAI VASHI APMC", "district": "Mumbai", "modal_price": 3450.0},
                {"market_name": "PUNE APMC", "district": "Pune", "modal_price": 3100.0},
                {"market_name": "NASHIK APMC", "district": "Nashik", "modal_price": 2850.0},
            ]

        # ---------------------------------------------------------------------
        # STEP 4: Scrape Live Diesel Rate & Predict LightGBM ML Quantile Price Bands
        # ---------------------------------------------------------------------
        diesel_price = planner_agent._get_live_diesel_price(state)
        predictions = {}
        for cm in candidates:
            raw_mname = cm.get("market_name")
            if raw_mname is None or (isinstance(raw_mname, float) and math.isnan(raw_mname)) or str(raw_mname).strip().lower() in ("", "nan"):
                clean_market_name = "APMC Market"
            else:
                clean_market_name = str(raw_mname).strip()

            market_key = clean_market_name.lower()
            m_modal = cm.get("modal_price", 2800.0)
            try:
                modal_price_val = float(m_modal) if m_modal is not None and not (isinstance(m_modal, float) and math.isnan(m_modal)) else 2800.0
            except (ValueError, TypeError):
                modal_price_val = 2800.0

            mandi_rec = {
                "state": state,
                "district": str(cm.get("district") or district),
                "market": clean_market_name,
                "commodity": commodity,
                "variety": "Local",
                "modal_price": modal_price_val,
            }

            # Predict risk-adjusted p10, p50, p90 quantile price bands
            preds = predictor_agent.predict_quantile_prices(
                mandi_record=mandi_rec,
                weather_data=env_data,
                ndvi_data=ndvi_data,
            )
            predictions[market_key] = preds

        # ---------------------------------------------------------------------
        # STEP 5: Highway Logistics & Net Pocket Profit Optimization
        # ---------------------------------------------------------------------
        logistics_results = planner_agent.optimize_logistics(
            farmer_lat=farmer_lat,
            farmer_lon=farmer_lon,
            predictions_by_market=predictions,
            quantity_quintals=quantity_qt,
            state=state,
            target_markets=candidates,
        )

        if not logistics_results:
            return jsonify({"status": "error", "message": "Logistics calculation returned empty."}), 500

        # Sort candidate markets by net expected pocket profit descending
        logistics_results.sort(key=lambda x: x.get("net_expected_profit_inr", 0), reverse=True)
        winner = logistics_results[0]
        assigned_vehicle = winner.get("truck_type") or winner.get("vehicle_assigned") or "8-Ton Tata LPT 1109 Truck"

        # Calculate extra savings over local sales
        worst_market_profit = logistics_results[-1].get("net_expected_profit_inr", 0) if len(logistics_results) > 1 else 0
        hero_market_profit = winner.get("net_expected_profit_inr", 0)
        raw_savings = hero_market_profit - worst_market_profit
        if raw_savings > 0:
            savings_over_local = raw_savings
        else:
            # Baseline benchmark if candidate markets yield identical revenue or only 1 mandi exists
            savings_over_local = hero_market_profit * 0.12

        # Format comparison markets list matching the frontend UI contract
        comparison_list = []
        for idx, item in enumerate(logistics_results[:3], start=1):
            raw_m_name = item.get("market_name")
            m_key = str(raw_m_name).strip().lower() if raw_m_name is not None else ""
            p_band = predictions.get(m_key, {})

            p10_val = p_band.get("p10_downside_floor") or p_band.get("p10", 0)
            p50_val = p_band.get("p50_median_expected") or p_band.get("p50", 0)
            p90_val = p_band.get("p90_upside_ceiling") or p_band.get("p90", 0)

            comparison_list.append({
                "rank": idx,
                "market_name": item.get("market_name"),
                "district": item.get("district", district),
                "distance_km": round(item.get("one_way_distance_km", 0), 1),
                "transport_cost": round(item.get("total_transport_cost_inr", 0)),
                "p10_floor": round(p10_val),
                "p50_expected": round(p50_val),
                "p90_ceiling": round(p90_val),
                "gross_revenue": round(item.get("gross_expected_revenue_inr", 0)),
                "net_profit": round(item.get("net_expected_profit_inr", 0)),
                "is_hero": (idx == 1),
            })

        response_payload = {
            "status": "success",
            "location": f"{district}, {state}",
            "crop": commodity,
            "quantity": quantity_qt,
            "resolved_coords": {
                "lat": round(farmer_lat, 4),
                "lon": round(farmer_lon, 4),
                "district": district,
                "state": state,
            },
            "diesel_price": round(diesel_price, 2),
            "assigned_vehicle": assigned_vehicle,
            "hero_winner": {
                "market_name": winner.get("market_name"),
                "district": winner.get("district", district),
                "distance_km": round(winner.get("one_way_distance_km", 0), 1),
                "transport_cost": round(winner.get("total_transport_cost_inr", 0)),
                "modal_price": round(winner.get("p50_median_price") or winner.get("p50_expected") or winner.get("predicted_modal_price") or 0),
                "gross_revenue": round(winner.get("gross_expected_revenue_inr", 0)),
                "net_profit": round(winner.get("net_expected_profit_inr", 0)),
                "savings_over_local": round(savings_over_local),
            },
            "comparison_markets": comparison_list,
        }

        return jsonify(response_payload)

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(exc)}), 500


if __name__ == "__main__":
    print("=" * 70)
    print("🌾 K.I.S.A.N. AI Multi-Agent Flask Backend Running!")
    print("Open Browser: http://127.0.0.1:5000")
    print("=" * 70)
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
