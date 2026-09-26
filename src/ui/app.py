"""K.I.S.A.N. AI - Integrated Flask Web UI Controller.

Connects the locked single-page HTML5/Tailwind frontend to real backend agents:
- ScoutAgent (Live Weather & APMC Mandi Data Ingestion)
- PredictorAgent (LightGBM ML Price Quantile Predictions: p10, p50, p90)
- PlannerAgent (Geocoding, Diesel Rate Scraping, Highway OSRM Routing & Economics)
"""

import base64
from datetime import datetime
import math
import sys
import traceback
from pathlib import Path
from flask import Flask, render_template, jsonify, request, Response

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
try:
    from src.agents.voice import VoiceAgent  # type: ignore
except ImportError:
    VoiceAgent = None


app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
    static_folder=str(Path(__file__).resolve().parent / "static"),
)

# Initialize Real Backend Multi-Agent System & Communication Services
print("[INIT] Loading real K.I.S.A.N. AI Backend Agents...")
try:
    scout_agent = ScoutAgent()
    predictor_agent = PredictorAgent()
    planner_agent = PlannerAgent()
    voice_agent = (
        VoiceAgent(
            scout_agent=scout_agent,
            predictor_agent=predictor_agent,
            planner_agent=planner_agent,
        )
        if VoiceAgent
        else None
    )
    print("[OK] Real Backend Agents initialized successfully!")
except Exception as exc:
    print(f"[WARNING] Agent initialization error: {exc}")
    scout_agent = None
    predictor_agent = None
    planner_agent = None
    voice_agent = None


@app.route("/")
def index():
    """Render the dedicated K.I.S.A.N. AI Landing Page."""
    return render_template("landing.html")


@app.route("/dashboard")
@app.route("/app")
@app.route("/main")
def dashboard():
    """Render the open-access K.I.S.A.N. AI single-page dashboard & results terminal."""
    return render_template("index.html")


@app.route("/api/benchmarks", methods=["GET"])
def benchmarks_api():
    """Dynamically return recent real APMC records directly from local data/ directory."""
    try:
        data_dir = _project_root / "data"
        csv_path = data_dir / "mandi_historical_fallback.csv"
        records = []

        if csv_path.exists():
            import csv
            with open(csv_path, mode="r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    st = row.get("State") or row.get("state") or ""
                    mkt = row.get("Market") or row.get("market") or ""
                    cm = row.get("Commodity") or row.get("commodity") or ""
                    dt = row.get("Arrival_Date") or row.get("arrival_date") or ""
                    mp = row.get("Modal_x0020_Price") or row.get("modal_price") or row.get("Modal Price") or 0
                    if st and mkt and cm:
                        records.append({
                            "state": st,
                            "market": mkt,
                            "commodity": cm,
                            "modal_price": mp,
                            "arrival_date": dt,
                        })
                    if len(records) >= 8:
                        break

        return jsonify({"status": "success", "records": records})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/optimize", methods=["POST"])
def optimize_api():
    """Execute real multi-agent decision pipeline for spatial arbitrage optimization."""
    try:
        data = request.json or {}
        location_query = data.get("location", "").strip()
        commodity = data.get("crop", "").strip()
        raw_quantity = data.get("quantity")

        if not location_query:
            return jsonify({"status": "error", "message": "Location query is required."}), 400
        if not commodity:
            return jsonify({"status": "error", "message": "Commodity / crop selection is required."}), 400
        if raw_quantity is None:
            return jsonify({"status": "error", "message": "Yield quantity in quintals is required."}), 400

        try:
            quantity_qt = float(raw_quantity)
            if quantity_qt <= 0:
                raise ValueError
        except ValueError:
            return jsonify({"status": "error", "message": "Quantity must be a positive number."}), 400

        if not planner_agent or not predictor_agent or not scout_agent:
            return jsonify({
                "status": "error",
                "message": "Backend multi-agent system not initialized."
            }), 500

        # ---------------------------------------------------------------------
        # STEP 1: Geocode Farmer Location via PlannerAgent (Pure Live Nominatim)
        # ---------------------------------------------------------------------
        loc_info = planner_agent.geocode_farmer_location(location_query)
        farmer_lat = loc_info["latitude"]
        farmer_lon = loc_info["longitude"]
        state = loc_info["state"]
        district = loc_info["district"]

        # ---------------------------------------------------------------------
        # STEP 2: Scout Live Weather & Real-Data Agro-Climatic Vigor (NDVI)
        # ---------------------------------------------------------------------
        env_data = scout_agent.fetch_weather(farmer_lat, farmer_lon)
        ndvi_data = predictor_agent.generate_ndvi_index(state, commodity, district=district)

        # ---------------------------------------------------------------------
        # STEP 3: Discover Candidate APMC Markets from data/ folder
        # ---------------------------------------------------------------------
        candidates = planner_agent.discover_candidate_markets(
            farmer_lat=farmer_lat,
            farmer_lon=farmer_lon,
            state=state,
            commodity=commodity,
        )

        if not candidates:
            return jsonify({
                "status": "error",
                "message": (
                    f"No active or historical APMC mandi records found in data folder for crop '{commodity}' "
                    f"in state '{state}'. Please choose a commodity available in the dataset (e.g. Onion, Potato, Wheat, Tomato, Soybean, Cotton, Paddy, Maize, Banana, Apple) "
                    f"or search near other producing regions."
                )
            }), 404

        # ---------------------------------------------------------------------
        # STEP 4: Scrape Live Diesel Rate & Predict ML Quantile Price Bands
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
            m_modal = cm.get("modal_price")
            try:
                modal_price_val = float(m_modal) if m_modal is not None and not (isinstance(m_modal, float) and math.isnan(m_modal)) else 0.0
            except (ValueError, TypeError):
                modal_price_val = 0.0

            mandi_rec = {
                "state": state,
                "district": str(cm.get("district") or district),
                "market": clean_market_name,
                "commodity": commodity,
                "variety": str(cm.get("variety") or "Local"),
                "modal_price": modal_price_val,
                "arrival_date": cm.get("arrival_date"),
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
        assigned_vehicle = winner.get("truck_type") or winner.get("vehicle_assigned") or "Commercial Freight Truck"

        # Calculate genuine extra savings over alternative local/worst options
        worst_market_profit = logistics_results[-1].get("net_expected_profit_inr", 0) if len(logistics_results) > 1 else 0
        hero_market_profit = winner.get("net_expected_profit_inr", 0)
        raw_savings = hero_market_profit - worst_market_profit
        savings_over_local = max(0.0, raw_savings)

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
            "diesel_state": state,
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


# =============================================================================
# K.I.S.A.N. AI FARMER CONVERSATIONAL CHATBOT & VOICE API
# =============================================================================

CHAT_SESSION_REGISTRY = {}


@app.route("/api/chat", methods=["POST"])
def chat_api():
    """Real-time conversational chatbot API connecting farmer directly with VoiceAgent."""
    try:
        if not voice_agent:
            return jsonify({
                "status": "error",
                "message": "VoiceAgent is currently not initialized on this instance."
            }), 500

        data = request.json or {}
        user_message = data.get("message", "").strip()
        session_id = data.get("session_id") or f"CHAT_{uuid_str()}"
        language = data.get("language", "hi-IN")
        audio_b64 = data.get("audio_base64")

        # If browser audio bytes were sent, transcribe with speech_recognition
        if audio_b64:
            try:
                if "," in audio_b64:
                    audio_b64 = audio_b64.split(",", 1)[1]
                raw_audio = base64.b64decode(audio_b64)
                stt_res = voice_agent.transcribe_audio(raw_audio, language=language)
                if stt_res.get("status") == "success" and stt_res.get("transcript"):
                    user_message = stt_res["transcript"]
            except Exception as stt_err:
                print(f"[LOG] Audio transcription exception: {stt_err}")

        if not user_message:
            return jsonify({"status": "error", "message": "Message is required."}), 400

        session_memory = CHAT_SESSION_REGISTRY.get(session_id, {})

        # Execute multi-turn conversational chat logic
        chat_res = voice_agent.chat(
            message=user_message,
            session_state=session_memory,
            language=language,
        )

        CHAT_SESSION_REGISTRY[session_id] = chat_res.get("session_state", {})

        return jsonify({
            "status": "success",
            "session_id": session_id,
            "user_message": user_message,
            "reply_text": chat_res.get("reply_text", ""),
            "spoken_text": chat_res.get("spoken_text", ""),
            "audio_url": chat_res.get("audio_url", ""),
            "recommendation": chat_res.get("recommendation"),
            "entities": chat_res.get("entities", {}),
            "language": language,
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/chat/reset", methods=["POST"])
def chat_reset_api():
    """Reset chatbot session memory for a fresh conversation."""
    data = request.json or {}
    session_id = data.get("session_id")
    if session_id and session_id in CHAT_SESSION_REGISTRY:
        del CHAT_SESSION_REGISTRY[session_id]
    return jsonify({"status": "success", "message": "Chat session reset successfully."})


@app.route("/api/voice/transcribe", methods=["POST"])
def voice_transcribe_api():
    """Transcribe raw audio from web microphone into text."""
    try:
        if not voice_agent:
            return jsonify({"status": "error", "message": "VoiceAgent unavailable."}), 500

        language = request.form.get("language") or request.args.get("language", "hi-IN")
        if "audio" in request.files:
            file_obj = request.files["audio"]
            audio_bytes = file_obj.read()
            stt = voice_agent.transcribe_audio(audio_bytes, language=language)
            return jsonify(stt)

        data = request.json or {}
        b64_str = data.get("audio_base64", "")
        if b64_str:
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            raw_audio = base64.b64decode(b64_str)
            stt = voice_agent.transcribe_audio(raw_audio, language=language)
            return jsonify(stt)

        return jsonify({"status": "error", "message": "No audio file or base64 data provided."}), 400
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500





def uuid_str():
    import uuid
    return uuid.uuid4().hex[:8].upper()


if __name__ == "__main__":
    print("=" * 70)
    print("🌾 K.I.S.A.N. AI Multi-Agent Flask Backend Running!")
    print("Open Browser: http://127.0.0.1:5000")
    print("=" * 70)
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
