---
trigger: always_on
---

# SYSTEM RULES FOR DEVELOPMENT: K.I.S.A.N. AI PROJECT
You are a highly rigorous, production-grade senior software engineer. Your absolute mandate is to write 100% data-driven, dynamic Python and Streamlit code. You are strictly forbidden from faking data, writing static mock outputs, or using pre-defined hardcoded variables in your script logic.

## 1. DATA SOURCE MANDATE: LOCAL DATABASES & FALLBACKS
- **NO Unstable Live Mandi APIs:** Do NOT use or attempt to query unstable third-party live agricultural price APIs.
- **Mandatory Local File Parsing:** All mandi locations, historical prices, and crop features must be parsed and queried dynamically from:
  * `historical_mandi_fallback.csv`
  * `data.csv`
  * The `data_vegetable_wise/` folder (which contains individual CSV files named by crop/vegetable containing localized market prices).
- **Dynamic File Ingestion:** Use `pandas` to read these local datasets live based on the selected crop and farmer's district. If the files are missing or corrupted, fail honestly with a clear error message instead of hardcoding fake data inside the script.

## 2. GEOSPATIAL & ROUTING RULES (SCOUT & PLANNER)
- **Dynamic Proximity & Coordinates Scan:** 
  - Read the locations of available mandis dynamically from the local `data.csv` or `historical_mandi_fallback.csv`. 
  - Geocode the farmer's location and the discovered mandi locations live via `geopy`'s Nominatim (with no hardcoded coordinate dictionaries). 
  - Calculate straight-line distance to find the closest markets dynamically, always including "Mumbai Vashi APMC" as the terminal market benchmark.
- **Roadway Distance:** Call the OSRM Routing API to calculate true driving distances. If offline, fallback to Haversine straight-line distance multiplied by a 1.3 winding factor.
- **Dynamic Vehicle & Logistics Calculations:**
  - Scale vehicle size dynamically based on yield: Bolero Pickup (<=15 quintals), 8-Ton Tata LPT 1109 Truck (<=100 quintals), or 20-Ton Multi-Axle Truck (>100 quintals).
  - Calculate round-trip fuel overhead using today's scraped diesel price from GoodReturns (fallback to ₹97.83/L if offline).

## 3. INTEGRATION WITH ML PREDICTOR
- **Feature Extraction:** Extract the historical price trends, crop variety, and location details from the local CSVs dynamically.
- **Live Prediction Loop:** Pass these extracted real-data features directly into our trained ML model equations inside the `PredictorAgent` to compute the risk-adjusted p10, p50, and p90 quantile bands live.

## 4. UI/UX DESIGN & DYNAMIC PRESENTATION
- **Complete Logic Separation:** Keep frontend presentation completely separate from the backend agent computing loops.
- **Visual Elevation (Glassmorphism):**
  - Inject custom CSS styles to override default Streamlit layouts.
  - Wrap metrics in rounded cards (`border-radius: 16px`) with modern soft shadows (`box-shadow: 0 4px 20px rgba(0,0,0,0.04)`), and add a smooth lifting scale-up effect when hovered.
  - Design a gorgeous, centered, secure login portal supporting our 5 default farmer profiles (Ramesh, Suresh, Savita, Anil, Sunita) that pre-loads their personal data and transaction history directly from a clean dictionary structure.

## 5. TRANSPARENT ERROR HANDLING
- If local files are unreadable, or geocoding times out, do NOT bypass them with fake fallcoded data. Display a clean, professional `st.error` alert to the user explaining the exact exception.






MOST IMPORTANT : NO HARDCODED DATA ANYWHERE SHOULD BE USED IN THIS PROJECT