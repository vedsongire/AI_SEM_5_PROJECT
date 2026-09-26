"""Planner Agent module for dynamic geospatial mandi discovery, live OSRM routing, diesel scraping, and logistics optimization."""

import asyncio
import csv
import inspect
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Dynamic path setup to ensure src and project root are always in sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import requests
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderRateLimited, GeocoderTimedOut, GeocoderServiceError


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle (Haversine) distance between two geographical points in kilometers.

    Args:
        lat1, lon1: Starting coordinates in decimal degrees.
        lat2, lon2: Destination coordinates in decimal degrees.

    Returns:
        Distance in kilometers (float).
    """
    R = 6371.0  # Earth's radius in kilometers
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def resolve_awaitable(val: Any) -> Any:
    """Safely resolve an awaitable/coroutine object if returned in synchronous execution contexts."""
    if inspect.isawaitable(val):
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, val)
                    return future.result()
            else:
                return asyncio.run(val)
        except Exception:
            return None
    return val


class PlannerAgent:
    """Agent responsible for dynamic geospatial market discovery, live routing, and transport economics."""

    def __init__(self) -> None:
        """Initialize PlannerAgent with Nominatim geocoder and in-memory geocoding cache."""
        self.geolocator = Nominatim(user_agent="KisanAI_Research_Project_v3/1.0 (academic_research_agent)")
        self._geocoding_cache: Dict[str, Any] = {}
        print("[OK] PlannerAgent initialized successfully.")

    def geocode_farmer_location(self, location_query: str) -> Dict[str, Any]:
        """Geocode a farmer's location query dynamically using live OpenStreetMap/Nominatim API.

        Args:
            location_query: City, village, district, or address query (e.g. "Muzaffarnagar, Uttar Pradesh").

        Returns:
            Dict containing 'latitude', 'longitude', 'state', 'district', and 'display_name'.

        Raises:
            ValueError: If location query is empty.
            RuntimeError: If live geocoding fails or location cannot be resolved.
        """
        clean_q = location_query.strip()
        if not clean_q:
            raise ValueError("Location query cannot be empty.")

        cache_key = f"farmer_{clean_q.lower()}"
        if cache_key in self._geocoding_cache:
            coords, state, district, display_name = self._geocoding_cache[cache_key]
            return {
                "latitude": coords[0],
                "longitude": coords[1],
                "state": state,
                "district": district,
                "display_name": display_name,
            }

        parts = [p.strip() for p in clean_q.split(",")]
        queries = [clean_q]
        if len(parts) > 1:
            queries.append(f"{parts[0]}, {parts[-1]}, India")
            queries.append(f"{parts[0]}, India")
        elif "india" not in clean_q.lower():
            queries.append(f"{clean_q}, India")

        last_error = None
        for q in queries:
            try:
                location = resolve_awaitable(
                    self.geolocator.geocode(q, addressdetails=True, timeout=5.0)
                )
                if location and hasattr(location, "latitude") and hasattr(location, "longitude"):
                    farmer_lat = float(location.latitude)
                    farmer_lon = float(location.longitude)
                    raw_addr = getattr(location, "raw", {}).get("address", {})

                    detected_state = (
                        raw_addr.get("state")
                        or raw_addr.get("region")
                        or raw_addr.get("state_district")
                        or (parts[-1] if len(parts) > 1 else "India")
                    ).strip()

                    detected_district = (
                        raw_addr.get("county")
                        or raw_addr.get("state_district")
                        or raw_addr.get("district")
                        or raw_addr.get("city")
                        or raw_addr.get("town")
                        or parts[0]
                    ).strip()

                    display_name = getattr(location, "address", location_query)
                    self._geocoding_cache[cache_key] = (
                        (farmer_lat, farmer_lon),
                        detected_state,
                        detected_district,
                        display_name,
                    )

                    return {
                        "latitude": farmer_lat,
                        "longitude": farmer_lon,
                        "state": detected_state,
                        "district": detected_district,
                        "display_name": display_name,
                    }
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            f"Could not geocode location '{location_query}' via live Nominatim API: {last_error or 'Location not found'}. "
            f"Please ensure network connectivity and provide a valid Indian district, city, or village name."
        )


    def _load_fallback_mandis_from_csv(
        self, state: str, commodity: str = ""
    ) -> List[Dict[str, Any]]:
        """Load historical fallback mandi records directly from CSV files in data/ or data/data_vegetable_wise/.

        Args:
            state: Name of the target state (e.g. 'Uttar Pradesh').
            commodity: Target commodity name (e.g. 'Wheat').

        Returns:
            List of mandi record dicts loaded from fallback CSV files.
        """
        project_root = Path(__file__).resolve().parent.parent.parent
        data_dir = project_root / "data"
        data_veg_dir = data_dir / "data_vegetable_wise"

        target_state = state.strip().lower()
        target_commodity = commodity.strip().lower()

        comm_records: List[Dict[str, Any]] = []
        state_records: List[Dict[str, Any]] = []

        # 1. Query curated fallback CSVs first (instant lookup: ~0.02s)
        for fn in ["mandi_historical_fallback.csv", "data2.csv"]:
            fp = data_dir / fn
            if not fp.exists():
                continue
            try:
                with open(fp, mode="r", encoding="utf-8-sig") as f:
                    for row in csv.DictReader(f):
                        row_state = (row.get("State") or row.get("state") or "").strip().lower()
                        if target_state not in row_state and row_state not in target_state:
                            continue
                        row_comm = (row.get("Commodity") or row.get("commodity") or "").strip().lower()
                        if target_commodity and (target_commodity in row_comm or row_comm in target_commodity):
                            m_name = row.get("Market") or row.get("market") or "APMC Market"
                            dist_name = row.get("District") or row.get("district") or state
                            modal_p = row.get("Modal_x0020_Price") or row.get("modal_price") or "2200"
                            comm_records.append({
                                "market_name": m_name,
                                "district": dist_name,
                                "state": row.get("State") or state,
                                "commodity": commodity,
                                "modal_price": modal_p,
                                "arrivals_tonnes": "10",
                            })
                            if len(comm_records) >= 30:
                                break
            except Exception as exc:
                print(f"[LOG] Error reading fallback '{fn}': {exc}")
            if len(comm_records) >= 30:
                break

        # 2. Query crop-specific CSV inside data/data_vegetable_wise if needed
        if len(comm_records) < 5 and data_veg_dir.exists() and target_commodity:
            matching_files = [
                f for f in data_veg_dir.glob("*.csv")
                if target_commodity in f.stem.lower() or f.stem.lower() in target_commodity
            ]
            if matching_files:
                target_file = matching_files[0]
                try:
                    with open(target_file, mode="r", encoding="utf-8-sig") as f:
                        for row in csv.DictReader(f):
                            row_state = (row.get("State Name") or row.get("state") or row.get("State") or "").strip().lower()
                            if target_state not in row_state and row_state not in target_state:
                                continue
                            m_name = row.get("Market Name") or row.get("market") or "APMC Market"
                            dist_name = row.get("District Name") or row.get("district") or state
                            modal_p = (
                                row.get("Modal Price (Rs./Quintal)")
                                or row.get("Modal Price")
                                or row.get("modal_price")
                                or "2200"
                            )
                            comm_records.append({
                                "market_name": m_name,
                                "district": dist_name,
                                "state": row.get("State Name") or state,
                                "commodity": commodity,
                                "modal_price": modal_p,
                                "arrivals_tonnes": row.get("Arrivals (Tonnes)") or row.get("Arrivals") or "0",
                            })
                            if len(comm_records) >= 30:
                                break
                except Exception as exc:
                    print(f"[LOG] Error reading crop CSV '{target_file.name}': {exc}")

        combined = comm_records if comm_records else state_records
        # Deduplicate and ensure target state mandis are prioritized
        seen_markets = set()
        unique_combined = []
        for r in combined:
            mkt = r.get("market_name", "")
            if mkt not in seen_markets:
                seen_markets.add(mkt)
                unique_combined.append(r)

        print(
            f"[LOG] Fallback CSV Engine: Loaded {len(unique_combined)} candidate mandi records for crop='{commodity}', state='{state}'."
        )
        return unique_combined

    def discover_candidate_markets(
        self,
        farmer_lat: float,
        farmer_lon: float,
        state: str,
        active_mandi_records: Optional[List[Dict[str, Any]]] = None,
        commodity: str = "Wheat",
    ) -> List[Dict[str, Any]]:
        """Dynamically discover candidate trading markets for a farmer's location.

        1. Takes active mandi price records (from ScoutAgent live API / historical dataset).
        2. Geocodes unique active mandis using live Nominatim API.
        3. Filters mandis geographically to ensure they are in/near the farmer's state/region (< 400 km).
        4. Ranks candidate mandis by Haversine distance and returns nearest top APMC markets.

        Args:
            farmer_lat: Latitude of the farmer's location.
            farmer_lon: Longitude of the farmer's location.
            state: Name of the target state (e.g. 'Uttar Pradesh').
            active_mandi_records: Optional list of live market records from ScoutAgent.
            commodity: Target commodity name.

        Returns:
            List of dynamically selected candidate APMC market dicts.
        """
        records = active_mandi_records
        if not records:
            try:
                try:
                    from src.agents.scout import ScoutAgent
                except ImportError:
                    try:
                        from agents.scout import ScoutAgent
                    except ImportError:
                        from .scout import ScoutAgent
                scout = ScoutAgent()
                records = scout.fetch_live_mandi_prices(state=state, commodity=commodity)
            except Exception as exc:
                print(f"[LOG] Could not fetch live mandi records from Scout: {exc}")
                records = []

        if not records:
            print(f"[LOG] Live records unavailable. Falling back to 'data/mandi_historical_fallback.csv' for '{state}'.")
            records = self._load_fallback_mandis_from_csv(state=state, commodity=commodity)

        target_state_clean = str(state or "").strip().lower()

        # Deduplicate active mandis by market_name matching farmer's state
        unique_mandis: Dict[str, Dict[str, Any]] = {}
        for rec in (records or []):
            rec_st = str(rec.get("state") or "").strip().lower()
            if rec_st and target_state_clean not in rec_st and rec_st not in target_state_clean:
                continue

            raw_m = rec.get("market_name") or rec.get("market")
            if raw_m is None or (isinstance(raw_m, float) and math.isnan(raw_m)) or str(raw_m).strip().lower() in ("", "nan"):
                m_name = "APMC Market"
            else:
                m_name = str(raw_m).strip()

            m_key = m_name.lower()
            if m_key not in unique_mandis:
                dist_val = rec.get("district")
                if dist_val is None or (isinstance(dist_val, float) and math.isnan(dist_val)):
                    dist = "N/A"
                else:
                    dist = str(dist_val).strip()

                unique_mandis[m_key] = {
                    "market_name": m_name,
                    "district": dist,
                    "state": state,
                }

        # If live records yield fewer than 3 mandis for that state, add fallback CSV mandis for state
        if len(unique_mandis) < 3:
            csv_fallback_records = self._load_fallback_mandis_from_csv(state=state, commodity="")
            for rec in csv_fallback_records:
                raw_m = rec.get("market_name") or rec.get("market")
                if raw_m is None or (isinstance(raw_m, float) and math.isnan(raw_m)) or str(raw_m).strip().lower() in ("", "nan"):
                    m_name = "APMC Market"
                else:
                    m_name = str(raw_m).strip()

                m_key = m_name.lower()
                if m_key not in unique_mandis:
                    dist_val = rec.get("district")
                    if dist_val is None or (isinstance(dist_val, float) and math.isnan(dist_val)):
                        dist = state
                    else:
                        dist = str(dist_val).strip()

                    unique_mandis[m_key] = {
                        "market_name": m_name,
                        "district": dist,
                        "state": state,
                    }

        # If still fewer than 3 mandis, query Nominatim live for APMC mandis in the state
        if len(unique_mandis) < 3:
            dynamic_queries = [
                f"{commodity} APMC Mandi, {state}, India",
                f"APMC Market, {state}, India",
            ]
            for dq in dynamic_queries:
                try:
                    locs = resolve_awaitable(
                        self.geolocator.geocode(dq, exactly_one=False, limit=5, timeout=1.5)
                    )
                    if locs:
                        for l in locs:
                            addr = l.address or ""
                            parts = [p.strip() for p in addr.split(",") if p.strip()]
                            first_part = parts[0] if parts else "APMC Mandi"

                            # Normalize market name, resolving terminal Vashi and eliminating duplicated 'APMC'
                            if "vashi" in addr.lower():
                                m_name = "Vashi APMC Market, Navi Mumbai"
                            elif "apmc" in first_part.lower():
                                if len(parts) > 1 and first_part.lower() in ("apmc", "apmc market", "apmc yard", "mandi"):
                                    m_name = f"{parts[1]} {first_part}"
                                else:
                                    m_name = first_part
                            else:
                                m_name = f"{first_part} APMC"

                            m_name = re.sub(r"(?i)\bapmc\s+apmc\b", "APMC", m_name).strip()
                            m_key = m_name.lower()
                            if m_key not in unique_mandis:
                                unique_mandis[m_key] = {
                                    "market_name": m_name,
                                    "district": state,
                                    "state": state,
                                    "latitude": float(l.latitude),
                                    "longitude": float(l.longitude),
                                }
                            if len(unique_mandis) >= 5:
                                break
                except Exception:
                    pass
                if len(unique_mandis) >= 3:
                    break

        evaluated_markets: List[Dict[str, Any]] = []
        unique_items = list(unique_mandis.items())[:10]
        for m_key, m_info in unique_items:
            lat = m_info.get("latitude")
            lon = m_info.get("longitude")

            if lat is None or lon is None:
                lat, lon = self._get_mandi_coordinates(
                    market_name=m_info["market_name"],
                    state=m_info["state"],
                    district=m_info["district"],
                )

            dist_km = haversine_distance(farmer_lat, farmer_lon, lat, lon)

            # Distance Sanity Filter: Filter out mandis > 400 km away when closer regional mandis exist
            if dist_km > 400.0 and len(evaluated_markets) >= 3:
                continue

            evaluated_markets.append(
                {
                    "market_name": m_info["market_name"],
                    "district": m_info["district"],
                    "state": m_info["state"],
                    "latitude": lat,
                    "longitude": lon,
                    "haversine_distance_km": round(dist_km, 2),
                }
            )

        # Sort active mandis by Haversine distance ascending
        evaluated_markets.sort(key=lambda x: x["haversine_distance_km"])

        candidates: List[Dict[str, Any]] = []
        for idx, m in enumerate(evaluated_markets[:3], 1):
            m_copy = dict(m)
            m_copy["role_label"] = f"Candidate Option #{idx}"
            candidates.append(m_copy)

        return candidates

    def _select_truck_spec(self, yield_quintals: float) -> Dict[str, Any]:
        """Scale truck size, mileage, tolls, and loading fees based on crop yield.

        Args:
            yield_quintals: Produce weight in quintals (1 Quintal = 100 kg = 0.1 Metric Ton).

        Returns:
            Dict containing truck specifications.
        """
        if yield_quintals <= 25.0:
            return {
                "truck_type": "Pickup Truck (2.5-Ton)",
                "capacity_tons": 2.5,
                "mileage_km_l": 10.0,
                "base_toll_inr": 100.0,
                "loading_fee_inr": 400.0,
            }
        elif yield_quintals <= 60.0:
            return {
                "truck_type": "Medium Commercial Truck (5.0-Ton)",
                "capacity_tons": 5.0,
                "mileage_km_l": 8.5,
                "base_toll_inr": 200.0,
                "loading_fee_inr": 800.0,
            }
        elif yield_quintals <= 120.0:
            return {
                "truck_type": "8-Ton Truck",
                "capacity_tons": 8.0,
                "mileage_km_l": 7.5,
                "base_toll_inr": 350.0,
                "loading_fee_inr": 1200.0,
            }
        else:
            return {
                "truck_type": "Heavy Freight Commercial Truck (16.0-Ton)",
                "capacity_tons": 16.0,
                "mileage_km_l": 5.0,
                "base_toll_inr": 600.0,
                "loading_fee_inr": 2500.0,
            }

    def _get_mandi_coordinates(
        self, market_name: str, state: str = "", district: Optional[str] = ""
    ) -> Tuple[float, float]:
        """Dynamically geocode latitude and longitude for a market using live Nominatim geocoder with caching.

        Args:
            market_name: Name of the market/mandi.
            state: Name of the state.
            district: Optional district name.

        Returns:
            Tuple of (latitude, longitude).
        """
        district_str = district or ""
        state_str = state or ""
        market_str = market_name or ""

        cache_key = f"{market_str.strip().lower()}_{district_str.strip().lower()}_{state_str.strip().lower()}"
        if cache_key in self._geocoding_cache:
            return self._geocoding_cache[cache_key]

        clean_market = (
            market_str.lower()
            .replace("apmc", "")
            .replace("market", "")
            .replace("mandi", "")
            .replace("()", "")
            .strip()
        )

        queries = []
        if clean_market:
            if district_str and district_str != "N/A" and district_str.lower() not in clean_market:
                queries.append(f"{clean_market}, {district_str}, {state_str}, India")
            queries.append(f"{clean_market}, {state_str}, India")
        if district_str and district_str != "N/A":
            queries.append(f"{district_str}, {state_str}, India")

        for q in queries:
            try:
                location = resolve_awaitable(self.geolocator.geocode(q, timeout=3.0))
                if location and hasattr(location, "latitude") and hasattr(location, "longitude"):
                    coords = (float(location.latitude), float(location.longitude))
                    self._geocoding_cache[cache_key] = coords
                    return coords
            except Exception:
                pass

        # If live geocoding cannot resolve the mandi, attempt to geocode the state
        if state_str:
            try:
                loc_state = resolve_awaitable(self.geolocator.geocode(f"{state_str}, India", timeout=3.0))
                if loc_state and hasattr(loc_state, "latitude") and hasattr(loc_state, "longitude"):
                    coords = (float(loc_state.latitude), float(loc_state.longitude))
                    self._geocoding_cache[cache_key] = coords
                    return coords
            except Exception:
                pass

        raise RuntimeError(
            f"Unable to geocode mandi '{market_name}' ({district_str}, {state_str}) via live Nominatim. "
            f"No hardcoded coordinates permitted."
        )

    def _get_driving_distance_and_time(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> Dict[str, float]:
        """Fetch highway driving distance (km) and duration (mins) using live OSRM routing.

        Falls back to Haversine * 1.3 road winding factor if offline or rate-limited.

        Args:
            lat1, lon1: Farmer starting coordinates.
            lat2, lon2: Mandi destination coordinates.

        Returns:
            Dict containing 'distance_km', 'duration_mins', and routing status.
        """
        osrm_url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
        try:
            response = requests.get(osrm_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                routes = data.get("routes", [])
                if routes:
                    route = routes[0]
                    dist_meters = float(route.get("distance", 0.0))
                    dur_seconds = float(route.get("duration", 0.0))
                    return {
                        "distance_km": round(dist_meters / 1000.0, 2),
                        "duration_mins": round(dur_seconds / 60.0, 1),
                        "is_osrm": True,
                    }
        except Exception as exc:
            print(f"[LOG] OSRM routing failed ({exc}). Using Haversine * 1.3 fallback.")

        # Fallback using Haversine distance with 1.3x road winding multiplier
        hav_dist_km = haversine_distance(lat1, lon1, lat2, lon2)
        road_dist_km = hav_dist_km * 1.3
        est_duration_mins = (road_dist_km / 40.0) * 60.0

        return {
            "distance_km": round(road_dist_km, 2),
            "duration_mins": round(est_duration_mins, 1),
            "is_osrm": False,
        }

    def _get_live_diesel_price(self, state: str = "Maharashtra") -> float:
        """Scrape today's active diesel price for the given state dynamically from GoodReturns.
        Parses state-specific table records instead of site-wide header averages.
        Falls back gracefully to Rs. 97.83/liter if offline.

        Args:
            state: Name of the state (e.g. 'Uttar Pradesh', 'Maharashtra').

        Returns:
            Diesel price per liter in INR (float).
        """
        st_clean = str(state or "Maharashtra").strip()
        state_slug = "-".join(st_clean.lower().split())
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        def _extract_price_from_soup(soup_obj: BeautifulSoup, target_state_str: str) -> Optional[float]:
            clean_target = target_state_str.lower().strip()
            for table in soup_obj.find_all("table"):
                for row in table.find_all("tr"):
                    cols = [c.get_text().strip() for c in row.find_all(["td", "th"])]
                    if len(cols) >= 2:
                        row_label = cols[0].strip().lower()
                        if row_label == clean_target or clean_target in row_label or row_label in clean_target:
                            m = re.search(r"(\d{2,3}\.\d{2})", cols[1])
                            if m:
                                return float(m.group(1))
            return None

        # 1. State-specific GoodReturns page (e.g. diesel-price-in-uttar-pradesh.html)
        try:
            url_state = f"https://www.goodreturns.in/diesel-price-in-{state_slug}.html"
            response = requests.get(url_state, headers=headers, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                price = _extract_price_from_soup(soup, st_clean)
                if price:
                    print(f"[LOG] Scraped live dynamic diesel price for '{st_clean}': Rs. {price:.2f}/liter")
                    return price
        except Exception as exc:
            print(f"[LOG] State page scrape exception ({exc}). Trying national directory...")

        # 2. National GoodReturns diesel directory table (contains all Indian states)
        try:
            url_national = "https://www.goodreturns.in/diesel-price.html"
            resp_nat = requests.get(url_national, headers=headers, timeout=5)
            if resp_nat.status_code == 200:
                soup_nat = BeautifulSoup(resp_nat.content, "html.parser")
                price = _extract_price_from_soup(soup_nat, st_clean)
                if price:
                    print(f"[LOG] Scraped live dynamic diesel price for '{st_clean}' from national directory: Rs. {price:.2f}/liter")
                    return price
        except Exception as exc:
            print(f"[LOG] National diesel directory scrape exception ({exc}).")

        fallback_price = 97.83
        print(f"[LOG] Using baseline diesel price for '{state}': Rs. {fallback_price:.2f}/liter")
        return fallback_price

    def optimize_logistics(
        self,
        farmer_lat: float,
        farmer_lon: float,
        predictions_by_market: Dict[str, Dict[str, Any]],
        quantity_quintals: float,
        state: str = "Maharashtra",
        target_markets: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Compute transport costs, live fuel economics, and net pocket profit for candidate mandis.

        Args:
            farmer_lat: Starting latitude of farmer.
            farmer_lon: Starting longitude of farmer.
            predictions_by_market: Dictionary mapping market_name -> ML quantile prediction dict.
            quantity_quintals: Quantity of crop produce in quintals.
            state: Target state for diesel lookup.
            target_markets: Optional list of candidate markets from discover_candidate_markets.

        Returns:
            Sorted list of market logistics recommendations ranked by net expected profit.
        """
        live_diesel_price = self._get_live_diesel_price(state)
        truck_spec = self._select_truck_spec(quantity_quintals)

        mileage = truck_spec["mileage_km_l"]
        base_toll = truck_spec["base_toll_inr"]
        loading_fee = truck_spec["loading_fee_inr"]

        # Map candidate market metadata if provided
        market_meta_map: Dict[str, Dict[str, Any]] = {}
        if target_markets:
            for tm in target_markets:
                tm_mname = tm.get("market_name")
                if tm_mname is not None:
                    market_meta_map[str(tm_mname).strip().lower()] = tm

        results: List[Dict[str, Any]] = []

        for market_name, pred_dict in predictions_by_market.items():
            clean_name = str(market_name).strip().lower()
            meta = market_meta_map.get(clean_name, {})

            mandi_lat = meta.get("latitude")
            mandi_lon = meta.get("longitude")
            role_label = meta.get("role_label", "Candidate Market")
            raw_district = meta.get("district") or pred_dict.get("district") or "N/A"
            district: str = str(raw_district)

            if mandi_lat is None or mandi_lon is None:
                mandi_lat, mandi_lon = self._get_mandi_coordinates(market_name, state, district)

            route_info = self._get_driving_distance_and_time(
                farmer_lat, farmer_lon, mandi_lat, mandi_lon
            )
            dist_km = route_info["distance_km"]
            dur_mins = route_info["duration_mins"]
            round_trip_dist = dist_km * 2.0

            # Calculate detailed transport cost
            fuel_cost = round((round_trip_dist / mileage) * live_diesel_price, 2)
            toll_cost = round(base_toll + (max(0.0, dist_km - 50.0) * 0.8), 2)
            total_transport_cost = round(fuel_cost + toll_cost + loading_fee, 2)

            # Extract ML quantile price forecasts (Rs/Quintal)
            p50_val = pred_dict.get("p50_median_expected") or pred_dict.get("modal_price") or 2200
            p50 = float(p50_val)

            p10_val = pred_dict.get("p10_downside_floor")
            p10 = float(p10_val) if p10_val is not None else (0.85 * p50)

            p90_val = pred_dict.get("p90_upside_ceiling")
            p90 = float(p90_val) if p90_val is not None else (1.18 * p50)

            # Revenue & Net Pocket Profit calculations
            gross_expected_revenue = round(p50 * quantity_quintals, 2)
            net_expected_profit = round(gross_expected_revenue - total_transport_cost, 2)
            net_downside_profit = round((p10 * quantity_quintals) - total_transport_cost, 2)
            net_upside_profit = round((p90 * quantity_quintals) - total_transport_cost, 2)

            results.append(
                {
                    "market_name": market_name,
                    "district": district,
                    "role_label": role_label,
                    "mandi_coordinates": (mandi_lat, mandi_lon),
                    "one_way_distance_km": dist_km,
                    "driving_duration_mins": dur_mins,
                    "round_trip_distance_km": round(round_trip_dist, 2),
                    "truck_type": truck_spec["truck_type"],
                    "vehicle_assigned": truck_spec["truck_type"],
                    "truck_mileage_km_l": mileage,
                    "live_diesel_price_per_l": live_diesel_price,
                    "fuel_cost_inr": fuel_cost,
                    "toll_cost_inr": toll_cost,
                    "loading_fee_inr": loading_fee,
                    "total_transport_cost_inr": total_transport_cost,
                    "p10_downside_price": int(round(p10)),
                    "p50_median_price": int(round(p50)),
                    "p90_upside_price": int(round(p90)),
                    "gross_expected_revenue_inr": gross_expected_revenue,
                    "net_expected_profit_inr": net_expected_profit,
                    "net_downside_profit_inr": net_downside_profit,
                    "net_upside_profit_inr": net_upside_profit,
                }
            )

        # Rank markets by net expected profit (p50) descending
        results.sort(key=lambda x: x["net_expected_profit_inr"], reverse=True)
        return results


if __name__ == "__main__":
    try:
        from src.agents.scout import ScoutAgent
        from src.agents.predictor import PredictorAgent
    except ImportError:
        try:
            from agents.scout import ScoutAgent
            from agents.predictor import PredictorAgent
        except ImportError:
            from .scout import ScoutAgent
            from .predictor import PredictorAgent

    scout = ScoutAgent()
    predictor = PredictorAgent()
    planner = PlannerAgent()

    print("\n" + "=" * 70)
    print("  KISAN AI END-TO-END MULTI-AGENT ARCHITECTURE DEMO")
    print("=" * 70)

    # 1. Geocode Farmer Location
    loc = planner.geocode_farmer_location("Sonipat, Haryana")
    print("\n[STEP 1] Resolved Farmer Location:")
    print(f"   Latitude: {loc['latitude']}, Longitude: {loc['longitude']}")
    print(f"   State: {loc['state']}, District: {loc['district']}")

    # 2. Scout Live Weather & Mandi Price Feed
    weather = scout.fetch_weather(latitude=loc["latitude"], longitude=loc["longitude"])
    print("\n[STEP 2] Scouted Live Weather & Environmental Conditions:")
    print(f"   Temperature: {weather.get('temperature')} deg C, Weather Code: {weather.get('weathercode')}")

    mandi_records = scout.fetch_live_mandi_prices(state=loc["state"], commodity="Wheat")

    # 3. Simulate Crop Vigor (NDVI)
    ndvi_info = predictor.generate_ndvi_index(state=loc["state"], commodity="Wheat")
    print("\n[STEP 3] Satellite Crop Vigor Assessment (NDVI):")
    print(f"   NDVI: {ndvi_info.get('ndvi')}, Condition: {ndvi_info.get('crop_condition')}")

    # 4. Discover Candidate Mandis
    candidates = planner.discover_candidate_markets(
        farmer_lat=loc["latitude"],
        farmer_lon=loc["longitude"],
        state=loc["state"],
        active_mandi_records=mandi_records,
        commodity="Wheat",
    )
    print(f"\n[STEP 4] Discovered Candidate Mandis ({len(candidates)} found):")
    for c in candidates:
        print(f"   - {c['market_name']} ({c['district']}, {c['state']}) | Distance: {c['haversine_distance_km']} km")

    # 5. Predict Quantile Prices & SHAP Feature Attributions
    market_preds = {}
    print("\n[STEP 5] LightGBM ML Quantile Price Forecasting & SHAP Attributions:")
    for cm in candidates:
        m_name = cm["market_name"]
        rec = {
            "state": loc["state"],
            "district": cm["district"],
            "market_name": m_name,
            "commodity": "Wheat",
            "variety": "Local",
            "arrival_date": "22/08/2026",
            "modal_price": 2200,
        }
        pred = predictor.predict_quantile_prices(
            mandi_record=rec, weather_data=weather, ndvi_data=ndvi_info
        )
        shap_attr = predictor.calculate_shap_explanations(
            mandi_record=rec, weather_data=weather, ndvi_data=ndvi_info
        )
        market_preds[m_name] = pred
        print(f"   [{m_name}] Floor (p10): Rs. {pred['p10_downside_floor']} | Expected (p50): Rs. {pred['p50_median_expected']} | Ceiling (p90): Rs. {pred['p90_upside_ceiling']}")

    # 6. Optimize Logistics & Pocket Profit
    print("\n[STEP 6] Highway Logistics & Net Pocket Profit Optimization (100 Quintals):")
    recommendations = planner.optimize_logistics(
        farmer_lat=loc["latitude"],
        farmer_lon=loc["longitude"],
        predictions_by_market=market_preds,
        quantity_quintals=100.0,
        state=loc["state"],
        target_markets=candidates,
    )

    print("\n[RESULT] Final Recommended Mandi Recommendations (Ranked by Net Pocket Profit):")
    print("-" * 70)
    for rank, rec in enumerate(recommendations, 1):
        print(f"Rank #{rank}: {rec['market_name']} ({rec['district']})")
        print(f"   Dist: {rec['one_way_distance_km']} km | Transport Cost: Rs. {rec['total_transport_cost_inr']:,.2f}")
        print(f"   Expected Price: Rs. {rec['p50_median_price']}/Quintal | Gross Rev: Rs. {rec['gross_expected_revenue_inr']:,.2f}")
        print(f"   Net Expected Pocket Profit: Rs. {rec['net_expected_profit_inr']:,.2f}")
        print("-" * 70)
