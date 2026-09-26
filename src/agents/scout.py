"""Scout Agent module for fetching real-time meteorological and agricultural market data."""

import csv
import os
import pprint
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import requests
from dotenv import load_dotenv


class ScoutAgent:
    """Agent responsible for scouting environmental and market intelligence."""

    OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"
    DATA_GOV_MANDI_ENDPOINT = (
        "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a86454359444"
    )

    FIELD_KEY_MAP = {
        "state": ["state", "State", "State Name", "state name", "state_name"],
        "commodity": ["commodity", "Commodity", "commodity name", "Commodity Name"],
        "market": ["market", "Market", "market_name", "Market_Name", "Market Name", "market name"],
        "variety": ["variety", "Variety"],
        "arrival_date": ["arrival_date", "Arrival_Date", "arrival date", "Arrival Date", "Reported Date", "reported date", "reported_date"],
        "min_price": [
            "min_price",
            "Min_x0020_Price",
            "Min Price",
            "Min_Price",
            "min price",
            "Min Price (Rs./Quintal)",
            "min price (rs./quintal)",
        ],
        "max_price": [
            "max_price",
            "Max_x0020_Price",
            "Max Price",
            "Max_Price",
            "max price",
            "Max Price (Rs./Quintal)",
            "max price (rs./quintal)",
        ],
        "modal_price": [
            "modal_price",
            "Modal_x0020_Price",
            "Modal Price",
            "Modal_Price",
            "modal price",
            "Modal Price (Rs./Quintal)",
            "modal price (rs./quintal)",
        ],
        "district": ["district", "District", "District Name", "district name", "district_name"],
    }

    def __init__(self, env_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize ScoutAgent and load configuration.

        Args:
            env_path: Optional path to the .env file. Defaults to config/.env
                      relative to project root.

        Raises:
            ValueError: If GOV_API_KEY is not found or is empty in the environment.
        """
        if env_path:
            resolved_env_path = Path(env_path).resolve()
        else:
            # Resolve config/.env relative to project root (2 levels up from src/agents)
            project_root = Path(__file__).resolve().parent.parent.parent
            resolved_env_path = project_root / "config" / ".env"

        # Dynamically load environment variables from config/.env
        if resolved_env_path.exists():
            load_dotenv(dotenv_path=resolved_env_path)
        else:
            load_dotenv()

        self.gov_api_key: Optional[str] = os.getenv("GOV_API_KEY")
        self.api_key: Optional[str] = self.gov_api_key

        if not self.gov_api_key or not self.gov_api_key.strip():
            raise ValueError(
                f"GOV_API_KEY is missing or empty. Please configure it in '{resolved_env_path}'."
            )

    @staticmethod
    def _get_field_value(row: Dict[str, Any], field: str) -> Any:
        """Retrieve value from dictionary/row matching key variations case-insensitively.

        Handles escaped spaces (e.g. Min_x0020_Price) and alternate key names.
        """
        candidates = ScoutAgent.FIELD_KEY_MAP.get(field.lower(), [field])

        for key in row.keys():
            if key is None:
                continue
            key_str = str(key).strip()
            for cand in candidates:
                if key_str.lower() == cand.lower():
                    return row[key]

        for key in row.keys():
            if key is None:
                continue
            normalized_key = (
                str(key).strip().replace("_x0020_", " ").replace("_", " ").lower()
            )
            for cand in candidates:
                normalized_cand = cand.replace("_x0020_", " ").replace("_", " ").lower()
                if normalized_key == normalized_cand:
                    return row[key]

        return None

    @staticmethod
    def _safe_int(val: Any, default: int = 0) -> int:
        """Safely convert numeric string or value to integer."""
        if val is None:
            return default
        try:
            clean_str = str(val).strip().replace(",", "")
            return int(float(clean_str))
        except (ValueError, TypeError):
            return default

    def fetch_weather(
        self, latitude: float, longitude: float, timeout: int = 10
    ) -> Dict[str, Any]:
        """Fetch current weather data for a given geographical coordinate.

        Args:
            latitude: Latitude coordinate (e.g. 19.0760).
            longitude: Longitude coordinate (e.g. 72.8777).
            timeout: Request timeout in seconds. Defaults to 10.

        Returns:
            Dict containing latitude, longitude, temperature, windspeed, weathercode, and time.

        Raises:
            RuntimeError: If request fails, times out, or returns invalid data.
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current_weather": "true",
        }

        try:
            response = requests.get(
                self.OPEN_METEO_BASE_URL, params=params, timeout=timeout
            )
            response.raise_for_status()
            data = response.json()

            current = data.get("current_weather")
            if not current:
                raise ValueError("Response missing 'current_weather' payload.")

            return {
                "latitude": data.get("latitude", latitude),
                "longitude": data.get("longitude", longitude),
                "temperature": current.get("temperature"),
                "windspeed": current.get("windspeed"),
                "weathercode": current.get("weathercode"),
                "time": current.get("time"),
            }

        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"Weather API request timed out after {timeout} seconds for coordinates ({latitude}, {longitude})."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"Failed to fetch weather data from Open-Meteo: {exc}"
            ) from exc
        except (ValueError, KeyError) as exc:
            raise RuntimeError(
                f"Error parsing weather response from Open-Meteo: {exc}"
            ) from exc

    def fetch_live_mandi_prices(
        self, state: str, commodity: str, timeout: int = 5
    ) -> List[Dict[str, Any]]:
        """Query local Mandi database records for mandi (market) prices.

        Parses local CSV datasets dynamically per Rule 1.

        Args:
            state: Target state name (e.g., 'Uttar Pradesh').
            commodity: Target commodity name (e.g., 'Wheat').
            timeout: Deprecated parameter kept for backwards compatibility.

        Returns:
            List of dictionaries representing active mandis with price details.
        """
        print(f"[DATA ENGINE] Ingesting dynamic local market prices for '{state}' - '{commodity}'...")
        return self._load_csv_fallback(state, commodity)

    def _load_csv_fallback(self, state: str, commodity: str) -> List[Dict[str, Any]]:
        """Load mandi price records from local CSV files in data/ directory and subdirectories.

        Locates all .csv files in data/ (including data_vegetable_wise/) relative to project root,
        dynamically handles raw government headers case-insensitively, and returns
        matched records with prices cast as integers.

        Args:
            state: Target state name.
            commodity: Target commodity name.

        Returns:
            List of matched records with prices cast as integers.
        """
        project_root = Path(__file__).resolve().parent.parent.parent
        data_dir = project_root / "data"

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found at '{data_dir}'.")

        target_state = state.strip().lower()
        target_commodity = commodity.strip().lower()
        matched_records: List[Dict[str, Any]] = []

        # 1. Prioritize targeted crop CSV in data/data_vegetable_wise/
        csv_files: List[Path] = []
        veg_dir = data_dir / "data_vegetable_wise"
        if veg_dir.exists():
            for p in veg_dir.glob("*.csv"):
                if target_commodity and target_commodity in p.stem.lower():
                    csv_files.append(p)

        # 2. Add root data/*.csv files
        for p in data_dir.glob("*.csv"):
            csv_files.append(p)

        for csv_path in csv_files:
            try:
                with open(csv_path, mode="r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row_state = str(self._get_field_value(row, "state") or "").strip().lower()
                        extracted_comm = self._get_field_value(row, "commodity")
                        if not extracted_comm and "data_vegetable_wise" in str(csv_path):
                            extracted_comm = csv_path.stem
                        row_commodity = str(extracted_comm or "").strip().lower()

                        state_match = bool(row_state) and (target_state in row_state or row_state in target_state)
                        commodity_match = bool(row_commodity) and (target_commodity in row_commodity or row_commodity in target_commodity)

                        if state_match and commodity_match:
                            market = self._get_field_value(row, "market") or "N/A"
                            district = self._get_field_value(row, "district") or "N/A"
                            variety = self._get_field_value(row, "variety") or "N/A"
                            arrival_date = (
                                self._get_field_value(row, "arrival_date") or "N/A"
                            )
                            min_price = self._safe_int(self._get_field_value(row, "min_price"))
                            max_price = self._safe_int(self._get_field_value(row, "max_price"))
                            modal_price = self._safe_int(
                                self._get_field_value(row, "modal_price")
                            )
                            st_val = self._get_field_value(row, "state") or state
                            cm_val = self._get_field_value(row, "commodity") or (csv_path.stem if "data_vegetable_wise" in str(csv_path) else commodity)

                            matched_records.append(
                                {
                                    "market_name": market,
                                    "state": st_val,
                                    "district": district,
                                    "commodity": cm_val,
                                    "variety": variety,
                                    "arrival_date": arrival_date,
                                    "min_price": min_price,
                                    "max_price": max_price,
                                    "modal_price": modal_price,
                                }
                            )
                            if len(matched_records) >= 30:
                                break
                    if len(matched_records) >= 30:
                        break
            except Exception as exc:
                print(f"[LOG] Error reading fallback CSV '{csv_path.name}': {exc}")
        return matched_records



if __name__ == "__main__":
    print("=" * 60)
    print("Testing ScoutAgent...")
    print("=" * 60)

    try:
        scout = ScoutAgent()
        print("[OK] ScoutAgent initialized successfully.\n")

        # 1. Fetch weather for Mumbai (19.0760, 72.8777)
        print("Fetching current weather for Mumbai (19.0760, 72.8777)...")
        mumbai_weather = scout.fetch_weather(latitude=19.0760, longitude=72.8777)
        print("Weather Data:")
        pprint.pprint(mumbai_weather)
        print("-" * 60)

        # 2. Fetch mandi prices for Potato in Maharashtra
        print("Fetching mandi prices for 'Potato' in 'Maharashtra'...")
        potato_prices = scout.fetch_live_mandi_prices(
            state="Maharashtra", commodity="Potato"
        )
        print(f"Mandi Prices ({len(potato_prices)} records found):")
        pprint.pprint(potato_prices)
        print("=" * 60)

    except Exception as exc:
        print(f"[!] Error during execution: {exc}")
