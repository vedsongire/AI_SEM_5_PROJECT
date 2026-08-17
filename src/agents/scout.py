"""Scout Agent module for fetching real-time meteorological and agricultural market data."""

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

        # Load environment variables if file exists
        if resolved_env_path.exists():
            load_dotenv(dotenv_path=resolved_env_path)
        else:
            # Also try default load_dotenv in case it's in standard location
            load_dotenv()

        self.gov_api_key: Optional[str] = os.getenv("GOV_API_KEY")
        self.api_key: Optional[str] = self.gov_api_key

        if not self.gov_api_key or not self.gov_api_key.strip():
            raise ValueError(
                f"GOV_API_KEY is missing or empty. Please configure it in '{resolved_env_path}'."
            )

    def fetch_weather(
        self, latitude: float, longitude: float, timeout: int = 10
    ) -> Dict[str, Any]:
        """Fetch current weather data for a given geographical coordinate.

        Args:
            latitude: Latitude coordinate (e.g. 19.0760).
            longitude: Longitude coordinate (e.g. 72.8777).
            timeout: Request timeout in seconds. Defaults to 10.

        Returns:
            Dict containing temperature, windspeed, and weathercode.

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
                "latitude": latitude,
                "longitude": longitude,
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
        self, state: str, commodity: str, limit: int = 5, timeout: int = 30
    ) -> List[Dict[str, Any]]:
        """Query official Indian Government API for live mandi (market) prices.

        Args:
            state: State name (e.g., 'Maharashtra', 'Uttar Pradesh').
            commodity: Commodity name (e.g., 'Potato', 'Onion').
            limit: Maximum number of active mandis to return (default: 5).
            timeout: Request timeout in seconds. Defaults to 30.

        Returns:
            List of dictionaries representing active mandis with price details.

        Raises:
            RuntimeError: If request fails, unauthorized, or cannot parse records.
        """
        params = {
            'api-key': self.api_key,
            'format': 'json',
            'limit': 5,
            'filters[state]': state,
            'filters[commodity]': commodity
        }

        try:
            print("Connecting to OGD India servers (this may take up to 30 seconds)...")
            response = requests.get(
                self.DATA_GOV_MANDI_ENDPOINT, params=params, timeout=timeout
            )
            response.raise_for_status()
            data = response.json()

            records = data.get("records", [])
            mandi_list: List[Dict[str, Any]] = []

            for record in records[:limit]:
                mandi_info = {
                    "market_name": record.get("market") or record.get("market_name", "N/A"),
                    "state": record.get("state", state),
                    "district": record.get("district", "N/A"),
                    "commodity": record.get("commodity", commodity),
                    "variety": record.get("variety", "N/A"),
                    "arrival_date": record.get("arrival_date", "N/A"),
                    "min_price": record.get("min_price"),
                    "max_price": record.get("max_price"),
                    "modal_price": record.get("modal_price"),
                }
                mandi_list.append(mandi_info)

            return mandi_list

        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"Mandi API request timed out after {timeout} seconds for state '{state}', commodity '{commodity}'."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else "Unknown"
            raise RuntimeError(
                f"Mandi API returned HTTP error {status_code}: {exc}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"Failed to fetch mandi price data from data.gov.in: {exc}"
            ) from exc
        except (ValueError, KeyError) as exc:
            raise RuntimeError(
                f"Error parsing mandi price response: {exc}"
            ) from exc


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

        # 2. Fetch live mandi prices for Potato in Maharashtra
        print("Fetching live mandi prices for 'Potato' in 'Maharashtra'...")
        potato_prices = scout.fetch_live_mandi_prices(
            state="Maharashtra", commodity="Potato", limit=5
        )
        print("Mandi Prices (Top 5 active markets):")
        pprint.pprint(potato_prices)
        print("=" * 60)

    except ValueError as val_err:
        print(f"[!] Configuration Warning: {val_err}")
        print("\nNote: Demonstrating standalone weather fetching without GOV_API_KEY:")
        try:
            temp_agent = object.__new__(ScoutAgent)
            weather_data = ScoutAgent.fetch_weather(temp_agent, 19.0760, 72.8777)
            print("Weather Data (Mumbai 19.0760, 72.8777):")
            pprint.pprint(weather_data)
        except Exception as err:
            print(f"Weather fetch error: {err}")
    except Exception as exc:
        print(f"[!] Unexpected error during execution: {exc}")
