"""Training script for Multi-Quantile Gradient Boosting Regressors on historical Mandi price data.

Includes a national-scale District-Wise Climatology Mapping Engine.
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder


def get_district_rainfall(state: str, district: str, month: int) -> float:
    """Calculate baseline rainfall (in mm) based on district climatology and month.

    Args:
        state: Name of the state.
        district: Name of the district.
        month: Integer month (1-12).

    Returns:
        Estimated monthly rainfall in millimeters (mm).
    """
    st = str(state or "").strip().lower()
    dist = str(district or "").strip().lower()

    try:
        m = int(month)
    except (ValueError, TypeError):
        m = 8

    high_rain_keywords = [
        "mumbai",
        "thane",
        "ratnagiri",
        "raigad",
        "sindhudurg",
        "goa",
        "south kannada",
        "udupi",
        "karwar",
        "ernakulam",
        "trivandrum",
        "kozhikode",
        "alappuzha",
        "midnapore",
        "parganas",
        "howrah",
        "hooghly",
        "darjeeling",
        "cochick",
        "shimla",
        "dehradun",
    ]

    arid_keywords = [
        "jaisalmer",
        "barmer",
        "jodhpur",
        "bikaner",
        "churu",
        "kutch",
        "banaskantha",
        "patan",
        "anantapur",
        "kurnool",
        "bellary",
        "bijapur",
        "raichur",
        "gulbarga",
    ]

    is_high_rain = any(kw in dist for kw in high_rain_keywords)
    is_arid = any(kw in dist for kw in arid_keywords)

    if is_high_rain:
        if m == 6:
            return 400.0
        elif m == 7:
            return 800.0
        elif m == 8:
            return 700.0
        elif m == 9:
            return 350.0
        else:
            return 10.0

    elif is_arid:
        if m == 6:
            return 40.0
        elif m == 7:
            return 90.0
        elif m == 8:
            return 70.0
        elif m == 9:
            return 40.0
        else:
            return 0.0

    else:
        coastal_heavy_states = [
            "maharashtra",
            "west bengal",
            "kerala",
            "karnataka",
            "assam",
            "odisha",
            "tripura",
        ]
        inland_plains_states = [
            "uttar pradesh",
            "bihar",
            "madhya pradesh",
            "punjab",
            "haryana",
            "uttarakhand",
            "himachal pradesh",
        ]

        if any(s in st for s in coastal_heavy_states):
            if m == 6:
                return 150.0
            elif m == 7:
                return 250.0
            elif m == 8:
                return 200.0
            elif m == 9:
                return 150.0
            else:
                return 5.0
        elif any(s in st for s in inland_plains_states):
            if m == 6:
                return 120.0
            elif m == 7:
                return 180.0
            elif m == 8:
                return 150.0
            elif m == 9:
                return 100.0
            else:
                return 2.0
        else:
            if m == 6:
                return 80.0
            elif m == 7:
                return 120.0
            elif m == 8:
                return 100.0
            elif m == 9:
                return 80.0
            else:
                return 2.0


def main() -> None:
    """Execute dataset loading, climatology mapping, model training, and serialization pipeline."""
    # 1. Path & Directory Setup
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "mandi_historical_fallback.csv"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Starting Mandi Price Quantile Model Training Pipeline...")
    print("District-Wise Climatology Mapping Engine Enabled")
    print("=" * 60)

    if not data_path.exists():
        raise FileNotFoundError(f"Training dataset not found at '{data_path}'.")

    # 2. Data Cleaning & Header Standardization
    print(f"\n[1/6] Loading raw dataset from: {data_path}")
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    print(f"      Loaded {len(df)} raw rows.")

    rename_mapping = {
        "State": "state",
        "District": "district",
        "Market": "market",
        "Commodity": "commodity",
        "Variety": "variety",
        "Grade": "grade",
        "Arrival_Date": "arrival_date",
        "Min_x0020_Price": "min_price",
        "Max_x0020_Price": "max_price",
        "Modal_x0020_Price": "modal_price",
    }

    for col in list(df.columns):
        col_clean = str(col).strip().replace("_x0020_", " ").replace("_", " ").lower()
        if "modal" in col_clean and "price" in col_clean:
            rename_mapping[col] = "modal_price"
        elif "min" in col_clean and "price" in col_clean:
            rename_mapping[col] = "min_price"
        elif "max" in col_clean and "price" in col_clean:
            rename_mapping[col] = "max_price"

    df = df.rename(columns=rename_mapping)
    print(f"      Standardized headers: {list(df.columns)}")

    df["modal_price"] = pd.to_numeric(df["modal_price"], errors="coerce")
    df = df.dropna(subset=["modal_price"])
    print(f"      Cleaned dataset has {len(df)} valid records with target price.")

    # 3. Feature Engineering (Temporal Extraction & District Climatology Mapping)
    print("\n[2/6] Performing temporal feature extraction...")
    arrival_dates = pd.to_datetime(
        df["arrival_date"], format="%d/%m/%Y", errors="coerce"
    )
    fallback_date = pd.Timestamp("2026-08-22")
    arrival_dates = arrival_dates.fillna(fallback_date)

    df["month"] = arrival_dates.dt.month
    df["day_of_week"] = arrival_dates.dt.dayofweek
    df["day"] = arrival_dates.dt.day

    print("\n[3/6] Mapping District-Wise Climatology (rainfall_mm)...")
    df["rainfall_mm"] = df.apply(
        lambda r: get_district_rainfall(r["state"], r["district"], r["month"]),
        axis=1,
    )
    print(
        f"      Rainfall stats (mm): min={df['rainfall_mm'].min()}, max={df['rainfall_mm'].max()}, mean={df['rainfall_mm'].mean():.2f}"
    )

    # 4. Categorical Encoding
    print("\n[4/6] Encoding categorical columns using OrdinalEncoder...")
    cat_cols = ["state", "district", "market", "commodity", "variety"]
    for col in cat_cols:
        if col not in df.columns:
            df[col] = "N/A"
        df[col] = df[col].astype(str)

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    df[cat_cols] = encoder.fit_transform(df[cat_cols])
    print(f"      Fitted OrdinalEncoder on categorical features: {cat_cols}")

    # 5. Model Training (Multi-Quantile Pinball Loss)
    print("\n[5/6] Training Multi-Quantile Gradient Boosting Regressors...")
    feature_cols = [
        "state",
        "district",
        "market",
        "commodity",
        "variety",
        "month",
        "day_of_week",
        "day",
        "rainfall_mm",
    ]
    X = df[feature_cols]
    y = df["modal_price"]

    print(
        f"      Feature matrix X shape: {X.shape}, Target vector y shape: {y.shape}"
    )
    print(f"      Features: {feature_cols}")

    print("      -> Training Model 1/3 (p10 Downside Floor, alpha=0.10)...")
    model_p10 = GradientBoostingRegressor(
        loss="quantile", alpha=0.10, n_estimators=100, random_state=42
    )
    model_p10.fit(X, y)

    print("      -> Training Model 2/3 (p50 Expected Median, alpha=0.50)...")
    model_p50 = GradientBoostingRegressor(
        loss="quantile", alpha=0.50, n_estimators=100, random_state=42
    )
    model_p50.fit(X, y)

    print("      -> Training Model 3/3 (p90 Upside Volatility, alpha=0.90)...")
    model_p90 = GradientBoostingRegressor(
        loss="quantile", alpha=0.90, n_estimators=100, random_state=42
    )
    model_p90.fit(X, y)

    # 6. Serialization & Saving
    print("\n[6/6] Saving models and artifacts to 'models/' directory...")
    joblib.dump(model_p10, models_dir / "model_p10.joblib")
    joblib.dump(model_p50, models_dir / "model_p50.joblib")
    joblib.dump(model_p90, models_dir / "model_p90.joblib")
    joblib.dump(encoder, models_dir / "encoder.joblib")
    joblib.dump(feature_cols, models_dir / "feature_cols.joblib")

    print("      [OK] Saved model_p10.joblib")
    print("      [OK] Saved model_p50.joblib")
    print("      [OK] Saved model_p90.joblib")
    print("      [OK] Saved encoder.joblib")
    print("      [OK] Saved feature_cols.joblib")

    print("\n" + "=" * 60)
    print("Multi-Quantile Model Training Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
