"""
Mandi Price Quantile Model Training Pipeline (memory-safe version)

Loads all CSV datasets from:

    data/
        ├── data2.csv
        ├── mandi_historical_fallback.csv
        └── data_vegetable_wise/*.csv

The JSON rainfall file is intentionally NOT used.

Models:
    P10 / P50 / P90 quantile regressors
    (HistGradientBoostingRegressor with quantile loss - fast, multi-threaded,
     and memory efficient compared to GradientBoostingRegressor)

Requires: scikit-learn >= 1.1, pandas, numpy, requests, joblib
"""

import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from pandas.api.types import union_categoricals
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder


# ============================================================
# CONFIG
# ============================================================

# 30M rows is far more than these models need and will not fit in RAM.
# Each CSV is randomly down-sampled to at most this many valid rows.
# Raise it if you have lots of RAM (e.g. 1_000_000), lower it if you still run out.
MAX_ROWS_PER_FILE = 400_000

# Set e.g. MIN_YEAR = 2020 to only train on recent prices (None = keep all years).
MIN_YEAR = None

RANDOM_STATE = 42
HOLDOUT_FRACTION = 0.05

FALLBACK_RAINFALL = 288.99
FALLBACK_DATE = pd.Timestamp("2026-08-22")
RAINFALL_YEAR = 2024

CATEGORICAL_COLUMNS = ["state", "district", "market", "commodity", "variety"]

# NOTE: "month" is new (it was computed in the old script but never used as a
# feature). Your inference code must now pass "month" too. If you want the old
# 6-feature model, just delete "month" from this list.
FEATURE_COLS = CATEGORICAL_COLUMNS + ["month", "rainfall_mm"]

QUANTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


# ============================================================
# PROJECT ROOT
# ============================================================

def find_project_root() -> Path:
    """Locate the project root containing the data/ directory."""

    current_file = Path(__file__).resolve()

    possible_roots = [
        current_file.parent.parent,
        current_file.parent,
        Path.cwd(),
    ]

    for root in possible_roots:
        if (root / "data").is_dir():
            return root

    raise FileNotFoundError(
        "Could not locate project root containing the 'data' folder."
    )


# ============================================================
# RAINFALL (Open-Meteo, cached on disk)
# ============================================================

# key: "district|state" (lowercase)  ->  {month(int): rainfall_mm(float)}
RAINFALL_CACHE: dict = {}
RAINFALL_CACHE_PATH: Path | None = None


def load_rainfall_cache(path: Path) -> None:
    global RAINFALL_CACHE, RAINFALL_CACHE_PATH
    RAINFALL_CACHE_PATH = path

    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            RAINFALL_CACHE = {
                key: {int(m): float(v) for m, v in months.items()}
                for key, months in raw.items()
            }
            print(f"[OK] Loaded rainfall cache: {len(RAINFALL_CACHE):,} districts")
        except Exception as error:
            print(f"[WARNING] Could not read rainfall cache: {error}")
            RAINFALL_CACHE = {}


def save_rainfall_cache() -> None:
    if RAINFALL_CACHE_PATH is None:
        return
    RAINFALL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAINFALL_CACHE_PATH.write_text(
        json.dumps(RAINFALL_CACHE), encoding="utf-8"
    )


def _get_json(url: str, params: dict, retries: int = 3) -> dict:
    """GET with retry / backoff (handles Open-Meteo 429 rate limits)."""

    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=15)

            if response.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue

            response.raise_for_status()
            return response.json()

        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(1 + attempt)

    raise RuntimeError("Open-Meteo rate limit: retries exhausted")


def _geocode(name: str):
    if not name:
        return None

    data = _get_json(
        GEOCODE_URL,
        {"name": name, "count": 10},
    )
    results = data.get("results")

    if results:
        for r in results:
            if r.get("country_code", "").upper() == "IN" or r.get("country", "").lower() == "india":
                return r["latitude"], r["longitude"]
        return results[0]["latitude"], results[0]["longitude"]
    return None


def _fetch_monthly_rainfall(state: str, district: str):
    """One geocode + one archive call -> rainfall for all 12 months."""

    query = (
        district
        .replace("(Calicut)", "")
        .replace("(Common)", "")
        .replace(" APMC", "")
        .strip()
    )

    coords = _geocode(query) or _geocode(state)

    if coords is None:
        return None

    latitude, longitude = coords

    data = _get_json(
        ARCHIVE_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": f"{RAINFALL_YEAR}-01-01",
            "end_date": f"{RAINFALL_YEAR}-12-31",
            "daily": "precipitation_sum",
            "timezone": "auto",
        },
    )

    daily = data.get("daily", {})
    times = daily.get("time", [])
    values = daily.get("precipitation_sum", [])

    if not times:
        return None

    series = pd.Series(
        values, index=pd.to_datetime(times), dtype="float64"
    )

    # Keep the original definition: sum of days 1-28 of each month
    series = series[series.index.day <= 28]

    monthly = series.groupby(series.index.month).sum()

    return {int(m): round(float(v), 2) for m, v in monthly.items()}


def get_district_monthly_rainfall(state: str, district: str) -> dict:
    """Return {month: rainfall_mm} for a district (empty dict on failure)."""

    key = f"{str(district).strip()}|{str(state).strip()}".lower()

    if key in RAINFALL_CACHE:
        return RAINFALL_CACHE[key]

    try:
        monthly = _fetch_monthly_rainfall(str(state).strip(), str(district).strip())
        time.sleep(0.1)  # be polite to the free API
    except Exception as error:
        print(f"[WARNING] Rainfall lookup failed for {district}, {state}: {error}")
        monthly = None

    if monthly:
        RAINFALL_CACHE[key] = monthly  # only successful lookups are cached
        return monthly

    return {}


def get_district_rainfall(state: str, district: str, month: int = 8) -> float:
    """Backwards-compatible single-value helper (same signature as before)."""

    try:
        month_num = int(month)
    except Exception:
        month_num = 8

    monthly = get_district_monthly_rainfall(state, district)
    return monthly.get(month_num, FALLBACK_RAINFALL)


# ============================================================
# COLUMN STANDARDIZATION (per file, memory-reduced)
# ============================================================

def standardize_dataset(
    df: pd.DataFrame,
    source_file: str,
    rng: np.random.Generator,
):
    """
    Standardize ONE dataset and shrink it immediately:
      - only the columns needed for training are kept
      - invalid prices are dropped
      - rows are down-sampled to MAX_ROWS_PER_FILE
      - strings become 'category', numbers become float32 / int8

    Returns None if the file has no usable rows.
    """

    # ---- clean column names ------------------------------------------------
    df.columns = [
        str(c)
        .replace("_x0020_", " ")
        .replace("_x003a_", ":")
        .replace("\ufeff", "")
        .strip()
        for c in df.columns
    ]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]

    def find_columns(condition):
        return [c for c in df.columns if condition(str(c).strip().lower())]

    def combine_columns(columns):
        if not columns:
            return pd.Series([None] * len(df), index=df.index, dtype="object")

        result = df[columns[0]].copy()
        for column in columns[1:]:
            result = result.combine_first(df[column])
        return result

    state = combine_columns(find_columns(
        lambda x: x in {"state", "state name", "statename"}))

    district = combine_columns(find_columns(
        lambda x: x in {"district", "district name", "districtname"}))

    market = combine_columns(find_columns(
        lambda x: x in {"market", "market name", "marketname"}))

    commodity = combine_columns(find_columns(
        lambda x: x in {"commodity", "commodity name", "commodityname"}
        or "commodity name" in x))

    variety = combine_columns(find_columns(lambda x: "variety" in x))

    arrival_date = combine_columns(find_columns(
        lambda x: x in {
            "arrival date", "arrival_date", "arrivaldate",
            "reported date", "reported_date", "reporteddate", "date"}
        or ("arrival" in x and "date" in x)
        or ("reported" in x and "date" in x)))

    modal_price = combine_columns(find_columns(
        lambda x: ("modal" in x and "price" in x)
        or x in {"modal price", "modal_price", "modalprice"}))

    # ---- target + date, then filter BEFORE doing any string work -----------
    modal = pd.to_numeric(modal_price, errors="coerce")
    dates = pd.to_datetime(arrival_date, errors="coerce", dayfirst=True)

    keep = modal.notna() & (modal > 0)

    if MIN_YEAR is not None:
        keep &= dates.isna() | (dates.dt.year >= MIN_YEAR)

    idx = np.flatnonzero(keep.to_numpy())

    if len(idx) == 0:
        return None

    if len(idx) > MAX_ROWS_PER_FILE:
        idx = np.sort(rng.choice(idx, MAX_ROWS_PER_FILE, replace=False))

    # ---- take the sampled rows only ----------------------------------------
    dates_sub = dates.iloc[idx]
    bad_dates = int(dates_sub.isna().sum())
    dates_sub = dates_sub.fillna(FALLBACK_DATE)

    strings = {}

    for name, series in (
        ("state", state),
        ("district", district),
        ("market", market),
        ("commodity", commodity),
        ("variety", variety),
    ):
        s = series.iloc[idx].fillna("N/A").astype(str).str.strip()
        s = s.mask(s.isin(["", "nan", "NaN", "None"]), "N/A")
        strings[name] = s.reset_index(drop=True)

    # Vegetable files without a commodity column -> use the file name
    file_commodity = Path(source_file).stem
    strings["commodity"] = strings["commodity"].mask(
        strings["commodity"] == "N/A", file_commodity
    )

    frame = pd.DataFrame(
        {name: s.astype("category") for name, s in strings.items()}
    )
    frame["modal_price"] = modal.iloc[idx].to_numpy(dtype="float32")
    frame["month"] = dates_sub.dt.month.to_numpy(dtype="int8")

    frame.attrs["bad_dates"] = bad_dates
    return frame


def combine_frames(frames: list) -> pd.DataFrame:
    """Concatenate without turning categoricals back into python strings."""

    data = {}

    for column in CATEGORICAL_COLUMNS:
        data[column] = union_categoricals(
            [f[column] for f in frames], sort_categories=True
        )

    data["modal_price"] = np.concatenate(
        [f["modal_price"].to_numpy() for f in frames]
    )
    data["month"] = np.concatenate(
        [f["month"].to_numpy() for f in frames]
    )

    return pd.DataFrame(data)


# ============================================================
# LOAD ALL CSV DATASETS
# ============================================================

def load_all_datasets(data_dir: Path) -> pd.DataFrame:
    """Recursively load every CSV, shrinking each one as it is read."""

    print("\n" + "=" * 60)
    print("DATASET DISCOVERY")
    print("=" * 60)

    csv_files = sorted(data_dir.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found inside: {data_dir}")

    print(f"[OK] Found {len(csv_files)} CSV files.")
    print(f"[OK] Sampling cap: {MAX_ROWS_PER_FILE:,} rows per file")

    rng = np.random.default_rng(RANDOM_STATE)
    frames = []
    total_raw = 0
    total_bad_dates = 0

    for file in csv_files:
        relative_path = str(file.relative_to(data_dir))

        try:
            raw_df = pd.read_csv(file, encoding="utf-8-sig", low_memory=False)
            raw_rows = len(raw_df)
            total_raw += raw_rows

            clean_df = standardize_dataset(raw_df, relative_path, rng)

            del raw_df
            gc.collect()

            if clean_df is None:
                print(f"[SKIPPED] {relative_path}: no valid price rows")
                continue

            total_bad_dates += clean_df.attrs.get("bad_dates", 0)

            print(
                f"[LOADED] {relative_path}: {raw_rows:,} raw "
                f"-> {len(clean_df):,} kept"
            )
            frames.append(clean_df)

        except Exception as error:
            print(f"[WARNING] Could not load {relative_path}: {error}")

    if not frames:
        raise RuntimeError("No CSV datasets were successfully loaded.")

    combined = combine_frames(frames)

    del frames
    gc.collect()

    print("\n" + "-" * 60)
    print(f"[OK] Raw rows scanned      : {total_raw:,}")
    print(f"[OK] Rows used for training: {len(combined):,}")
    print(f"[OK] Rows with unparseable date (month set to fallback): {total_bad_dates:,}")
    print(f"[OK] Memory usage          : {combined.memory_usage(deep=True).sum() / 1e6:,.0f} MB")
    print("-" * 60)

    return combined


# ============================================================
# RAINFALL FEATURE (vectorised - no row-wise apply)
# ============================================================

def add_rainfall_feature(df: pd.DataFrame) -> pd.DataFrame:
    print("\nMapping District-Wise Climatology (rainfall_mm)...")

    n_districts = len(df["district"].cat.categories)

    state_codes = df["state"].cat.codes.to_numpy().astype("int64")
    district_codes = df["district"].cat.codes.to_numpy().astype("int64")
    months = df["month"].to_numpy().astype("int64")

    pair_key = state_codes * n_districts + district_codes
    full_key = pair_key * 13 + months

    unique_keys, inverse = np.unique(full_key, return_inverse=True)
    unique_pairs = np.unique(unique_keys // 13)

    print(f"Unique state/district pairs      : {len(unique_pairs):,}")
    print(f"Unique state/district/month combos: {len(unique_keys):,}")

    state_names = df["state"].cat.categories
    district_names = df["district"].cat.categories

    monthly_lookup = {}

    for i, pk in enumerate(unique_pairs):
        if i % 50 == 0:
            print(f"    Rainfall lookup: {i:,}/{len(unique_pairs):,}")
            save_rainfall_cache()  # resumable if you Ctrl+C

        state_name = state_names[pk // n_districts]
        district_name = district_names[pk % n_districts]

        monthly_lookup[pk] = get_district_monthly_rainfall(
            state_name, district_name
        )

    save_rainfall_cache()

    rainfall_unique = np.array(
        [
            monthly_lookup[k // 13].get(int(k % 13), FALLBACK_RAINFALL)
            for k in unique_keys
        ],
        dtype="float32",
    )

    df["rainfall_mm"] = rainfall_unique[inverse]

    failed = sum(1 for v in monthly_lookup.values() if not v)
    print(f"Districts using fallback rainfall: {failed:,}")

    print("\nRainfall statistics:")
    print(f"    Minimum : {df['rainfall_mm'].min():.2f} mm")
    print(f"    Maximum : {df['rainfall_mm'].max():.2f} mm")
    print(f"    Mean    : {df['rainfall_mm'].mean():.2f} mm")

    return df


# ============================================================
# EVALUATION HELPERS
# ============================================================

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    project_root = find_project_root()
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    load_rainfall_cache(project_root / "cache" / "rainfall_cache.json")

    print("=" * 60)
    print("Starting Mandi Price Quantile Model Training Pipeline...")
    print("=" * 60)
    print(f"\nProject root  : {project_root}")
    print(f"Data directory: {data_dir}")

    # ---- 1. load ------------------------------------------------------------
    print("\n[1/5] Loading all CSV datasets...")
    df = load_all_datasets(data_dir)

    # ---- 2. rainfall --------------------------------------------------------
    print("\n[2/5] Rainfall feature...")
    df = add_rainfall_feature(df)

    # ---- 3. encoding --------------------------------------------------------
    print("\n[3/5] Encoding categorical columns...")

    # Encoder built straight from the category vocabularies (no 30M-row fit).
    encoder = OrdinalEncoder(
        categories=[list(df[c].cat.categories) for c in CATEGORICAL_COLUMNS],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    encoder.fit(df[CATEGORICAL_COLUMNS].head(1).astype(str))

    X = pd.DataFrame(
        {c: df[c].cat.codes.to_numpy().astype("float32") for c in CATEGORICAL_COLUMNS}
    )
    X["month"] = df["month"].to_numpy().astype("float32")
    X["rainfall_mm"] = df["rainfall_mm"].to_numpy().astype("float32")
    X = X[FEATURE_COLS]

    y = df["modal_price"].to_numpy().astype("float32")

    n_rows = len(df)
    del df
    gc.collect()

    print(f"Feature matrix X shape: {X.shape}")
    print(f"Target vector y shape : {y.shape}")
    print(f"Features              : {FEATURE_COLS}")

    # ---- 4. train -----------------------------------------------------------
    print("\n[4/5] Training quantile models...")

    rng = np.random.default_rng(RANDOM_STATE)
    holdout = rng.random(n_rows) < HOLDOUT_FRACTION

    X_train, y_train = X[~holdout], y[~holdout]
    X_test, y_test = X[holdout], y[holdout]

    print(f"Train rows: {len(X_train):,} | Holdout rows: {len(X_test):,}")

    models = {}

    for name, q in QUANTILES.items():
        print(f"\nTraining {name.upper()} (quantile={q})...")

        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=q,
            max_iter=300,
            learning_rate=0.1,
            max_leaf_nodes=63,
            min_samples_leaf=50,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        coverage = float(np.mean(y_test <= pred))

        print(
            f"[OK] {name.upper()} trained in {model.n_iter_} iterations | "
            f"holdout pinball loss: {pinball_loss(y_test, pred, q):.2f} | "
            f"coverage: {coverage:.3f} (target {q:.2f})"
        )

        models[name] = model

    # ---- 5. save ------------------------------------------------------------
    print("\n[5/5] Saving models and artifacts...")

    for name, model in models.items():
        joblib.dump(model, models_dir / f"model_{name}.joblib")
        print(f"[OK] Saved model_{name}.joblib")

    joblib.dump(encoder, models_dir / "encoder.joblib")
    print("[OK] Saved encoder.joblib")

    joblib.dump(FEATURE_COLS, models_dir / "feature_cols.joblib")
    print("[OK] Saved feature_cols.joblib")

    print("\n" + "=" * 60)
    print("Multi-Quantile Model Training Completed Successfully!")
    print("=" * 60)
    print(f"Total training records: {n_rows:,}")
    print(f"Total features        : {len(FEATURE_COLS)}")
    print(f"Models directory      : {models_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()