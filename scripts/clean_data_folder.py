"""Script to clean and purge unnecessary and null data from data/ folder.

1. Removes all CSV files in data/data_vegetable_wise/ for commodities not used in the project.
2. Filters null, zero-price, and missing-location records from the active crop CSV files.
3. Cleans data/mandi_historical_fallback.csv and data/data2.csv to retain only clean, valid active crop records.
"""

import os
import sys
import glob
import time
import pandas as pd
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


ACTIVE_CROPS = [
    "wheat", "potato", "onion", "tomato", "green chilli", "brinjal",
    "cauliflower", "cabbage", "bhindi", "carrot", "paddy", "rice",
    "maize", "bajra", "mustard", "soyabean", "cotton", "bengal gram",
    "banana", "apple"
]

EXCLUDE_STEM_SUBSTRINGS = [
    "pineapple", "custard apple", "sweet potato"
]


def is_active_crop_file(filename: str) -> bool:
    """Determine if a filename belongs to one of our active crops."""
    stem = os.path.splitext(filename)[0].lower()
    for excl in EXCLUDE_STEM_SUBSTRINGS:
        if excl in stem:
            return False
    for crop in ACTIVE_CROPS:
        if crop in stem:
            return True
    return False


def clean_csv_file(file_path: Path) -> dict:
    """Filter out null, NaN, and <=0 modal prices from an active CSV file."""
    t0 = time.time()
    orig_size = file_path.stat().st_size
    temp_path = file_path.with_suffix(".tmp")

    first_chunk = True
    total_in = 0
    total_out = 0

    try:
        for chunk in pd.read_csv(file_path, chunksize=100000, low_memory=False):
            total_in += len(chunk)

            # Detect price column
            col_modal = None
            for c in chunk.columns:
                if "modal" in c.lower():
                    col_modal = c
                    break

            col_state = None
            for c in chunk.columns:
                if "state" in c.lower():
                    col_state = c
                    break

            col_market = None
            for c in chunk.columns:
                if "market" in c.lower():
                    col_market = c
                    break

            mask = pd.Series(True, index=chunk.index)
            if col_modal:
                p_num = pd.to_numeric(chunk[col_modal], errors="coerce")
                mask = mask & (p_num > 0)
            if col_state:
                mask = mask & chunk[col_state].notna() & (chunk[col_state].astype(str).str.strip() != "")
            if col_market:
                mask = mask & chunk[col_market].notna() & (chunk[col_market].astype(str).str.strip() != "")

            chunk_clean = chunk[mask]
            total_out += len(chunk_clean)

            chunk_clean.to_csv(temp_path, mode="w" if first_chunk else "a", header=first_chunk, index=False)
            first_chunk = False

        if temp_path.exists():
            temp_path.replace(file_path)

        new_size = file_path.stat().st_size
        return {
            "file": file_path.name,
            "rows_in": total_in,
            "rows_out": total_out,
            "rows_dropped": total_in - total_out,
            "orig_mb": orig_size / (1024 * 1024),
            "new_mb": new_size / (1024 * 1024),
            "time_sec": time.time() - t0,
        }
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()
        print(f"  [ERROR] Failed to clean {file_path.name}: {exc}")
        return {
            "file": file_path.name,
            "rows_in": total_in,
            "rows_out": total_in,
            "rows_dropped": 0,
            "orig_mb": orig_size / (1024 * 1024),
            "new_mb": orig_size / (1024 * 1024),
            "time_sec": time.time() - t0,
        }


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    veg_dir = data_dir / "data_vegetable_wise"

    print("=" * 70)
    print("🌾 K.I.S.A.N. AI DATA FOLDER CLEANING & PURGE PIPELINE")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # STEP 1: Purge Unused CSV Files in data/data_vegetable_wise/
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Scanning data/data_vegetable_wise/ for unused commodity datasets...")
    all_veg_files = list(veg_dir.glob("*.csv"))
    print(f"         Total files found: {len(all_veg_files)}")

    to_keep = []
    to_delete = []

    for f in all_veg_files:
        if is_active_crop_file(f.name):
            to_keep.append(f)
        else:
            to_delete.append(f)

    space_freed_bytes = sum(f.stat().st_size for f in to_delete)
    print(f"         Active crop datasets to KEEP: {len(to_keep)}")
    print(f"         Unused commodity datasets to DELETE: {len(to_delete)} ({space_freed_bytes / (1024*1024):.1f} MB / {space_freed_bytes / (1024*1024*1024):.2f} GB)")

    for f in to_delete:
        try:
            f.unlink()
        except Exception as exc:
            print(f"         Failed to delete {f.name}: {exc}")

    print(f"         [OK] Deleted {len(to_delete)} unused commodity files!")

    # -------------------------------------------------------------------------
    # STEP 2: Clean Null, Zero-Price, and Missing-Location Data from Active CSVs
    # -------------------------------------------------------------------------
    print(f"\n[STEP 2] Cleaning null, zero-price, and invalid records from {len(to_keep)} active datasets...")
    total_rows_dropped = 0
    for idx, f in enumerate(to_keep, start=1):
        res = clean_csv_file(f)
        total_rows_dropped += res["rows_dropped"]
        print(f"         [{idx:02d}/{len(to_keep):02d}] {res['file']:<32} | {res['rows_in']:>8,} -> {res['rows_out']:>8,} rows (dropped {res['rows_dropped']:>6,} invalid) in {res['time_sec']:.1f}s")

    print(f"         [OK] Total invalid/null records purged from active crop datasets: {total_rows_dropped:,}")

    # -------------------------------------------------------------------------
    # STEP 3: Clean data/mandi_historical_fallback.csv
    # -------------------------------------------------------------------------
    fb_path = data_dir / "mandi_historical_fallback.csv"
    if fb_path.exists():
        print(f"\n[STEP 3] Cleaning {fb_path.name}...")
        df_fb = pd.read_csv(fb_path)
        orig_fb_len = len(df_fb)

        mask_fb = df_fb["Commodity"].astype(str).str.lower().apply(
            lambda x: any(k in x for k in ACTIVE_CROPS) and not any(excl in x for excl in EXCLUDE_STEM_SUBSTRINGS)
        )
        df_fb_clean = df_fb[mask_fb].copy()
        p_fb = pd.to_numeric(df_fb_clean["Modal_x0020_Price"], errors="coerce")
        df_fb_clean = df_fb_clean[p_fb > 0]
        df_fb_clean = df_fb_clean.dropna(subset=["State", "Market", "Commodity"])
        df_fb_clean = df_fb_clean.drop_duplicates()

        df_fb_clean.to_csv(fb_path, index=False)
        print(f"         Original: {orig_fb_len:,} rows -> Cleaned: {len(df_fb_clean):,} rows (removed {orig_fb_len - len(df_fb_clean):,} unused/null records)")

    # -------------------------------------------------------------------------
    # STEP 4: Clean data/data2.csv
    # -------------------------------------------------------------------------
    d2_path = data_dir / "data2.csv"
    if d2_path.exists():
        print(f"\n[STEP 4] Cleaning {d2_path.name}...")
        df_d2 = pd.read_csv(d2_path)
        orig_d2_len = len(df_d2)

        mask_d2 = df_d2["commodity"].astype(str).str.lower().apply(
            lambda x: any(k in x for k in ACTIVE_CROPS) and not any(excl in x for excl in EXCLUDE_STEM_SUBSTRINGS)
        )
        df_d2_clean = df_d2[mask_d2].copy()
        p_d2 = pd.to_numeric(df_d2_clean["modal_price"], errors="coerce")
        df_d2_clean = df_d2_clean[p_d2 > 0]
        df_d2_clean = df_d2_clean.dropna(subset=["state", "market", "commodity"])
        df_d2_clean = df_d2_clean.drop_duplicates()

        df_d2_clean.to_csv(d2_path, index=False)
        print(f"         Original: {orig_d2_len:,} rows -> Cleaned: {len(df_d2_clean):,} rows (removed {orig_d2_len - len(df_d2_clean):,} unused/null records)")

    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("✅ DATA CLEANING & PURGE COMPLETED SUCCESSFULLY!")
    print(f"   • Unused files removed: {len(to_delete)} files")
    print(f"   • Active files retained & cleaned: {len(to_keep)} files")
    print(f"   • Storage reclaimed: {space_freed_bytes / (1024*1024):.1f} MB (~{space_freed_bytes / (1024*1024*1024):.2f} GB)")
    print(f"   • Corrupted / zero-price / null rows purged: {total_rows_dropped:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
