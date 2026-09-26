"""Generate publication-ready IEEE model evaluation plots and save to evaluation_results/."""

import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

project_root = Path(__file__).resolve().parent.parent
data_file = project_root / "data" / "mandi_historical_fallback.csv"
models_dir = project_root / "models"
out_dir = project_root / "evaluation_results"
out_dir.mkdir(parents=True, exist_ok=True)

print("[1/5] Loading data and trained models...")
df = pd.read_csv(data_file)
encoder = joblib.load(models_dir / "encoder.joblib")
model_p10 = joblib.load(models_dir / "model_p10.joblib")
model_p50 = joblib.load(models_dir / "model_p50.joblib")
model_p90 = joblib.load(models_dir / "model_p90.joblib")
feature_cols = joblib.load(models_dir / "feature_cols.joblib")

# Clean & normalize headers to lowercase snake_case
df.columns = [c.strip().lower().replace("_x0020_", "_").replace(" ", "_") for c in df.columns]
df["modal_price"] = pd.to_numeric(df["modal_price"], errors="coerce")
df = df.dropna(subset=["modal_price"])

# Temporal features
dates = pd.to_datetime(df["arrival_date"], format="%d/%m/%Y", errors="coerce").fillna(pd.Timestamp("2026-08-22"))
df["month"] = dates.dt.month
df["day_of_week"] = dates.dt.dayofweek
df["day"] = dates.dt.day

# Rainfall mapping from cache
cache_path = project_root / "data" / "district_rainfall_realtime.json"
import json
with open(cache_path, "r", encoding="utf-8") as f:
    rf_cache = json.load(f)

def get_rf(d):
    k = str(d).strip().lower()
    return float(rf_cache.get(k, {}).get("rainfall_mm", 288.99))

df["rainfall_mm"] = df["district"].apply(get_rf)

cat_cols = ["state", "district", "market", "commodity", "variety"]
for c in cat_cols:
    df[c] = df[c].astype(str)

df[cat_cols] = encoder.transform(df[cat_cols])

X = df[feature_cols]
y = df["modal_price"]

print(f"[2/5] Evaluating on test split (Dataset size: {len(X)} rows)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

pred_p10 = model_p10.predict(X_test)
pred_p50 = model_p50.predict(X_test)
pred_p90 = model_p90.predict(X_test)

# Calculate metrics
mae = np.mean(np.abs(y_test - pred_p50))
rmse = np.sqrt(np.mean((y_test - pred_p50)**2))
ss_res = np.sum((y_test - pred_p50)**2)
ss_tot = np.sum((y_test - np.mean(y_test))**2)
r2_score = 1 - (ss_res / ss_tot)
picp = np.mean((y_test >= pred_p10) & (y_test <= pred_p90)) * 100.0

print(f"[3/5] Computed metrics: MAE=Rs.{mae:.2f}, RMSE=Rs.{rmse:.2f}, R2={r2_score:.4f}, PICP={picp:.2f}%")

print("[4/5] Rendering publication-quality IEEE composite plot (300 DPI)...")
fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# 1. Actual vs Predicted (p50)
axes[0, 0].scatter(y_test, pred_p50, alpha=0.5, color='#0284c7', s=24, edgecolors='none', label='Test Observations')
min_v, max_v = min(float(y_test.min()), float(pred_p50.min())), max(float(y_test.max()), float(pred_p50.max()))
axes[0, 0].plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Ideal Parity (y = x)')
axes[0, 0].set_xlabel('Actual Mandi Price (INR / Quintal)', fontweight='bold')
axes[0, 0].set_ylabel('Predicted Median Price p50 (INR / Quintal)', fontweight='bold')
axes[0, 0].set_title(f'(a) Parity Plot: Actual vs Predicted p50 ($R^2$ = {r2_score:.3f})', fontweight='bold', fontsize=12)
axes[0, 0].grid(True, linestyle=':', alpha=0.6)
axes[0, 0].legend(frameon=True)

# 2. Quantile Uncertainty Envelope Ribbon
sample_n = 50
sort_indices = np.argsort(y_test.values[:sample_n])
sorted_actual = y_test.values[:sample_n][sort_indices]
sorted_p10 = pred_p10[:sample_n][sort_indices]
sorted_p50 = pred_p50[:sample_n][sort_indices]
sorted_p90 = pred_p90[:sample_n][sort_indices]

axes[0, 1].plot(range(sample_n), sorted_actual, 'o-', color='#16a34a', markersize=4, linewidth=1.5, label='Actual Mandi Price')
axes[0, 1].plot(range(sample_n), sorted_p50, '--', color='#2563eb', linewidth=1.5, label='Predicted p50 Median')
axes[0, 1].fill_between(range(sample_n), sorted_p10, sorted_p90, color='#38bdf8', alpha=0.35, label=f'80% Prediction Band [p10, p90] (PICP={picp:.1f}%)')
axes[0, 1].set_xlabel('Sorted Commodity Samples', fontweight='bold')
axes[0, 1].set_ylabel('Price (INR / Quintal)', fontweight='bold')
axes[0, 1].set_title('(b) Multi-Quantile Uncertainty Envelope [p10, p50, p90]', fontweight='bold', fontsize=12)
axes[0, 1].grid(True, linestyle=':', alpha=0.6)
axes[0, 1].legend(frameon=True)

# 3. Feature Importance Ranking
importances = model_p50.feature_importances_
relative_imp = (importances / importances.sum()) * 100.0
fi_df = pd.DataFrame({'Feature': feature_cols, 'Importance': relative_imp}).sort_values('Importance', ascending=True)

colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(fi_df)))
axes[1, 0].barh(fi_df['Feature'], fi_df['Importance'], color=colors, edgecolor='none', height=0.65)
axes[1, 0].set_xlabel('Relative Gini Importance (%)', fontweight='bold')
axes[1, 0].set_title('(c) Feature Importance Ranking (MDI)', fontweight='bold', fontsize=12)
axes[1, 0].grid(axis='x', linestyle=':', alpha=0.6)
for i, v in enumerate(fi_df['Importance']):
    axes[1, 0].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9, fontweight='semibold')

# 4. Residual Distribution
residuals = y_test - pred_p50
sns.histplot(residuals, kde=True, color='#8b5cf6', bins=30, stat='density', ax=axes[1, 1], alpha=0.45)
axes[1, 1].axvline(0, color='#ef4444', linestyle='--', linewidth=2, label='Zero Error Benchmark')
axes[1, 1].set_xlabel('Residual Error: Actual - Predicted (INR/Quintal)', fontweight='bold')
axes[1, 1].set_ylabel('Probability Density', fontweight='bold')
axes[1, 1].set_title(f'(d) Residual Error Distribution (MAE = ₹{mae:.0f})', fontweight='bold', fontsize=12)
axes[1, 1].grid(True, linestyle=':', alpha=0.6)
axes[1, 1].legend(frameon=True)

plt.tight_layout()
out_plot_path = out_dir / "ieee_evaluation_plots.png"
fig.savefig(out_plot_path, dpi=300)
plt.close(fig)

print(f"[5/5] Successfully saved publication plot to: {out_plot_path}")
