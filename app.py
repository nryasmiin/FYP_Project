"""
================================================================================
HEALTHCARE INVENTORY & DEMAND PREDICTION SYSTEM
Model Training Script
================================================================================
This script trains three forecasting models (Random Forest, XGBoost, ARIMA)
on historical medicine sales data and selects the best performing model
based on a combined ranking of MAE, RMSE, and R².
================================================================================
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ================================================================================
# SECTION 1: DATA LOADING
# ================================================================================

print("Loading data...")

sales    = pd.read_csv("data/raw/medicine_sales.csv", encoding="latin-1")
medicine = pd.read_csv("data/raw/medicine_reference.csv")

# Keep only relevant sales columns
sales = sales[["date", "medicine_id", "quantity_sold", "selling_price", "promotion"]]
sales["date"] = pd.to_datetime(sales["date"])


# ================================================================================
# SECTION 2: DATA INTEGRATION (MERGE)
# ================================================================================

print("Merging datasets...")

df = sales.merge(medicine, on="medicine_id", how="left")


# ================================================================================
# SECTION 3: TIME-BASED FEATURE EXTRACTION
# ================================================================================

print("Extracting time-based features...")

df["day_of_week"]  = df["date"].dt.day_name()
df["month"]        = df["date"].dt.month
df["year"]         = df["date"].dt.year
df["is_weekend"]   = df["date"].dt.weekday.isin([5, 6]).astype(int)
df["week_number"]  = df["date"].dt.isocalendar().week.astype(int)
df["quarter"]      = df["date"].dt.quarter
df["day_of_month"] = df["date"].dt.day


# ================================================================================
# SECTION 4: ENCODING CATEGORICAL VARIABLES
# ================================================================================

print("Encoding categorical variables...")

med_encoder = LabelEncoder()
cat_encoder = LabelEncoder()
day_mapping = {"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4,
               "Friday": 5, "Saturday": 6, "Sunday": 7}

df["medicine_id_encoded"] = med_encoder.fit_transform(df["medicine_id"])
df["category_encoded"]    = cat_encoder.fit_transform(df["category"])
df["day_encoded"]         = df["day_of_week"].map(day_mapping)


# ================================================================================
# SECTION 5: ADDITIONAL FEATURE ENGINEERING (HOLIDAY & SEASON)
# ================================================================================

print("Engineering holiday and seasonal features...")

MALAYSIA_PUBLIC_HOLIDAYS = [
    "2024-01-01", "2024-02-10", "2024-02-11", "2024-04-10", "2024-05-01", "2024-06-17",
    "2024-08-31", "2024-09-16", "2024-12-25",
    "2025-01-01", "2025-01-29", "2025-01-30", "2025-03-31", "2025-05-01", "2025-06-02",
    "2025-08-31", "2025-09-16", "2025-12-25",
    "2026-01-01", "2026-02-17", "2026-02-18", "2026-03-21", "2026-05-01", "2026-06-01",
    "2026-08-31", "2026-09-16", "2026-12-25",
]

# Public holiday indicator — captures stockpiling / demand shift around holidays
df["is_public_holiday"] = df["date"].isin(pd.to_datetime(MALAYSIA_PUBLIC_HOLIDAYS)).astype(int)

# Monsoon season indicator (Nov-Mar) — proxy for flu/cough/allergy season
df["is_monsoon_season"] = df["date"].dt.month.isin([11, 12, 1, 2, 3]).astype(int)


# ================================================================================
# SECTION 6: LAG FEATURES (HISTORICAL DEMAND)
# ================================================================================

print("Engineering lag features...")

df = df.sort_values(["medicine_id", "date"])

df["lag_1"]  = df.groupby("medicine_id")["quantity_sold"].shift(1)
df["lag_7"]  = df.groupby("medicine_id")["quantity_sold"].shift(7)
df["lag_14"] = df.groupby("medicine_id")["quantity_sold"].shift(14)
df["lag_28"] = df.groupby("medicine_id")["quantity_sold"].shift(28)


# ================================================================================
# SECTION 7: ROLLING AVERAGE FEATURES (DEMAND TREND)
# ================================================================================

print("Engineering rolling average features...")

df["rolling_7"]     = df.groupby("medicine_id")["quantity_sold"].rolling(7).mean().reset_index(level=0, drop=True)
df["rolling_14"]    = df.groupby("medicine_id")["quantity_sold"].rolling(14).mean().reset_index(level=0, drop=True)
df["rolling_28"]    = df.groupby("medicine_id")["quantity_sold"].rolling(28).mean().reset_index(level=0, drop=True)
df["rolling_std_7"] = df.groupby("medicine_id")["quantity_sold"].rolling(7).std().reset_index(level=0, drop=True)

# Drop rows with nulls created by lag/rolling operations (insufficient history)
df = df.dropna().sort_values("date").reset_index(drop=True)


# ================================================================================
# SECTION 8: FEATURE LIST DEFINITION
# ================================================================================

FEATURES = [
    # Identity features
    "medicine_id_encoded", "category_encoded",

    # Time-based features
    "day_encoded", "month", "year", "quarter",
    "day_of_month", "is_weekend", "week_number",

    # Holiday & seasonal features
    "is_public_holiday", "is_monsoon_season",

    # Price & promotion features
    "selling_price", "promotion",

    # Lag features (historical demand)
    "lag_1", "lag_7", "lag_14", "lag_28",

    # Rolling average features (demand trend)
    "rolling_7", "rolling_14", "rolling_28", "rolling_std_7",
]

TARGET = "quantity_sold"

print(f"\nTotal features used: {len(FEATURES)}")


# ================================================================================
# SECTION 9: TRAIN/TEST SPLIT (CHRONOLOGICAL 80/20)
# ================================================================================

split    = int(len(df) * 0.8)
train_df = df.iloc[:split]
test_df  = df.iloc[split:]

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test,  y_test  = test_df[FEATURES],  test_df[TARGET]

print(f"Train size: {len(X_train)} rows")
print(f"Test size:  {len(X_test)} rows")


# ================================================================================
# SECTION 10: MODEL TRAINING — RANDOM FOREST
# ================================================================================

print("\nTraining Random Forest...")

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_mae  = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2   = r2_score(y_test, rf_pred)

print(f"\nRandom Forest")
print(f"  MAE  : {rf_mae:.3f}")
print(f"  RMSE : {rf_rmse:.3f}")
print(f"  R²   : {rf_r2:.3f}")


# ================================================================================
# SECTION 11: MODEL TRAINING — XGBOOST
# ================================================================================

print("\nTraining XGBoost...")

xgb = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.02,
    max_depth=4,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
    eval_metric="mae"
)
xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

xgb_pred_raw = xgb.predict(X_test)
xgb_bias     = float(np.mean(y_test.values - xgb_pred_raw))
xgb_pred     = xgb_pred_raw + xgb_bias

print(f"  XGBoost bias correction: {xgb_bias:+.3f} units")

xgb_mae  = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2   = r2_score(y_test, xgb_pred)

print(f"\nXGBoost")
print(f"  MAE  : {xgb_mae:.3f}")
print(f"  RMSE : {xgb_rmse:.3f}")
print(f"  R²   : {xgb_r2:.3f}")


# ================================================================================
# SECTION 12: MODEL TRAINING — ARIMA (PER MEDICINE)
# ================================================================================

print("\nTraining ARIMA for all medicines...")

arima_true_all = []
arima_pred_all = []

for med_id in sales["medicine_id"].unique():
    med_sales = sales[sales["medicine_id"] == med_id].sort_values("date")
    ts = med_sales["quantity_sold"].values
    if len(ts) < 30:
        continue
    tr = int(len(ts) * 0.8)
    try:
        fit  = ARIMA(ts[:tr], order=(5, 1, 0)).fit()
        pred = fit.forecast(steps=len(ts[tr:]))
        arima_true_all.extend(ts[tr:])
        arima_pred_all.extend(pred)
        print(f"  {med_id} done")
    except Exception as e:
        print(f"  {med_id} failed: {e}")

arima_mae  = mean_absolute_error(arima_true_all, arima_pred_all)
arima_rmse = np.sqrt(mean_squared_error(arima_true_all, arima_pred_all))
arima_r2   = r2_score(arima_true_all, arima_pred_all)

print(f"\nARIMA")
print(f"  MAE  : {arima_mae:.3f}")
print(f"  RMSE : {arima_rmse:.3f}")
print(f"  R²   : {arima_r2:.3f}")


# ================================================================================
# SECTION 13: MODEL COMPARISON & SELECTION
# ================================================================================

print("\n========== MODEL COMPARISON ==========")

comparison = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost", "ARIMA"],
    "MAE":   [round(rf_mae, 3),  round(xgb_mae, 3),  round(arima_mae, 3)],
    "RMSE":  [round(rf_rmse, 3), round(xgb_rmse, 3), round(arima_rmse, 3)],
    "R2":    [round(rf_r2, 3),   round(xgb_r2, 3),   round(arima_r2, 3)],
})
print(comparison.to_string(index=False))

# Combined ranking — lower MAE/RMSE better, higher R2 better
comparison["MAE_rank"]  = comparison["MAE"].rank()
comparison["RMSE_rank"] = comparison["RMSE"].rank()
comparison["R2_rank"]   = comparison["R2"].rank(ascending=False)
comparison["avg_rank"]  = (comparison["MAE_rank"] + comparison["RMSE_rank"] + comparison["R2_rank"]) / 3

best_row  = comparison.loc[comparison["avg_rank"].idxmin()]
best_name = best_row["Model"]

print(f"\nSelected model for deployment: {best_name}")
print(f"  MAE rank: {best_row['MAE_rank']:.0f}, RMSE rank: {best_row['RMSE_rank']:.0f}, R2 rank: {best_row['R2_rank']:.0f}")
print(f"  Avg rank: {best_row['avg_rank']:.2f} (lower is better)")

comparison = comparison.drop(columns=["MAE_rank", "RMSE_rank", "R2_rank", "avg_rank"])


# ================================================================================
# SECTION 14: SAVE TRAINED MODELS & ARTIFACTS
# ================================================================================

joblib.dump(rf,          "models/rf_model.pkl")
joblib.dump(xgb,         "models/xgb_model.pkl")
joblib.dump(xgb_bias,    "models/xgb_bias.pkl")
joblib.dump(med_encoder, "models/med_encoder.pkl")
joblib.dump(cat_encoder, "models/cat_encoder.pkl")
joblib.dump(FEATURES,    "models/features.pkl")
joblib.dump(comparison,  "models/model_comparison.pkl")

print("\nSaved:")
print("  models/rf_model.pkl")
print("  models/xgb_model.pkl")
print("  models/xgb_bias.pkl")
print("  models/med_encoder.pkl")
print("  models/cat_encoder.pkl")
print("  models/features.pkl")
print("  models/model_comparison.pkl")