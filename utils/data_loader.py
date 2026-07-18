import pandas as pd
import joblib
import streamlit as st

MALAYSIA_PUBLIC_HOLIDAYS = pd.to_datetime([
    "2024-01-01","2024-02-10","2024-02-11","2024-04-10","2024-05-01","2024-06-17",
    "2024-08-31","2024-09-16","2024-12-25",
    "2025-01-01","2025-01-29","2025-01-30","2025-03-31","2025-05-01","2025-06-02",
    "2025-08-31","2025-09-16","2025-12-25",
    "2026-01-01","2026-02-17","2026-02-18","2026-03-21","2026-05-01","2026-06-01",
    "2026-08-31","2026-09-16","2026-12-25",
])


@st.cache_data
def load_raw_data():
    medicine  = pd.read_csv("data/raw/medicine_reference.csv")
    sales     = pd.read_csv("data/raw/medicine_sales.csv", encoding="latin-1")
    sales     = sales[["date", "medicine_id", "quantity_sold", "selling_price", "promotion"]]
    inventory = pd.read_csv("data/raw/pharmacy_inventory.csv")

    sales["date"]              = pd.to_datetime(sales["date"])
    inventory["expiry_date"]   = pd.to_datetime(inventory["expiry_date"])
    inventory["purchase_date"] = pd.to_datetime(inventory["purchase_date"])

    # Calculate derived inventory columns
    today = pd.Timestamp.today().normalize()
    inventory["total_value"]       = inventory["quantity"] * inventory["unit_cost"]
    inventory["days_until_expiry"] = (inventory["expiry_date"] - today).dt.days

    def classify_risk(days):
        if days < 0:
            return "High"
        elif days <= 30:
            return "High"
        elif days <= 90:
            return "Medium"
        else:
            return "Low"

    inventory["risk_status"] = inventory["days_until_expiry"].apply(classify_risk)

    return medicine, sales, inventory


@st.cache_data
def load_merged_sales():
    medicine, sales, _ = load_raw_data()

    df = sales.merge(medicine, on="medicine_id", how="left")
    df["date"] = pd.to_datetime(df["date"])

    # Extract time features from date
    df["day_of_week"]  = df["date"].dt.day_name()
    df["month"]        = df["date"].dt.month
    df["year"]         = df["date"].dt.year
    df["is_weekend"]   = df["date"].dt.weekday.isin([5, 6]).astype(int)
    df["week_number"]  = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"]      = df["date"].dt.quarter
    df["day_of_month"] = df["date"].dt.day

    # Holiday and seasonal features
    df["is_public_holiday"] = df["date"].isin(MALAYSIA_PUBLIC_HOLIDAYS).astype(int)
    df["is_monsoon_season"] = df["date"].dt.month.isin([11, 12, 1, 2, 3]).astype(int)

    df = df.sort_values(["medicine_id", "date"])

    # Lag features
    df["lag_1"]  = df.groupby("medicine_id")["quantity_sold"].shift(1)
    df["lag_7"]  = df.groupby("medicine_id")["quantity_sold"].shift(7)
    df["lag_14"] = df.groupby("medicine_id")["quantity_sold"].shift(14)
    df["lag_28"] = df.groupby("medicine_id")["quantity_sold"].shift(28)

    # Rolling averages
    df["rolling_7"]  = (
        df.groupby("medicine_id")["quantity_sold"]
        .rolling(7).mean().reset_index(level=0, drop=True)
    )
    df["rolling_14"] = (
        df.groupby("medicine_id")["quantity_sold"]
        .rolling(14).mean().reset_index(level=0, drop=True)
    )
    df["rolling_28"] = (
        df.groupby("medicine_id")["quantity_sold"]
        .rolling(28).mean().reset_index(level=0, drop=True)
    )

    # Rolling std
    df["rolling_std_7"] = (
        df.groupby("medicine_id")["quantity_sold"]
        .rolling(7).std().reset_index(level=0, drop=True)
    )

    df = df.dropna()

    # Load encoders
    med_encoder = joblib.load("models/med_encoder.pkl")
    cat_encoder = joblib.load("models/cat_encoder.pkl")

    day_mapping = {
        "Monday": 1, "Tuesday": 2, "Wednesday": 3,
        "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 7
    }

    df["medicine_id_encoded"] = med_encoder.transform(df["medicine_id"])
    df["category_encoded"]    = cat_encoder.transform(df["category"])
    df["day_encoded"]         = df["day_of_week"].map(day_mapping)

    return df


@st.cache_resource
def load_models():
    xgb        = joblib.load("models/xgb_model.pkl")
    rf         = joblib.load("models/rf_model.pkl")
    features   = joblib.load("models/features.pkl")
    comparison = joblib.load("models/model_comparison.pkl")
    xgb_bias   = joblib.load("models/xgb_bias.pkl")
    return xgb, rf, features, comparison, xgb_bias