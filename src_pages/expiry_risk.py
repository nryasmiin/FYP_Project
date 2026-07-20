import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_raw_data, load_merged_sales, load_models


MALAYSIA_PUBLIC_HOLIDAYS = pd.to_datetime([
    "2024-01-01","2024-02-10","2024-02-11","2024-04-10","2024-05-01","2024-06-17",
    "2024-08-31","2024-09-16","2024-12-25",
    "2025-01-01","2025-01-29","2025-01-30","2025-03-31","2025-05-01","2025-06-02",
    "2025-08-31","2025-09-16","2025-12-25",
    "2026-01-01","2026-02-17","2026-02-18","2026-03-21","2026-05-01","2026-06-01",
    "2026-08-31","2026-09-16","2026-12-25",
])


def inject_styles():
    st.markdown("""
    <style>
    .chart-card {
        background-color: rgba(99, 102, 241, 0.08);
        border-radius: 16px;
        padding: 14px 16px;
        border: 1px solid rgba(99, 102, 241, 0.15);
        margin-bottom: 16px;
    }
    .chart-title {
        font-size: 18px;
        font-weight: 600;
        color: #000000;
        margin-bottom: 10px;
        text-align: center;
    }
    .kpi-card {
        background-color: rgba(99, 102, 241, 0.08);
        border-radius: 16px;
        padding: 24px 20px;
        text-align: center;
        border: 1px solid rgba(99, 102, 241, 0.15);
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .kpi-label {
        font-size: 13px;
        color: #4f46e5;
        font-weight: 500;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #000000;
    }
    .kpi-delta-bad {
        font-size: 11px;
        color: #ef4444;
        margin-top: 4px;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def forecast_demand_30days(med_id, df, features, _xgb_model, xgb_bias):
    med_df = df[df["medicine_id"] == med_id].sort_values("date")
    if med_df.empty:
        return 0

    last_row     = med_df.iloc[-1].copy()
    last_date    = last_row["date"]
    recent_sales = med_df["quantity_sold"].values.tolist()

    recent_30d = med_df.tail(30)
    last_selling_price = recent_30d["selling_price"].mean() if "selling_price" in med_df.columns else 0
    last_promotion      = recent_30d["promotion"].mean() if "promotion" in med_df.columns else 0

    total = 0

    for i in range(1, 31):
        future_date   = last_date + pd.Timedelta(days=i)
        lag_1         = recent_sales[-1]
        lag_7         = recent_sales[-7]  if len(recent_sales) >= 7  else np.mean(recent_sales)
        lag_14        = recent_sales[-14] if len(recent_sales) >= 14 else np.mean(recent_sales)
        lag_28        = recent_sales[-28] if len(recent_sales) >= 28 else np.mean(recent_sales)
        rolling_7     = np.mean(recent_sales[-7:])  if len(recent_sales) >= 7  else np.mean(recent_sales)
        rolling_14    = np.mean(recent_sales[-14:]) if len(recent_sales) >= 14 else np.mean(recent_sales)
        rolling_28    = np.mean(recent_sales[-28:]) if len(recent_sales) >= 28 else np.mean(recent_sales)
        rolling_std_7 = np.std(recent_sales[-7:])   if len(recent_sales) >= 7  else 0

        is_public_holiday = int(future_date.normalize() in MALAYSIA_PUBLIC_HOLIDAYS)
        is_monsoon_season  = int(future_date.month in [11, 12, 1, 2, 3])

        row = {
            "medicine_id_encoded": last_row["medicine_id_encoded"],
            "day_encoded":         future_date.isoweekday(),
            "month":               future_date.month,
            "year":                future_date.year,
            "quarter":             (future_date.month - 1) // 3 + 1,
            "day_of_month":        future_date.day,
            "is_weekend":          int(future_date.weekday() >= 5),
            "week_number":         future_date.isocalendar()[1],
            "category_encoded":    last_row["category_encoded"],
            "lag_1":               lag_1,
            "lag_7":               lag_7,
            "lag_14":              lag_14,
            "lag_28":              lag_28,
            "rolling_7":           rolling_7,
            "rolling_14":          rolling_14,
            "rolling_28":          rolling_28,
            "rolling_std_7":       rolling_std_7,
            "selling_price":       last_selling_price,
            "promotion":           last_promotion,
            "is_public_holiday":   is_public_holiday,
            "is_monsoon_season":   is_monsoon_season,
        }

        pred = float(_xgb_model.predict(pd.DataFrame([row])[features])[0]) + xgb_bias
        pred = max(0, pred)
        total += pred
        recent_sales.append(pred)

    return round(total)


def classify_risk_3factor(days_until_expiry, quantity, forecasted_demand_30days):
    if days_until_expiry < 0:
        return "High", "Stock has already expired."

    if days_until_expiry <= 30:
        return "High", "Stock is unsellable — less than 30 days remaining shelf life."

    forecast_covers = forecasted_demand_30days >= quantity

    if days_until_expiry <= 60:
        if not forecast_covers:
            return "High", "Expiring within 31–60 days and forecasted demand is lower than current stock."
        else:
            return "Medium", "Expiring within 31–60 days but forecasted demand is sufficient to clear stock."
    elif days_until_expiry <= 100:
        if not forecast_covers:
            return "Medium", "Expiring within 61–100 days and forecasted demand is lower than current stock."
        else:
            return "Low", "Expiring within 61–100 days and demand will clear stock in time."
    else:
        return "Low", "More than 100 days remaining. Sufficient time to clear stock."


def get_recommendation(risk, days, quantity, forecasted_demand):
    if days < 0:
        return "⛔ Stock has expired. Remove from shelf immediately and dispose according to pharmacy protocol."
    elif days <= 30:
        return (
            f"🚨 Stock is unsellable. {quantity} units have only {days} days remaining — "
            f"insufficient time for patients to complete medication. "
            f"Return to supplier immediately if within return window, "
            f"otherwise dispose according to pharmaceutical waste regulations."
        )
    elif risk == "High":
        shortfall = quantity - forecasted_demand
        return (
            f"🚨 Critical action required. {quantity} units expiring in {days} days. "
            f"Forecasted demand is only {forecasted_demand} units — "
            f"shortfall of {shortfall} units. "
            f"Prioritise dispensing, apply discount promotion, or return to supplier immediately."
        )
    elif risk == "Medium":
        if days <= 60:
            return (
                f"⚠️ Expiring in {days} days. Forecasted demand of {forecasted_demand} units "
                f"is sufficient to cover {quantity} units in stock. "
                f"Maintain current dispensing rate and monitor closely."
            )
        else:
            return (
                f"⚠️ Expiring in {days} days. Monitor stock closely. "
                f"30-day forecasted demand is {forecasted_demand} units against {quantity} units in stock. "
                f"Consider promotional activities to increase dispensing rate."
            )
    else:
        return (
            f"✅ Stock is safe. {quantity} units with {days} days remaining. "
            f"30-day forecasted demand is {forecasted_demand} units. "
            f"Continue regular monitoring."
        )


def show():
    inject_styles()

    st.title("Expiry Risk Assessment")
    st.markdown("Risk classification based on remaining shelf life, stock levels, and forecasted demand.")

    medicine, _, inventory = load_raw_data()
    df                     = load_merged_sales()
    xgb, rf, features, comparison, xgb_bias = load_models()

    inv = inventory.merge(medicine, on="medicine_id", how="left")

    # Cache forecasts per medicine (not per batch) to avoid recalculating
    # the same medicine's demand forecast multiple times when it has
    # several inventory batches.
    forecast_cache = {}

    rows_list = []
    for _, row in inv.iterrows():
        med_id = row["medicine_id"]
        days     = int(row["days_until_expiry"])
        quantity = int(row["quantity"])

        if med_id not in forecast_cache:
            forecast_cache[med_id] = forecast_demand_30days(med_id, df, features, xgb, xgb_bias)
        forecasted_dem = forecast_cache[med_id]

        risk, reason = classify_risk_3factor(days, quantity, forecasted_dem)

        rows_list.append({
            "medicine_id":       med_id,
            "medicine_name":     row["medicine_name"],
            "batch_number":      row["batch_number"],
            "quantity":          quantity,
            "expiry_date":       row["expiry_date"],
            "days_until_expiry": days,
            "forecasted_demand": forecasted_dem,
            "risk_status":       risk,
            "risk_reason":       reason,
        })

    risk_df = pd.DataFrame(rows_list)

    high   = risk_df[risk_df["risk_status"] == "High"]
    medium = risk_df[risk_df["risk_status"] == "Medium"]
    low    = risk_df[risk_df["risk_status"] == "Low"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">🔴 High Risk Batches</div>
            <div class="kpi-value">{len(high)}</div>
            <div class="kpi-delta-bad">{"Immediate action required" if len(high) > 0 else ""}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">🟡 Medium Risk Batches</div>
            <div class="kpi-value">{len(medium)}</div>
            <div class="kpi-delta-bad">{"Monitor closely" if len(medium) > 0 else ""}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">🟢 Low Risk Batches</div>
            <div class="kpi-value">{len(low)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="chart-card"><div class="chart-title">Expiry Risk Overview</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        med_options = ["All"] + sorted(risk_df["medicine_name"].unique().tolist())
        med_filter  = st.selectbox("Filter by Medicine", med_options)
    with col2:
        risk_filter = st.selectbox("Filter by Risk", ["All", "High", "Medium", "Low"])

    filtered = risk_df.copy()
    if med_filter != "All":
        filtered = filtered[filtered["medicine_name"] == med_filter]
    if risk_filter != "All":
        filtered = filtered[filtered["risk_status"] == risk_filter]

    filtered = filtered.sort_values("days_until_expiry").reset_index(drop=True)

    headers     = ["Batch", "Medicine Name", "Quantity", "Expiry Date", "Days Until Expiry", "Forecasted Demand", "Risk", "Action"]
    header_cols = st.columns([1, 2, 0.8, 1.3, 1.2, 1.3, 0.8, 0.8])
    for col, h in zip(header_cols, headers):
        col.markdown(f"**{h}**")

    st.divider()

    for _, row in filtered.iterrows():
        days     = int(row["days_until_expiry"])
        risk     = row["risk_status"]
        forecast = int(row["forecasted_demand"])

        if risk == "High":
            risk_badge = "🔴 High"
            color      = "#ef4444"
        elif risk == "Medium":
            risk_badge = "🟡 Medium"
            color      = "#f59e0b"
        else:
            risk_badge = "🟢 Low"
            color      = "#22c55e"

        cols = st.columns([1, 2, 0.8, 1.3, 1.2, 1.3, 0.8, 0.8])
        cols[0].write(row["batch_number"])
        cols[1].write(row["medicine_name"])
        cols[2].markdown(f'<span style="color:{color}; font-weight:600;">{row["quantity"]}</span>', unsafe_allow_html=True)
        cols[3].markdown(f'<span style="color:{color}; font-weight:600;">{row["expiry_date"].strftime("%d %b %Y")}</span>', unsafe_allow_html=True)
        cols[4].markdown(f'<span style="color:{color}; font-weight:600;">{days}</span>', unsafe_allow_html=True)
        cols[5].markdown(f'<span style="color:#4f46e5; font-weight:600;">{forecast} units</span>', unsafe_allow_html=True)
        cols[6].write(risk_badge)

        with cols[7]:
            if st.button("Details", key=f"detail_{row['batch_number']}_{row['medicine_id']}"):
                st.session_state["detail_medicine"] = {
                    "name":           row["medicine_name"],
                    "batch":          row["batch_number"],
                    "expiry_date":    row["expiry_date"].strftime("%d %b %Y"),
                    "days":           days,
                    "risk":           risk,
                    "risk_reason":    row["risk_reason"],
                    "quantity":       row["quantity"],
                    "forecasted":     forecast,
                    "recommendation": get_recommendation(risk, days, row["quantity"], forecast)
                }

    st.markdown('</div>', unsafe_allow_html=True)

    if "detail_medicine" in st.session_state and st.session_state["detail_medicine"]:
        d = st.session_state["detail_medicine"]

        @st.dialog(f"📋 {d['name']} — Batch {d['batch']}")
        def show_detail():
            if d["risk"] == "High":
                st.error(f"Risk Level: 🔴 High — {d['risk_reason']}")
            elif d["risk"] == "Medium":
                st.warning(f"Risk Level: 🟡 Medium — {d['risk_reason']}")
            else:
                st.success(f"Risk Level: 🟢 Low — {d['risk_reason']}")

            col1, col2 = st.columns(2)
            col1.metric("Expiry Date",       d["expiry_date"])
            col2.metric("Days Until Expiry", d["days"])
            col1.metric("Current Stock",     f"{d['quantity']} units")
            col2.metric("Forecasted Demand", f"{d['forecasted']} units")

            st.markdown("---")
            st.markdown("**📌 Recommendation**")
            st.info(d["recommendation"])

            if st.button("Close", use_container_width=True):
                st.session_state["detail_medicine"] = None
                st.rerun()

        show_detail()