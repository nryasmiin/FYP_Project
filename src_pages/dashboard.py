import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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
        font-size: 26px;
        font-weight: 700;
        color: #000000;
    }
    .kpi-delta-bad {
        font-size: 11px;
        color: #ef4444;
        margin-top: 4px;
    }
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
    </style>
    """, unsafe_allow_html=True)


def classify_stock(qty):
    if qty < 50:
        return "Low"
    elif qty <= 200:
        return "Medium"
    else:
        return "High"


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
        return "High"

    if days_until_expiry <= 30:
        return "High"

    forecast_covers = forecasted_demand_30days >= quantity

    if days_until_expiry <= 60:
        if not forecast_covers:
            return "High"
        else:
            return "Medium"
    elif days_until_expiry <= 100:
        if not forecast_covers:
            return "Medium"
        else:
            return "Low"
    else:
        return "Low"


def show():
    inject_styles()

    st.title("Dashboard")
    st.markdown("Healthcare Inventory Overview")

    medicine, sales, inventory = load_raw_data()
    df                          = load_merged_sales()
    xgb, rf, features, comparison, xgb_bias = load_models()

    inv = inventory.merge(medicine, on="medicine_id", how="left")

    stock_by_med = (
        inv.groupby("medicine_id")["quantity"]
        .sum()
        .reset_index()
    )
    stock_by_med["stock_status"] = stock_by_med["quantity"].apply(classify_stock)

    # Cache forecasts per medicine (not per batch) to avoid recalculating
    # the same medicine's demand forecast multiple times when it has
    # several inventory batches.
    forecast_cache = {}

    risk_rows = []
    for _, row in inv.iterrows():
        med_id   = row["medicine_id"]
        days     = int(row["days_until_expiry"])
        quantity = int(row["quantity"])

        if med_id not in forecast_cache:
            forecast_cache[med_id] = forecast_demand_30days(med_id, df, features, xgb, xgb_bias)
        forecast = forecast_cache[med_id]

        risk = classify_risk_3factor(days, quantity, forecast)
        risk_rows.append({
            "medicine_id":   med_id,
            "medicine_name": row["medicine_name"],
            "batch_number":  row["batch_number"],
            "risk_status":   risk
        })

    risk_df         = pd.DataFrame(risk_rows)
    high_risk_count = int((risk_df["risk_status"] == "High").sum())

    best_row        = comparison.loc[comparison["MAE"].idxmin()]
    total_medicines = medicine["medicine_id"].nunique()
    total_stock     = inv["quantity"].sum()
    low_stock_count = int((stock_by_med["stock_status"] == "Low").sum())

    # ==============================
    # KPI CARDS
    # ==============================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Medicines</div>
            <div class="kpi-value">{total_medicines}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Stock Units</div>
            <div class="kpi-value">{total_stock:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        delta = f'<div class="kpi-delta-bad">⚠ {low_stock_count} need reorder</div>' if low_stock_count > 0 else ""
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Low Stock Medicines</div>
            <div class="kpi-value">{low_stock_count}</div>
            {delta}
        </div>
        """, unsafe_allow_html=True)

    with col4:
        delta = f'<div class="kpi-delta-bad">⚠ {high_risk_count} need attention</div>' if high_risk_count > 0 else ""
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">High Expiry Risk</div>
            <div class="kpi-value">{high_risk_count} batches</div>
            {delta}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==============================
    # BAR CHART + EXPIRY STATUS BAR CHART
    # ==============================

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="chart-card"><div class="chart-title">Stock Status</div>', unsafe_allow_html=True)

        stock_counts = (
            stock_by_med["stock_status"]
            .value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
            .reset_index()
        )
        stock_counts.columns = ["Status", "Count"]

        fig_bar = px.bar(
            stock_counts,
            x="Status",
            y="Count",
            color="Status",
            color_discrete_map={"Low": "#ef4444", "Medium": "#f59e0b", "High": "#22c55e"},
            text="Count"
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(showgrid=True, gridcolor="#d0d0d0", gridwidth=0.5),
            xaxis_title="",
            yaxis_title="Number of Medicines",
            margin=dict(t=10, b=10, l=10, r=10),
            height=280
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="chart-card"><div class="chart-title">Expiry Status</div>', unsafe_allow_html=True)

        risk_counts = (
            risk_df["risk_status"]
            .value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
            .reset_index()
        )
        risk_counts.columns = ["Status", "Count"]

        fig_risk_bar = px.bar(
            risk_counts,
            x="Status",
            y="Count",
            color="Status",
            color_discrete_map={"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"},
            text="Count"
        )
        fig_risk_bar.update_traces(textposition="outside")
        fig_risk_bar.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(showgrid=True, gridcolor="#d0d0d0", gridwidth=0.5),
            xaxis_title="",
            yaxis_title="Number of Batches",
            margin=dict(t=10, b=10, l=10, r=10),
            height=280
        )
        st.plotly_chart(fig_risk_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ==============================
    # DAILY SALES TREND
    # ==============================

    st.markdown('<div class="chart-card"><div class="chart-title">Total Daily Sales Trend</div>', unsafe_allow_html=True)

    daily_sales = (
        sales.groupby("date")["quantity_sold"]
        .sum()
        .reset_index()
        .rename(columns={"quantity_sold": "Total Sold"})
    )

    fig_line = px.line(
        daily_sales,
        x="date",
        y="Total Sold",
        labels={"date": "Date", "Total Sold": "Units Sold"},
        color_discrete_sequence=["#6366f1"]
    )
    fig_line.update_traces(line=dict(width=2))
    fig_line.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=True, gridcolor="#d0d0d0", gridwidth=0.5),
        xaxis=dict(showgrid=False),
        margin=dict(t=10, b=10, l=10, r=10),
        height=300
    )
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ==============================
    # BEST MODEL SUMMARY
    # ==============================

    # st.markdown('<div class="chart-card"><div class="chart-title">🏆 Best Forecasting Model</div>', unsafe_allow_html=True)
    #
    # col1, col2 = st.columns([1, 2])
    # with col1:
    #     st.success(f"**{best_row['Model']}**")
    #     st.caption(f"MAE: {best_row['MAE']} — lowest error among all models")
    # with col2:
    #     st.dataframe(
    #         comparison.style.highlight_min(subset=["MAE", "RMSE"], color="#d4edda")
    #                         .highlight_max(subset=["R2"],           color="#d4edda"),
    #         use_container_width=True
    #     )
    # st.markdown('</div>', unsafe_allow_html=True)