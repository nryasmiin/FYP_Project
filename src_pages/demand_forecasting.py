import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
    .week-banner {
        background-color: rgba(99, 102, 241, 0.1);
        border-radius: 10px;
        padding: 10px 16px;
        border-left: 4px solid #6366f1;
        margin-bottom: 14px;
        font-size: 14px;
        color: #4f46e5;
        font-weight: 500;
    }
    .kpi-card {
        background-color: rgba(99, 102, 241, 0.08);
        border-radius: 14px;
        padding: 16px;
        border: 1px solid rgba(99, 102, 241, 0.15);
        text-align: center;
        margin-bottom: 12px;
    }
    .kpi-label {
        font-size: 11px;
        color: #4f46e5;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
        font-weight: 500;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: 700;
        color: #000000;
    }
    .kpi-sub {
        font-size: 11px;
        color: #666666;
        margin-top: 4px;
    }
    .med-card {
        background-color: rgba(99, 102, 241, 0.08);
        border-radius: 14px;
        padding: 16px 16px 12px 16px;
        border: 1px solid rgba(99, 102, 241, 0.15);
        margin-bottom: 4px;
        min-height: 110px;
    }
    .med-name {
        font-size: 13px;
        font-weight: 700;
        color: #000000;
        margin-bottom: 10px;
    }
    .med-stat-label {
        font-size: 10px;
        color: #4f46e5;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 1px;
    }
    .med-stat-value {
        font-size: 18px;
        font-weight: 700;
        color: #000000;
        margin-bottom: 8px;
    }
    .med-trend-up   { font-size: 11px; color: #22c55e; font-weight: 500; }
    .med-trend-down { font-size: 11px; color: #ef4444; font-weight: 500; }
    .med-trend-flat { font-size: 11px; color: #aaaaaa; font-weight: 500; }

    [data-testid="stTable"] th {
        color: #000000 !important;
        font-weight: 600 !important;
        background-color: rgba(99, 102, 241, 0.1) !important;
    }
    [data-testid="stTable"] td {
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)


COLORS = ["#6366f1", "#f59e0b", "#22c55e", "#ef4444", "#a855f7", "#06b6d4"]


def get_weekly_sales(sales, medicine):
    full = sales.copy()
    full["week"] = full["date"].dt.to_period("W").apply(lambda r: r.start_time)
    week_day_counts = full.groupby("week")["date"].nunique()
    complete_weeks  = week_day_counts[week_day_counts >= 7].index
    full            = full[full["week"].isin(complete_weeks)]
    result = {}
    for _, med_row in medicine.iterrows():
        med_id   = med_row["medicine_id"]
        med_name = med_row["medicine_name"]
        med_data = full[full["medicine_id"] == med_id]
        weekly   = med_data.groupby("week")["quantity_sold"].sum().reset_index()
        result[med_id] = {"name": med_name, "weekly": weekly}
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def get_forecast(med_df, features, _xgb_model, xgb_bias):
    med_df     = med_df.sort_values("date").copy()
    split_date = med_df["date"].iloc[int(len(med_df) * 0.8)]
    test_df    = med_df[med_df["date"] >= split_date]
    X_test     = test_df[features]
    y_pred_raw = np.clip(_xgb_model.predict(X_test) + xgb_bias, 0, None)

    pred_df         = test_df[["date"]].copy()
    pred_df["pred"] = y_pred_raw
    pred_df["week"] = pred_df["date"].dt.to_period("W").apply(lambda r: r.start_time)

    week_day_counts  = pred_df.groupby("week")["date"].nunique()
    complete_weeks   = week_day_counts[week_day_counts >= 7].index
    pred_df_complete = pred_df[pred_df["week"].isin(complete_weeks)]
    weekly_pred      = pred_df_complete.groupby("week")["pred"].sum().round(0).astype(int).reset_index()
    if len(weekly_pred) > 2:
        weekly_pred = weekly_pred.iloc[2:].reset_index(drop=True)

    last_row     = med_df.iloc[-1].copy()
    last_date    = last_row["date"]
    recent_sales = med_df["quantity_sold"].values.tolist()

    # Use recent 30-day average rather than the single last value, since
    # freezing the most recent promotion status (which may be a one-off
    # spike) would incorrectly assume it continues for the entire
    # 8-week forecast horizon.
    recent_30d = med_df.tail(30)
    last_selling_price = recent_30d["selling_price"].mean() if "selling_price" in med_df.columns else 0
    last_promotion      = recent_30d["promotion"].mean() if "promotion" in med_df.columns else 0

    same_period_last_year = med_df[
        (med_df["date"] >= last_date - pd.Timedelta(days=393)) &
        (med_df["date"] <= last_date - pd.Timedelta(days=337))
    ]["quantity_sold"].values.tolist()

    forecast_days = []
    for i in range(1, 64):
        future_date   = last_date + pd.Timedelta(days=i)
        lag_1         = recent_sales[-1]
        lag_7         = recent_sales[-7]  if len(recent_sales) >= 7  else np.mean(recent_sales)
        lag_14        = recent_sales[-14] if len(recent_sales) >= 14 else np.mean(recent_sales)
        if len(same_period_last_year) > 0 and i <= len(same_period_last_year):
            lag_28 = same_period_last_year[i - 1]
        elif len(recent_sales) >= 28:
            lag_28 = recent_sales[-28]
        else:
            lag_28 = np.mean(recent_sales)
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
            "lag_1": lag_1, "lag_7": lag_7, "lag_14": lag_14, "lag_28": lag_28,
            "rolling_7": rolling_7, "rolling_14": rolling_14, "rolling_28": rolling_28,
            "rolling_std_7": rolling_std_7,
            "selling_price": last_selling_price,
            "promotion": last_promotion,
            "is_public_holiday": is_public_holiday,
            "is_monsoon_season": is_monsoon_season,
        }

        pred_qty = int(max(0, round(float(_xgb_model.predict(pd.DataFrame([row])[features])[0]) + xgb_bias)))
        forecast_days.append({"date": future_date, "pred": pred_qty})
        recent_sales.append(pred_qty)

    forecast_df          = pd.DataFrame(forecast_days)
    forecast_df["week"]  = forecast_df["date"].dt.to_period("W").apply(lambda r: r.start_time)
    forecast_week_counts = forecast_df.groupby("week")["date"].nunique()
    complete_forecast    = forecast_week_counts[forecast_week_counts >= 7].index
    weekly_forecast      = (
        forecast_df[forecast_df["week"].isin(complete_forecast)]
        .groupby("week")["pred"].sum().round(0).astype(int).reset_index()
    )

    bridge_week = weekly_pred["week"].iloc[-1]
    bridge_val  = int(weekly_pred["pred"].iloc[-1])

    # Dampened seasonal adjustment: nudges the flat autoregressive
    # forecast using last year's relative week-to-week pattern, but
    # capped to +/-20% of the model's own baseline so it can never
    # flatten completely nor spike unrealistically.
    if len(same_period_last_year) >= 14:
        ly_weekly = [sum(same_period_last_year[i:i+7]) for i in range(0, len(same_period_last_year) - 6, 7)]
        if len(ly_weekly) >= 2:
            ly_mean = np.mean(ly_weekly)
            if ly_mean > 0:
                raw_idx    = [(w / ly_mean) - 1 for w in ly_weekly]        # e.g. +0.1, -0.05 ...
                capped_idx = [np.clip(r, -0.20, 0.20) for r in raw_idx]     # cap swing to +/-20%
                weekly_forecast["pred"] = [
                    int(round(weekly_forecast["pred"].iloc[i] * (1 + capped_idx[i % len(capped_idx)])))
                    for i in range(len(weekly_forecast))
                ]

    # Confidence band: uncertainty grows with each week further into
    # the future, based on the model's weekly prediction error (RMSE)
    # observed during the historical test period.
    if len(weekly_pred) >= 2 and "pred" in weekly_pred.columns:
        weekly_rmse = float(np.std(weekly_pred["pred"].diff().dropna())) if len(weekly_pred) > 2 else weekly_pred["pred"].std()
        if pd.isna(weekly_rmse) or weekly_rmse <= 0:
            weekly_rmse = weekly_forecast["pred"].mean() * 0.1
    else:
        weekly_rmse = weekly_forecast["pred"].mean() * 0.1

    weekly_forecast["lower"] = [
        max(0, int(round(weekly_forecast["pred"].iloc[i] - weekly_rmse * np.sqrt(i + 1))))
        for i in range(len(weekly_forecast))
    ]
    weekly_forecast["upper"] = [
        int(round(weekly_forecast["pred"].iloc[i] + weekly_rmse * np.sqrt(i + 1)))
        for i in range(len(weekly_forecast))
    ]

    return weekly_pred, weekly_forecast, bridge_week, bridge_val


def show():
    inject_styles()

    st.title("Demand Forecasting")
    st.markdown("Weekly medicine demand overview and forecast.")

    medicine, sales, _ = load_raw_data()
    df                  = load_merged_sales()
    xgb, rf, features, comparison, xgb_bias = load_models()

    med_list = medicine[["medicine_id", "medicine_name", "category"]].copy()

    # ==============================
    # CURRENT WEEK BANNER
    # ==============================

    today      = pd.Timestamp.today().normalize()
    week_start = today - pd.Timedelta(days=today.weekday())
    week_end   = week_start + pd.Timedelta(days=6)

    st.markdown(f"""
    <div class="week-banner">
        📅 Current Week: <b>{week_start.strftime('%d %b %Y')} – {week_end.strftime('%d %b %Y')}</b>
        &nbsp;&nbsp;|&nbsp;&nbsp; Today: <b>{today.strftime('%d %b %Y (%A)')}</b>
    </div>
    """, unsafe_allow_html=True)

    # ==============================
    # DEFINE SELECTIONS
    # ==============================

    col1, col2 = st.columns([1, 2])
    with col1:
        available_years  = ["All"] + sorted(sales["date"].dt.year.unique().tolist(), reverse=True)
        selected_year    = st.selectbox("Year", available_years, index=0)
    with col2:
        med_options      = ["All"] + med_list["medicine_name"].tolist()
        selected_display = st.selectbox("Medicine", med_options, index=0)

    selected_med_id = None
    if selected_display != "All":
        selected_med_id = med_list.loc[med_list["medicine_name"] == selected_display, "medicine_id"].values[0]

    # ==============================
    # NEXT WEEK KPI (only when medicine selected)
    # ==============================

    if selected_med_id is not None:
        med_df_kpi = df[df["medicine_id"] == selected_med_id].sort_values("date")
        if not med_df_kpi.empty:
            try:
                _, wf_kpi, _, _ = get_forecast(med_df_kpi, features, xgb, xgb_bias)
                if not wf_kpi.empty:
                    nw       = wf_kpi.iloc[0]
                    nw_start = pd.Timestamp(nw["week"])
                    nw_end   = nw_start + pd.Timedelta(days=6)
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">📦 Next Week Predicted Demand — {selected_display}</div>
                        <div class="kpi-value">{int(nw['pred']):,} units</div>
                        <div class="kpi-sub">{nw_start.strftime('%d %b')} – {nw_end.strftime('%d %b %Y')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"KPI forecast error: {e}")

    # ==============================
    # MAIN CHART
    # ==============================

    weekly_data = get_weekly_sales(sales, med_list)

    all_vals = []
    for med_id, data in weekly_data.items():
        all_vals.extend(data["weekly"]["quantity_sold"].tolist())
    y_max = max(all_vals) * 1.15 if all_vals else 100

    fig = go.Figure()

    if selected_med_id is None:
        for idx, (med_id, data) in enumerate(weekly_data.items()):
            weekly = data["weekly"].copy()
            if selected_year != "All":
                weekly = weekly[weekly["week"].dt.year == int(selected_year)]
            weekly["week_end"]   = weekly["week"] + pd.Timedelta(days=6)
            weekly["week_label"] = weekly.apply(
                lambda r: f"{r['week'].strftime('%d %b')} – {r['week_end'].strftime('%d %b %Y')}", axis=1
            )
            color = COLORS[idx % len(COLORS)]

            fig.add_trace(go.Scatter(
                x=weekly["week"],
                y=weekly["quantity_sold"],
                name=data["name"],
                mode="lines",
                line=dict(color=color, width=2),
                customdata=weekly["week_label"],
                hovertemplate="<b>%{customdata}</b><br>Units Sold: %{y:,}<extra></extra>",
                legendgroup=med_id,
            ))

            med_df_all = df[df["medicine_id"] == med_id].sort_values("date")
            if not med_df_all.empty:
                try:
                    _, wf, bw, bv = get_forecast(med_df_all, features, xgb, xgb_bias)
                    if not wf.empty:
                        wf["week_end"]   = wf["week"] + pd.Timedelta(days=6)
                        wf["week_label"] = wf.apply(
                            lambda r: f"{r['week'].strftime('%d %b')} – {r['week_end'].strftime('%d %b %Y')}", axis=1
                        )
                        bridge_lbl = f"{pd.Timestamp(bw).strftime('%d %b')} – {(pd.Timestamp(bw) + pd.Timedelta(days=6)).strftime('%d %b %Y')}"
                        fig.add_trace(go.Scatter(
                            x=[bw] + list(wf["week"]),
                            y=[bv] + list(wf["pred"]),
                            name=f"{data['name']} (Forecast)",
                            mode="lines",
                            line=dict(color=color, width=1.5, dash="dot"),
                            customdata=[bridge_lbl] + list(wf["week_label"]),
                            hovertemplate="<b>%{customdata}</b><br>Forecast: %{y:,}<extra></extra>",
                            legendgroup=med_id,
                            showlegend=False,
                        ))
                except Exception as e:
                    st.warning(f"{data['name']} forecast error: {e}")

    else:
        data   = weekly_data[selected_med_id]
        weekly = data["weekly"].copy()
        if selected_year != "All":
            weekly = weekly[weekly["week"].dt.year == int(selected_year)]
        weekly["week_end"]   = weekly["week"] + pd.Timedelta(days=6)
        weekly["week_label"] = weekly.apply(
            lambda r: f"{r['week'].strftime('%d %b')} – {r['week_end'].strftime('%d %b %Y')}", axis=1
        )

        med_df = df[df["medicine_id"] == selected_med_id].sort_values("date")
        weekly_pred, weekly_forecast, bridge_week, bridge_val = get_forecast(med_df, features, xgb, xgb_bias)

        weekly_pred["week_end"]   = weekly_pred["week"] + pd.Timedelta(days=6)
        weekly_pred["week_label"] = weekly_pred.apply(
            lambda r: f"{r['week'].strftime('%d %b')} – {r['week_end'].strftime('%d %b %Y')}", axis=1
        )
        weekly_forecast["week_end"]   = weekly_forecast["week"] + pd.Timedelta(days=6)
        weekly_forecast["week_label"] = weekly_forecast.apply(
            lambda r: f"{r['week'].strftime('%d %b')} – {r['week_end'].strftime('%d %b %Y')}", axis=1
        )
        bridge_label  = f"{pd.Timestamp(bridge_week).strftime('%d %b')} – {(pd.Timestamp(bridge_week) + pd.Timedelta(days=6)).strftime('%d %b %Y')}"
        forecast_labels = [bridge_label] + list(weekly_forecast["week_label"])

        # Confidence band (upper/lower bound) — drawn first so the
        # forecast/actual/predicted lines render on top of it
        band_x = [bridge_week] + list(weekly_forecast["week"]) + list(weekly_forecast["week"])[::-1] + [bridge_week]
        band_y = (
            [bridge_val] + list(weekly_forecast["upper"]) +
            list(weekly_forecast["lower"])[::-1] + [bridge_val]
        )
        fig.add_trace(go.Scatter(
            x=band_x,
            y=band_y,
            fill="toself",
            fillcolor="rgba(245, 158, 11, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Forecast Range",
            showlegend=True,
        ))

        fig.add_trace(go.Bar(
            x=weekly["week"],
            y=weekly["quantity_sold"],
            name="Actual",
            marker_color="rgba(99, 102, 241, 0.4)",
            customdata=weekly["week_label"],
            hovertemplate="<b>%{customdata}</b><br>Actual: %{y:,}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=weekly_pred["week"],
            y=weekly_pred["pred"],
            name="Predicted",
            mode="lines+markers",
            line=dict(color="#ef4444", width=2),
            marker=dict(size=5),
            customdata=weekly_pred["week_label"],
            hovertemplate="<b>%{customdata}</b><br>Predicted: %{y:,}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[bridge_week] + list(weekly_forecast["week"]),
            y=[bridge_val]  + list(weekly_forecast["pred"]),
            name="Forecast",
            mode="lines+markers",
            line=dict(color="#f59e0b", width=2, dash="dash"),
            marker=dict(size=5),
            customdata=forecast_labels,
            hovertemplate="<b>%{customdata}</b><br>Forecast: %{y:,}<extra></extra>",
        ))


    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(showgrid=True, gridcolor="#e0e0e0", gridwidth=0.5, title="Units Sold (Weekly)"),
        xaxis=dict(showgrid=False, title="Week"),
        margin=dict(t=30, b=10, l=10, r=10),
        height=400,
        barmode="overlay",
    )

    st.plotly_chart(fig, use_container_width=True, key="main_chart")
    st.markdown("<br>", unsafe_allow_html=True)

    # ==============================
    # 8-WEEK FORECAST TABLE (single medicine view)
    # ==============================

    if selected_med_id is not None:
        st.markdown('<div class="chart-card"><div class="chart-title">8-Week Forecast</div>', unsafe_allow_html=True)

        med_df_single = df[df["medicine_id"] == selected_med_id].sort_values("date")
        if not med_df_single.empty:
            try:
                _, wf_single, _, _ = get_forecast(med_df_single, features, xgb, xgb_bias)
                single_rows = []
                for i in range(min(8, len(wf_single))):
                    wk_start = pd.Timestamp(wf_single.iloc[i]["week"])
                    wk_end   = wk_start + pd.Timedelta(days=6)
                    single_rows.append({
                        "Week": f"{wk_start.strftime('%d %b')} – {wk_end.strftime('%d %b %Y')}",
                        "Lower Bound": int(wf_single.iloc[i]["lower"]),
                        "Predicted Units": int(wf_single.iloc[i]["pred"]),
                        "Upper Bound": int(wf_single.iloc[i]["upper"]),
                    })
                single_table_df = pd.DataFrame(single_rows)
                st.table(single_table_df.set_index("Week"))
            except Exception as e:
                st.error(f"Forecast table error: {e}")

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ==============================
    # KPI CARDS + 8-WEEK TABLE (only "All" view)
    # ==============================

    if selected_med_id is None:
        st.markdown('<div class="chart-card"><div class="chart-title">Medicine Overview</div>', unsafe_allow_html=True)

        rows = [med_list.iloc[0:3], med_list.iloc[3:6]]
        for row in rows:
            cols = st.columns(3)
            for i, (_, med_row) in enumerate(row.iterrows()):
                med_id      = med_row["medicine_id"]
                med_name    = med_row["medicine_name"]
                med_sales   = sales[sales["medicine_id"] == med_id]["quantity_sold"]
                total_sales = int(med_sales.sum())
                avg_daily   = int(round(med_sales.mean()))
                last_7      = med_sales.tail(7).mean()
                prev_7      = med_sales.tail(14).head(7).mean()
                if last_7 > prev_7 * 1.05:
                    trend_html = '<div class="med-trend-up">↑ Trending up</div>'
                elif last_7 < prev_7 * 0.95:
                    trend_html = '<div class="med-trend-down">↓ Trending down</div>'
                else:
                    trend_html = '<div class="med-trend-flat">→ Stable</div>'

                next_week_val = "—"
                med_df_card = df[df["medicine_id"] == med_id].sort_values("date")
                if not med_df_card.empty:
                    try:
                        _, wf_card, _, _ = get_forecast(med_df_card, features, xgb, xgb_bias)
                        if not wf_card.empty:
                            next_week_val = f"{int(wf_card.iloc[0]['pred']):,} units"
                    except Exception:
                        pass

                with cols[i]:
                    st.markdown(f"""
                    <div class="med-card">
                        <div class="med-name">{med_name}</div>
                        <div class="med-stat-label">Total Sales</div>
                        <div class="med-stat-value">{total_sales:,}</div>
                        <div class="med-stat-label">Avg Daily Demand</div>
                        <div class="med-stat-value">{avg_daily} units</div>
                        <div class="med-stat-label">Next Week Forecast</div>
                        <div class="med-stat-value">{next_week_val}</div>
                        {trend_html}
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # 8-week forecast table (all medicines)
        st.markdown('<div class="chart-card"><div class="chart-title">8-Week Forecast by Medicine</div>', unsafe_allow_html=True)

        forecast_table_rows = []
        for _, med_row in med_list.iterrows():
            med_id_f   = med_row["medicine_id"]
            med_name_f = med_row["medicine_name"]
            med_df_f   = df[df["medicine_id"] == med_id_f].sort_values("date")
            row_data   = {"Medicine": med_name_f}
            if not med_df_f.empty:
                try:
                    _, wf_table, _, _ = get_forecast(med_df_f, features, xgb, xgb_bias)
                    for i in range(min(8, len(wf_table))):
                        wk_start  = pd.Timestamp(wf_table.iloc[i]["week"])
                        wk_end    = wk_start + pd.Timedelta(days=6)
                        col_label = f"{wk_start.strftime('%d %b')}–{wk_end.strftime('%d %b')}"
                        row_data[col_label] = int(wf_table.iloc[i]["pred"])
                except Exception as e:
                    st.warning(f"{med_name_f}: {e}")
            forecast_table_rows.append(row_data)

        forecast_table_df = pd.DataFrame(forecast_table_rows)
        st.table(forecast_table_df.set_index("Medicine"))

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Model comparison
        # st.markdown('<div class="chart-card"><div class="chart-title">Model Comparison</div>', unsafe_allow_html=True)
        # st.table(comparison.set_index("Model"))
        # st.caption("✅ XGBoost selected as the final model based on combined ranking across MAE, RMSE and R².")
        # st.markdown('</div>', unsafe_allow_html=True)