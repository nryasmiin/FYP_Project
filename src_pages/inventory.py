import streamlit as st
import pandas as pd
from utils.data_loader import load_raw_data


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
    </style>
    """, unsafe_allow_html=True)


def show():
    inject_styles()

    st.title("Inventory Monitoring")
    st.markdown("Monitor current stock levels, batch details, and inventory value.")

    medicine, _, inventory = load_raw_data()
    inv = inventory.merge(medicine, on="medicine_id", how="left")

    if "total_value" not in inv.columns:
        if "unit_cost" in inv.columns:
            inv["total_value"] = inv["quantity"] * inv["unit_cost"]
        else:
            inv["total_value"] = 0

    # ==============================
    # KPI CARDS
    # ==============================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Medicines</div>
            <div class="kpi-value">{inv['medicine_id'].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Batches</div>
            <div class="kpi-value">{len(inv)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Stock Units</div>
            <div class="kpi-value">{inv['quantity'].sum():,}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    """
    # ==============================
    # STOCK LEVEL PER MEDICINE
    # ==============================
    st.subheader("Stock Level by Medicine")

    stock_summary = (
        inv.groupby(["medicine_id", "medicine_name", "category"])
        .agg(
            Total_Quantity=("quantity", "sum"),
            Total_Value=("total_value", "sum"),
            Batches=("batch_number", "count")
        )
        .reset_index()
        .sort_values("Total_Quantity", ascending=False)
    )
    st.dataframe(stock_summary, use_container_width=True)
    st.bar_chart(stock_summary.set_index("medicine_name")["Total_Quantity"])

    st.divider()
    """

    # ==============================
    # BATCH DETAILS
    # ==============================
    st.subheader("Batch Details")

    col1, col2 = st.columns(2)
    with col1:
        selected_med = st.selectbox(
            "Filter by Medicine",
            ["All"] + sorted(inv["medicine_name"].unique().tolist())
        )
    with col2:
        selected_cat = st.selectbox(
            "Filter by Category",
            ["All"] + sorted(inv["category"].unique().tolist())
        )

    filtered = inv.copy()
    if selected_med != "All":
        filtered = filtered[filtered["medicine_name"] == selected_med]
    if selected_cat != "All":
        filtered = filtered[filtered["category"] == selected_cat]

    display_cols = [
        "medicine_id", "medicine_name", "category",
        "batch_number", "quantity", "purchase_date", "expiry_date", "days_until_expiry"
    ]
    if "unit_cost" in filtered.columns:
        display_cols.append("unit_cost")
    if "total_value" in filtered.columns:
        display_cols.append("total_value")

    display_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[display_cols].sort_values("days_until_expiry"),
        use_container_width=True
    )

    st.divider()

    """
    # ==============================
    # VALUE BY CATEGORY
    # ==============================
    st.subheader("Inventory Value by Category")

    cat_value = (
        inv.groupby("category")["total_value"]
        .sum()
        .reset_index()
        .sort_values("total_value", ascending=False)
    )
    st.bar_chart(cat_value.set_index("category")["total_value"])
    """