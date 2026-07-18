import streamlit as st
import pandas as pd
from utils.data_loader import load_raw_data, load_merged_sales, load_models


def inject_styles():
    st.markdown("""
    <style>

    /* =========================
       Cards
    ========================== */
    .admin-card {
        background-color: rgba(99,102,241,0.08);
        border-radius: 12px;
        padding: 10px 12px;
        border: 1px solid rgba(99,102,241,0.15);
        margin-bottom: 8px;
    }

    .info-box {
        background-color: rgba(99,102,241,0.05);
        border-radius: 8px;
        padding: 6px 10px;
        border-left: 3px solid #6366f1;
        margin-bottom: 4px;
        font-size: 13px;
        color: #4f46e5;
        line-height: 1.4;
    }

    .stat-row {
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:8px 0;
        border-bottom:1px solid rgba(99,102,241,0.1);
    }

    .stat-label{
        font-size:14px;
        color:#4f46e5;
    }

    .stat-value{
        font-size:14px;
        font-weight:700;
        color:#000000;
    }

    /* =========================
       File uploader
    ========================== */

    [data-testid="stFileUploader"]{
        padding-top:0 !important;
        padding-bottom:0 !important;
    }

    [data-testid="stFileUploader"] label p{
        font-size:12px !important;
    }

    [data-testid="stFileUploader"] button{
        font-size:11px !important;
        height:30px !important;
        padding:3px 12px !important;
    }

    [data-testid="stFileUploader"] small{
        font-size:10px !important;
    }

    /* =========================
       Save button (green)
    ========================== */

    div[data-testid="stButton"] button[kind="primary"]{
        background:#22c55e !important;
        border:1px solid #22c55e !important;
        color:white !important;
    }

    div[data-testid="stButton"] button[kind="primary"]:hover{
        background:#16a34a !important;
        border:1px solid #16a34a !important;
    }

    </style>
    """, unsafe_allow_html=True)


def validate_sales_csv(df):
    # Updated: now requires selling_price and promotion for the 21-feature model
    required = {"date", "medicine_id", "quantity_sold", "selling_price", "promotion"}
    missing  = required - set(df.columns)
    if missing:
        return False, f"Missing columns: {', '.join(missing)}"
    try:
        pd.to_datetime(df["date"])
    except Exception:
        return False, "Column 'date' has invalid date format."
    if df["quantity_sold"].isnull().any():
        return False, "Column 'quantity_sold' has missing values."
    if df["selling_price"].isnull().any():
        return False, "Column 'selling_price' has missing values."
    if df["promotion"].isnull().any():
        return False, "Column 'promotion' has missing values."
    return True, "Valid"


def validate_inventory_csv(df):
    required = {"medicine_id", "batch_number", "quantity", "purchase_date", "expiry_date", "unit_cost"}
    missing  = required - set(df.columns)
    if missing:
        return False, f"Missing columns: {', '.join(missing)}"
    try:
        pd.to_datetime(df["expiry_date"])
        pd.to_datetime(df["purchase_date"])
    except Exception:
        return False, "Date columns have invalid format."
    return True, "Valid"


def show():
    inject_styles()

    st.title("Data Management")

    col_left, col_right = st.columns([1, 1.6], gap="large")

    # ==============================
    # LEFT — DATASET STATUS
    # ==============================

    with col_left:
        st.markdown("""
        <div class="admin-card">
            <div style="font-size:17px; font-weight:600; color:#000000; margin-bottom:6px;">Dataset Status</div>
        """, unsafe_allow_html=True)

        try:
            sales_df     = pd.read_csv("data/raw/medicine_sales.csv", encoding="latin-1")
            inventory_df = pd.read_csv("data/raw/pharmacy_inventory.csv")
            sales_df["date"] = pd.to_datetime(sales_df["date"])

            stats = [
                ("Total Sales Records", f"{len(sales_df):,}"),
                ("Unique Medicines",    str(sales_df["medicine_id"].nunique())),
                ("Last Sales Entry",    sales_df["date"].max().strftime("%d %b %Y")),
                ("Inventory Batches",   str(len(inventory_df))),
                ("Total Stock Units",   f"{int(inventory_df['quantity'].sum()):,}"),
            ]

            for label, value in stats:
                st.markdown(f"""
                <div class="stat-row">
                    <span class="stat-label">{label}</span>
                    <span class="stat-value">{value}</span>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Could not load datasets: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

    # ==============================
    # RIGHT — UPLOADS + SAVE
    # ==============================

    with col_right:

        # Weekly Sales
        st.markdown("""
        <div class="admin-card">
            <div style="font-size:15px; font-weight:600; color:#000000; margin-bottom:6px;">Upload Weekly Sales Data</div>
            <div class="info-box">Required: <code>date</code>, <code>medicine_id</code>, <code>quantity_sold</code>, <code>selling_price</code>, <code>promotion</code></div>
        </div>
        """, unsafe_allow_html=True)

        sales_file = st.file_uploader("Sales CSV", type="csv", key="sales_upload", label_visibility="collapsed")

        if sales_file:
            new_sales = pd.read_csv(sales_file, encoding="latin-1")
            valid, msg = validate_sales_csv(new_sales)
            if not valid:
                st.error(f"❌ {msg}")
            else:
                st.success(f"✅ {len(new_sales)} records ready.")
                if st.button("💾 Save", use_container_width=True, type="primary", key="append_sales"):
                    try:
                        existing = pd.read_csv("data/raw/medicine_sales.csv", encoding="latin-1")
                        existing["date"] = pd.to_datetime(existing["date"])
                        new_sales["date"] = pd.to_datetime(new_sales["date"])
                        combined = pd.concat([existing, new_sales], ignore_index=True)
                        combined = combined.drop_duplicates(subset=["date", "medicine_id"], keep="last")
                        combined = combined.sort_values(["medicine_id", "date"])
                        combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
                        combined.to_csv("data/raw/medicine_sales.csv", index=False)
                        load_raw_data.clear()
                        load_merged_sales.clear()
                        st.success(f"✅ Sales updated. Total: {len(combined):,}")
                    except Exception as e:
                        st.error(f"❌ {e}")

        # Weekly Inventory
        st.markdown("""
        <div class="admin-card">
            <div style="font-size:15px; font-weight:600; color:#000000; margin-bottom:6px;">Upload Weekly Inventory Data</div>
            <div class="info-box">Required: <code>medicine_id</code>, <code>batch_number</code>, <code>quantity</code>, <code>purchase_date</code>, <code>expiry_date</code>, <code>unit_cost</code></div>
        </div>
        """, unsafe_allow_html=True)

        inv_file = st.file_uploader("Inventory CSV", type="csv", key="inv_upload", label_visibility="collapsed")

        if inv_file:
            new_inv = pd.read_csv(inv_file)
            valid, msg = validate_inventory_csv(new_inv)
            if not valid:
                st.error(f"❌ {msg}")
            else:
                st.success(f"✅ {len(new_inv)} batches ready.")
                if st.button("💾 Save", use_container_width=True, type="primary", key="replace_inv"):
                    try:
                        new_inv.to_csv("data/raw/pharmacy_inventory.csv", index=False)
                        load_raw_data.clear()
                        st.success(f"✅ Inventory updated.")
                    except Exception as e:
                        st.error(f"❌ {e}")