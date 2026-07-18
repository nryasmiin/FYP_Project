import streamlit as st
from streamlit_option_menu import option_menu
from utils.auth import require_login, sidebar_user_panel

st.set_page_config(
    page_title="Healthcare Inventory System",
    page_icon="💊",
    layout="wide"
)

# require_login()  # temporarily disabled
st.session_state.logged_in  = True
st.session_state.user_role  = "Admin"
st.session_state.user_name  = "Admin User"
st.session_state.user_email = "admin@pharma.com"

if "page" not in st.session_state:
    st.session_state.page = "Overview"

nav_items = [
    ("Overview",          "Overview"),
    ("Demand Forecast",   "Demand Forecast"),
    ("Expiry Monitoring", "Expiry Monitoring"),
    ("Inventory",         "Inventory"),
]
if st.session_state.user_role == "Admin":
    nav_items.append(("Data Management", "Data Management"))

st.sidebar.markdown("""
<style>
.sidebar-title {
    font-size: 20px;
    font-weight: 800;
    color: #000000;
    padding: 12px 4px;
    line-height: 1.2;
}
section[data-testid="stSidebar"] > div {
    display: flex;
    flex-direction: column;
    height: 100%;
}
.sidebar-spacer { flex-grow: 1; }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    '<div class="sidebar-title">Healthcare Inventory System</div>',
    unsafe_allow_html=True
)
st.sidebar.divider()

with st.sidebar:
    selected_page = option_menu(
        menu_title=None,
        options=[item[0] for item in nav_items],
        icons=["house", "graph-up-arrow", "exclamation-triangle", "box-seam", "database"],
        default_index=[item[0] for item in nav_items].index(st.session_state.page),
        styles={
            "container":        {"padding": "0px", "background-color": "transparent"},
            "icon":             {"color": "#2563eb", "font-size": "18px"},
            "nav-link":         {
                "font-size": "14px", "font-weight": "500",
                "text-align": "left", "margin": "4px 0px",
                "padding": "10px 14px", "border-radius": "10px",
                "color": "#1a56db",
            },
            "nav-link-selected": {
                "background-color": "#2563eb",
                "color": "#ffffff",
                "font-weight": "700",
            },
        }
    )

if selected_page != st.session_state.page:
    st.session_state.page = selected_page
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)
sidebar_user_panel()

page = st.session_state.page

if page == "Overview":
    from src_pages.dashboard import show
    show()
elif page == "Demand Forecast":
    from src_pages.demand_forecasting import show
    show()
elif page == "Expiry Monitoring":
    from src_pages.expiry_risk import show
    show()
elif page == "Inventory":
    from src_pages.inventory import show
    show()
elif page == "Data Management":
    if st.session_state.user_role == "Admin":
        from src_pages.admin import show
        show()
    else:
        st.error("Access denied. Admin only.")