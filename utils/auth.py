import streamlit as st

# ==============================
# USER CREDENTIALS
# ==============================

USERS = {
    "staff@pharma.com": {
        "password": "staff123",
        "role": "Staff",
        "name": "Staff User"
    },
    "admin@pharma.com": {
        "password": "admin123",
        "role": "Admin",
        "name": "Admin User"
    }
}


def init_session():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_email" not in st.session_state:
        st.session_state.user_email = None
    if "user_role" not in st.session_state:
        st.session_state.user_role = None
    if "user_name" not in st.session_state:
        st.session_state.user_name = None


def login_screen():
    st.markdown("""
    <style>
    .login-card {
        background-color: rgba(99, 102, 241, 0.08);
        border-radius: 16px;
        padding: 40px;
        border: 1px solid rgba(99, 102, 241, 0.15);
        max-width: 420px;
        margin: 60px auto 0 auto;
    }
    .login-title {
        font-size: 22px;
        font-weight: 700;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 4px;
    }
    .login-subtitle {
        font-size: 13px;
        color: #4f46e5;
        text-align: center;
        margin-bottom: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="login-card">
            <div class="login-title">💊 MedTrack</div>
            <div class="login-subtitle">Healthcare Inventory & Demand Prediction System</div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            email    = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit   = st.form_submit_button("Login", use_container_width=True)

            if submit:
                user = USERS.get(email)
                if user and user["password"] == password:
                    st.session_state.logged_in  = True
                    st.session_state.user_email = email
                    st.session_state.user_role  = user["role"]
                    st.session_state.user_name  = user["name"]
                    st.rerun()
                else:
                    st.error("Invalid email or password.")

        with st.expander("Demo credentials"):
            st.caption("**Staff:** staff@pharma.com / staff123")
            st.caption("**Admin:** admin@pharma.com / admin123")


def sidebar_user_panel():
    st.sidebar.markdown("""
    <style>
    .user-panel {
        background-color: rgba(99, 102, 241, 0.08);
        border-radius: 12px;
        padding: 12px;
        border: 1px solid rgba(99, 102, 241, 0.15);
        margin-top: 10px;
    }
    .user-name {
        font-size: 13px;
        font-weight: 700;
        color: #1a1a1a;
    }
    .user-role {
        font-size: 11px;
        color: #4f46e5;
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(f"""
    <div class="user-panel">
        <div class="user-name">👤 {st.session_state.user_name}</div>
        <div class="user-role">{st.session_state.user_role}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.logged_in  = False
        st.session_state.user_email = None
        st.session_state.user_role  = None
        st.session_state.user_name  = None
        st.rerun()


def require_login():
    init_session()
    if not st.session_state.logged_in:
        login_screen()
        st.stop()