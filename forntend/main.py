import streamlit as st

from config import API_URL
from auth_ui import ensure_auth, top_bar, current_lang, tr
from tab_classification import render_classification_tab
from tab_similar_docs import render_similar_docs_tab
from tab_data_upload import render_data_upload_tab
from i18n import t

st.markdown("<style>[data-testid='stSidebar']{display:none!important}</style>", unsafe_allow_html=True)
st.session_state.setdefault("auth", {"ok": False, "username": None, "password": None, "token": None})
st.session_state.setdefault("lang", "ru")
st.session_state.setdefault("lang_login", "ru")

# ===== page =====
st.set_page_config(
    page_title=t("app.title", current_lang()),
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ===== app =====
ensure_auth(API_URL)
_ = tr()
top_bar()
st.title(t("app.title", current_lang()))

username = st.session_state["auth"]["username"]
password = st.session_state["auth"]["password"]
token = st.session_state["auth"]["token"]  # если понадобится — есть в state

tab1, tab2, tab3 = st.tabs([_["tabs.classification"], _["tabs.similar_docs"], _["tabs.data_upload"]])

with tab1:
    render_classification_tab(API_URL, username, password)

with tab2:
    render_similar_docs_tab(API_URL, username, password)

with tab3:
    render_data_upload_tab(API_URL, username, password)
