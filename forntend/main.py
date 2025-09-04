# main.py

import streamlit as st

import os
from auth_ui import ensure_auth, top_bar, current_lang, tr
from tabs.tab_classification import render_classification_tab
from tabs.tab_similar_docs import render_similar_docs_tab
from tabs.tab_data_upload import render_data_upload_tab
from tabs.tab_retrain import render_retrain_tab
from i18n import t

API_URL = os.getenv("API_URL")

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

tab1, tab2, tab3, tab4 = st.tabs([
    _["tabs.classification"],
    _["tabs.similar_docs"],
    _["tabs.data_upload"],
    _["tabs.retrain"],
])

with tab1:
    render_classification_tab(API_URL, username, password)

with tab2:
    render_similar_docs_tab(API_URL, username, password)

with tab3:
    render_data_upload_tab(API_URL, username, password)

with tab4:
    render_retrain_tab(API_URL, username, password)
