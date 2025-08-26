# main.py
import streamlit as st

from config import API_URL, USERNAME, PASSWORD
from tab_classification import render_classification_tab
from tab_similar_docs import render_similar_docs_tab
from tab_data_upload import render_data_upload_tab
from i18n import t, get_translations, get_lang_options, get_lang_display_name

# --- Инициализация языка в session_state ---
if "lang" not in st.session_state:
    st.session_state["lang"] = "ru"
lang = st.session_state["lang"]
TR = get_translations(lang)

# --- Конфиг страницы (title зависит от языка) ---
st.set_page_config(
    page_title=t("app.title", lang),
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Верхняя строка с селектором языка (рядом с верхней панелью) ---
top_left, _, _, top_right = st.columns([1, 1, 1, 1])
with top_right:
    st.selectbox(
        TR["common.lang_label"],
        options=get_lang_options(),
        index=get_lang_options().index(lang),
        key="lang",
        format_func=get_lang_display_name,
    )
# Обновляем локальные ссылки на язык/переводчик (на случай смены)
lang = st.session_state["lang"]
TR = get_translations(lang)

# --- Глобальные параметры API (из env / config) ---
api_url = API_URL
username = USERNAME
password = PASSWORD

# --- Sidebar: локализованные подписи ---
with st.sidebar:
    st.title(TR["sidebar.title"])
    username = st.text_input(TR["sidebar.username"])
    password = st.text_input(TR["sidebar.password"], type="password")

    if st.button(TR["sidebar.save"]):
        st.success(TR["sidebar.saved_success"])

# --- Tabs: локализованные названия ---
tab1, tab2, tab3 = st.tabs([
    TR["tabs.classification"],
    TR["tabs.similar_docs"],
    TR["tabs.data_upload"],
])

with tab1:
    render_classification_tab(api_url, username, password)

with tab2:
    render_similar_docs_tab(api_url, username, password)

with tab3:
    render_data_upload_tab(api_url, username, password)