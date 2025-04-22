import streamlit as st

from config import API_URL, USERNAME, PASSWORD
from tab_classification import render_classification_tab
from tab_similar_docs import render_similar_docs_tab
from tab_data_upload import render_data_upload_tab

# Настройка страницы
st.set_page_config(
    page_title="Классификатор запросов",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Глобальные переменные для настроек API
api_url = API_URL
username = USERNAME
password = PASSWORD

# Боковая панель для настроек API
with st.sidebar:
    st.title("Настройки API")
    api_url = st.text_input("URL API", value=API_URL)
    username = st.text_input("Имя пользователя", value=USERNAME)
    password = st.text_input("Пароль", value=PASSWORD, type="password")

    if st.button("Сохранить настройки"):
        st.success("Настройки сохранены")

# Создаем вкладки
tab1, tab2, tab3 = st.tabs(["Классификация", "Похожие документы", "Загрузка данных"])

# Вкладка 1: Классификация
with tab1:
    render_classification_tab(api_url, username, password)

# Вкладка 2: Похожие документы
with tab2:
    render_similar_docs_tab(api_url, username, password)

# Вкладка 3: Загрузка данных и оценка
with tab3:
    render_data_upload_tab(api_url, username, password)
