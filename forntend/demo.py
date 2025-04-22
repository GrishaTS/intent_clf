import time
import uuid

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
import os 
from evaluate import upload_data, predict, accuracy_score, train_test_split, classification_report, confusion_matrix


API_URL = os.getenv("API_URL", "http://localhost:8190") 
USERNAME = os.getenv("USERNAME", "admin") 
PASSWORD = os.getenv("PASSWORD", "secret")

# Примеры запросов для выбора
DEFAULT_EXAMPLES = [
    {
        "id": "INC0027099",
        "description": "Удалить старые версии 1С клиента на пк коменданта.",
        "subject": "Старая версия 1С клиента. Садовники д.4 к.2",
        "class": "Сопровождение сервисов сотрудника",
        "task": "1С клиент",
    }
    # Можно добавить больше примеров в будущем
]


# Функция для получения токена
def get_token():
    try:
        response = requests.post(
            f"{API_URL}/token",
            data={
                "username": USERNAME,
                "password": PASSWORD,
                "scope": "predict upload search",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if response.status_code != 200:
            st.error(f"Ошибка аутентификации: {response.text}")
            return None

        return response.json()["access_token"]
    except requests.exceptions.ConnectionError:
        st.error(f"Не удалось подключиться к API по адресу {API_URL}")
        return None


# Функция для классификации
def classify_request(subject, description, token):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Генерируем ID для запроса
    item_id = str(uuid.uuid4())

    payload = {
        "id": item_id,  # Обязательное поле!
        "subject": subject if subject else "no_subject",
        "description": description if description else "no_description",
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, headers=headers)
        response.raise_for_status()

        result = response.json()
        return result
    except Exception as e:
        st.error(f"Ошибка при классификации: {str(e)}")
        return None


# Функция для поиска похожих документов - ИСПРАВЛЕНО
def search_similar(subject, description, token, limit=10):
    """
    Поиск похожих документов на основе темы и описания
    """
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Создаем объект запроса с двумя полями - subject и description
    # Также добавляем id, так как это может требоваться API
    payload = {
        "id": str(uuid.uuid4()),
        "subject": subject if subject else "no_subject",
        "description": description if description else "no_description",
        "limit": limit,
    }

    try:
        response = requests.post(f"{API_URL}/search", json=payload, headers=headers)
        response.raise_for_status()

        result = response.json()
        return result
    except Exception as e:
        st.error(f"Ошибка при поиске: {str(e)}")
        if hasattr(e, "response") and e.response is not None:
            st.error(f"Ответ сервера: {e.response.text}")
        return None


# Настройка страницы
st.set_page_config(
    page_title="Классификатор запросов",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Боковая панель для настроек API
with st.sidebar:
    st.title("Настройки API")
    api_url = st.text_input("URL API", value=API_URL)
    username = st.text_input("Имя пользователя", value=USERNAME)
    password = st.text_input("Пароль", value=PASSWORD, type="password")

    if st.button("Сохранить настройки"):
        API_URL = api_url
        USERNAME = username
        PASSWORD = password
        st.success("Настройки сохранены")

# Создаем вкладки
tab1, tab2, tab3 = st.tabs(["Классификация", "Похожие документы", "Загрузка данных"])

# Вкладка 1: Классификация
with tab1:
    st.title("Классификация запросов")

    # Выбор примера запроса
    st.subheader("Выбор запроса")

    use_default = st.checkbox("Использовать предустановленный запрос")

    if use_default:
        example_index = st.selectbox(
            "Выберите пример запроса:",
            options=range(len(DEFAULT_EXAMPLES)),
            format_func=lambda i: f"{DEFAULT_EXAMPLES[i]['id']} - {DEFAULT_EXAMPLES[i]['subject']}",
        )

        selected_example = DEFAULT_EXAMPLES[example_index]

        # Отображаем детали выбранного примера
        st.info(f"""
        **ID:** {selected_example["id"]}
        **Тема:** {selected_example["subject"]}
        **Описание:** {selected_example["description"]}
        **Класс:** {selected_example["class"]}
        **Задача:** {selected_example["task"]}
        """)

        # Предзаполняем поля формы
        default_subject = selected_example["subject"]
        default_description = selected_example["description"]
    else:
        default_subject = ""
        default_description = ""

    # Форма для ввода данных
    st.subheader("Данные для классификации")
    subject = st.text_input("Тема (subject):", value=default_subject)
    description = st.text_area(
        "Описание (description):", value=default_description, height=200
    )

    if st.button("Классифицировать"):
        if not subject and not description:
            st.warning("Пожалуйста, введите тему или описание")
        else:
            with st.spinner("Получение токена..."):
                token = get_token()

            if token:
                with st.spinner("Классификация запроса..."):
                    result = classify_request(subject, description, token)

                if result and "predictions" in result:
                    st.success("Запрос успешно классифицирован!")

                    # Показываем результаты в виде таблицы
                    predictions_df = pd.DataFrame(result["predictions"])
                    st.subheader("Результаты классификации:")
                    st.dataframe(predictions_df, width=800)

                    # Визуализация вероятностей
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.barplot(
                        x="class_name",
                        y="probability",
                        data=predictions_df.head(5),
                        ax=ax,
                    )
                    ax.set_xlabel("Класс")
                    ax.set_ylabel("Вероятность")
                    ax.set_title("Топ-5 классов по вероятности")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Показываем предсказанный класс
                    st.subheader(
                        f"Предсказанный класс: {result['predictions'][0]['class_name']}"
                    )
                    st.subheader(
                        f"Вероятность: {result['predictions'][0]['probability']:.2f}"
                    )

                else:
                    st.error("Не удалось получить результаты классификации")

# Вкладка 2: Похожие документы - ИСПРАВЛЕНО
with tab2:
    st.title("Поиск похожих документов")

    # Выбор примера запроса для поиска
    use_default_search = st.checkbox("Использовать предустановленный запрос для поиска")

    if use_default_search:
        example_index_search = st.selectbox(
            "Выберите пример запроса для поиска:",
            options=range(len(DEFAULT_EXAMPLES)),
            format_func=lambda i: f"{DEFAULT_EXAMPLES[i]['id']} - {DEFAULT_EXAMPLES[i]['subject']}",
            key="search_example",
        )

        selected_example_search = DEFAULT_EXAMPLES[example_index_search]
        default_search_subject = selected_example_search["subject"]
        default_search_description = selected_example_search["description"]
    else:
        default_search_subject = ""
        default_search_description = ""

    # Разделяем ввод на subject и description
    st.subheader("Данные для поиска")
    search_subject = st.text_input("Тема запроса:", value=default_search_subject)
    search_description = st.text_area(
        "Описание запроса:", value=default_search_description, height=150
    )

    limit = st.slider("Количество результатов", min_value=1, max_value=20, value=10)

    if st.button("Искать"):
        if not search_subject and not search_description:
            st.warning("Пожалуйста, введите тему или описание для поиска")
        else:
            with st.spinner("Получение токена..."):
                token = get_token()

            if token:
                with st.spinner("Поиск похожих документов..."):
                    search_results = search_similar(
                        search_subject, search_description, token, limit
                    )

                if search_results and "results" in search_results:
                    st.success(f"Найдено {len(search_results['results'])} документов")

                    for i, result in enumerate(search_results["results"]):
                        with st.expander(
                            f"{i + 1}. {result['subject']} (Класс: {result['class_name']}, Оценка: {result['score']:.4f})"
                        ):
                            st.write(f"**ID запроса:** {result['request_id']}")
                            st.write(f"**Тема:** {result['subject']}")
                            st.write(f"**Описание:** {result['description']}")
                            st.write(f"**Класс:** {result['class_name']}")
                            if "task" in result:
                                st.write(f"**Задача:** {result['task']}")
                            st.write(f"**Оценка сходства:** {result['score']:.4f}")

                    # Визуализация оценок сходства
                    if search_results["results"]:
                        scores = [
                            result["score"] for result in search_results["results"]
                        ]
                        titles = [
                            f"{i + 1}. {result['subject'][:30]}..."
                            for i, result in enumerate(search_results["results"])
                        ]

                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = ax.barh(range(len(scores)), scores, align="center")
                        ax.set_yticks(range(len(scores)))
                        ax.set_yticklabels(titles)
                        ax.set_xlabel("Оценка сходства")
                        ax.set_title("Топ документов по сходству")

                        # Добавляем значения к столбцам
                        for i, v in enumerate(scores):
                            ax.text(v + 0.01, i, f"{v:.4f}", va="center")

                        plt.tight_layout()
                        st.pyplot(fig)

                else:
                    st.warning("Не найдено похожих документов")

# Вкладка 3: Загрузка данных и оценка
with tab3:
    st.title("Загрузка данных и оценка качества")
    uploaded_file = st.file_uploader(
        "Выберите CSV-файл с колонками subject, description и class", type="csv"
    )
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Файл успешно загружен")
        st.subheader("Предварительный просмотр данных")
        st.dataframe(df.head())

        if st.button("Загрузить данные и рассчитать метрики"):
            token = get_token()
            if token:
                with st.spinner("Загрузка данных в систему..."):
                    upload_data(df, token)
                st.success("Данные загружены")

                # Разбиваем данные на обучающую и тестовую выборки
                train_df, test_df = train_test_split(
                    df, test_size=0.2, random_state=42, stratify=df["class"]
                )

                with st.spinner("Получение предсказаний для тестовой выборки..."):
                    preds = predict(test_df, token)

                # Вычисляем метрики
                y_true = test_df["class"].tolist()
                y_pred = preds
                acc = accuracy_score(y_true, y_pred)
                report_text = classification_report(y_true, y_pred)
                cm = confusion_matrix(y_true, y_pred)

                st.subheader("Результаты оценки")
                st.write(f"Accuracy: {acc:.4f}")
                st.text(report_text)

                # Визуализация матрицы ошибок
                fig, ax = plt.subplots(figsize=(6, 6))
                sns.heatmap(cm, annot=True, fmt="d", ax=ax)
                ax.set_xlabel("Предсказано")
                ax.set_ylabel("Истинный класс")
                st.pyplot(fig)

                # Возможность скачать последнюю загруженную версию данных
                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Скачать последнюю загруженную версию",
                    csv_data,
                    "uploaded_data.csv",
                    "text/csv",
                )
    else:
        st.info("CSV-файл не загружен. Загрузите файл, чтобы начать.")
