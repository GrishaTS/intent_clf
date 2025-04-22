import matplotlib.pyplot as plt
import streamlit as st

from api_utils import get_token, search_similar
from config import DEFAULT_EXAMPLES

def render_similar_docs_tab(api_url, username, password):
    """Отображение вкладки поиска похожих документов"""
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
                token = get_token(api_url, username, password)

            if token:
                with st.spinner("Поиск похожих документов..."):
                    search_results = search_similar(
                        search_subject, search_description, token, api_url, limit
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
