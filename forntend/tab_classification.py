import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from api_utils import get_token, classify_request
from config import DEFAULT_EXAMPLES

def render_classification_tab(api_url, username, password):
    """Отображение вкладки классификации"""
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
                token = get_token(api_url, username, password)

            if token:
                with st.spinner("Классификация запроса..."):
                    result = classify_request(subject, description, token, api_url)

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
