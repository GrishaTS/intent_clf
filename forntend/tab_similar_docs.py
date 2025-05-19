import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

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
                    
                    # Отображение запроса пользователя
                    st.subheader("Ваш запрос")
                    query_data = {
                        "Тема": [search_subject],
                        "Описание": [search_description]
                    }
                    st.dataframe(pd.DataFrame(query_data), use_container_width=True)
                    
                    # Отображение топ-3 самых похожих документов
                    if len(search_results["results"]) > 0:
                        st.subheader("Топ-3 самых похожих документа")
                        top3_results = search_results["results"][:min(3, len(search_results["results"]))]
                        
                        top3_data = []
                        for result in top3_results:
                            top3_data.append({
                                "ID заявки": result['request_id'],
                                "Тема": result['subject'],
                                "Описание": result['description'],
                                "Класс": result['class_name'],
                                "Оценка сходства": f"{result['score']:.4f}"
                            })
                        
                        st.dataframe(pd.DataFrame(top3_data), use_container_width=True)
                    
                    # Отображение всех результатов в табличном формате
                    st.subheader("Все найденные документы")
                    
                    # Подготовка данных для таблицы
                    table_data = []
                    for i, result in enumerate(search_results["results"]):
                        table_data.append({
                            "ID заявки": result['request_id'],
                            "Тема": result['subject'],
                            "Описание": result['description'],
                            "Класс": result['class_name'],
                            "Оценка": f"{result['score']:.4f}"
                        })
                    
                    # Создаем DataFrame и отображаем таблицу
                    results_df = pd.DataFrame(table_data)
                    st.dataframe(results_df, use_container_width=True)

                    # Визуализация распределения классов
                    if search_results["results"]:
                        # Подсчет количества документов по классам
                        class_counts = Counter([result["class_name"] for result in search_results["results"]])
                        
                        # Сортировка по количеству
                        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
                        classes = [cls for cls, _ in sorted_classes]
                        counts = [count for _, count in sorted_classes]
                        
                        # Создание визуализации
                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = ax.bar(classes, counts, color='skyblue')
                        
                        # Добавление подписей
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                   f'{height}', ha='center', va='bottom')
                        
                        plt.title('Распределение документов по классам')
                        plt.xlabel('Класс')
                        plt.ylabel('Количество документов')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        
                        st.pyplot(fig)
                        
                        # Отображение процентного соотношения классов в виде круговой диаграммы
                        if len(class_counts) > 1:  # Если есть более одного класса
                            fig2, ax2 = plt.subplots(figsize=(8, 8))
                            wedges, texts, autotexts = ax2.pie(
                                counts, 
                                labels=classes, 
                                autopct='%1.1f%%',
                                textprops={'fontsize': 9}
                            )
                            plt.title('Процентное соотношение классов')
                            plt.tight_layout()
                            
                            st.pyplot(fig2)

                else:
                    st.warning("Не найдено похожих документов")
