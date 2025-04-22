import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
import json
from datetime import datetime

from api_utils import get_token
from evaluate import upload_data, predict, accuracy_score, train_test_split, classification_report, confusion_matrix

def render_data_upload_tab(api_url, username, password):
    """Отображение вкладки загрузки данных и оценки качества"""
    st.title("Загрузка данных и оценка качества")
    
    # Создаем директории для сохранения файлов и метрик, если они не существуют
    data_dir = "uploaded_data"
    metrics_dir = "metrics"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    
    # Загрузка файла (CSV или Excel)
    uploaded_file = st.file_uploader(
        "Выберите файл с данными", type=["csv", "xlsx", "xls"]
    )
    
    if uploaded_file:
        # Определяем тип файла и загружаем соответствующим образом
        file_extension = uploaded_file.name.split(".")[-1]
        
        try:
            if file_extension.lower() == "csv":
                df = pd.read_csv(uploaded_file)
            else:  # Excel файл
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            st.success(f"Файл {uploaded_file.name} успешно загружен")
            
            # Сохраняем файл локально
            with open(os.path.join(data_dir, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.subheader("Предварительный просмотр данных")
            st.dataframe(df.head())
            
            # Выбор колонок для обработки
            st.subheader("Выбор колонок для обработки")
            
            # Получаем список всех колонок
            all_columns = df.columns.tolist()
            
            # Выбор колонок для subject и description
            col1, col2 = st.columns(2)
            
            with col1:
                subject_col = st.selectbox(
                    "Выберите колонку для поля subject:",
                    options=all_columns,
                    index=all_columns.index("subject") if "subject" in all_columns else 0
                )
            
            with col2:
                description_col = st.selectbox(
                    "Выберите колонку для поля description:",
                    options=all_columns,
                    index=all_columns.index("description") if "description" in all_columns else 0
                )
            
            # Выбор колонки с целевой переменной (class)
            target_col = st.selectbox(
                "Выберите колонку с целевой переменной (class):",
                options=all_columns,
                index=all_columns.index("class") if "class" in all_columns else 0
            )
            
            # Переименовываем колонки для соответствия требуемому формату
            df_processed = df.copy()
            
            # Создаем новый DataFrame только с нужными колонками
            df_processed = df_processed.rename(columns={
                subject_col: "subject",
                description_col: "description",
                target_col: "class"
            })[[subject_col, description_col, target_col]].rename(columns={
                subject_col: "subject",
                description_col: "description",
                target_col: "class"
            })
            
            st.subheader("Данные после обработки")
            st.dataframe(df_processed.head())
            
            # Опция очистки индекса перед загрузкой
            clear_index = st.checkbox("Очистить существующий индекс перед загрузкой", value=True)
            
            if st.button("Загрузить данные и рассчитать метрики"):
                token = get_token(api_url, username, password)
                if token:
                    # Разбиваем данные на обучающую и тестовую выборки
                    train_df, test_df = train_test_split(
                        df_processed, test_size=0.2, random_state=42, stratify=df_processed["class"]
                    )
                    
                    with st.spinner("Загрузка данных в систему..."):
                        # Если нужно очистить индекс перед загрузкой
                        if clear_index:
                            st.info("Очистка существующего индекса...")
                            # Здесь должен быть вызов API для очистки индекса
                            # clear_index_api(token)
                        
                        upload_data(train_df, token)
                    
                    st.success("Данные успешно загружены")
                    
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
                    
                    # Сохраняем метрики
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    metrics_filename = f"metrics_{timestamp}.json"
                    metrics_data = {
                        "filename": uploaded_file.name,
                        "timestamp": timestamp,
                        "accuracy": float(acc),
                        "classification_report": report_text,
                        "confusion_matrix": cm.tolist()
                    }
                    
                    with open(os.path.join(metrics_dir, metrics_filename), "w", encoding="utf-8") as f:
                        json.dump(metrics_data, f, ensure_ascii=False, indent=4)
                    
                    # Возможность скачать обработанные данные
                    csv_data = df_processed.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Скачать обработанные данные",
                        csv_data,
                        f"processed_{uploaded_file.name.split('.')[0]}.csv",
                        "text/csv",
                    )
                    
                    # Сохраняем фигуру с матрицей ошибок
                    fig_path = os.path.join(metrics_dir, f"confusion_matrix_{timestamp}.png")
                    fig.savefig(fig_path)
        
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {str(e)}")
    
    # Отображение последних метрик
    st.subheader("Последние метрики")
    
    # Проверяем, есть ли сохраненные метрики
    metrics_files = [f for f in os.listdir(metrics_dir) if f.endswith('.json')] if os.path.exists(metrics_dir) else []
    
    if metrics_files:
        # Сортируем файлы по времени создания (от новых к старым)
        metrics_files.sort(reverse=True)
        
        # Загружаем последний файл с метриками
        with open(os.path.join(metrics_dir, metrics_files[0]), "r", encoding="utf-8") as f:
            last_metrics = json.load(f)
        
        st.write(f"Файл: {last_metrics['filename']}")
        st.write(f"Дата загрузки: {last_metrics['timestamp']}")
        st.write(f"Accuracy: {last_metrics['accuracy']:.4f}")
        st.text(last_metrics['classification_report'])
        
        # Отображаем сохраненную матрицу ошибок, если она существует
        cm_file = os.path.join(metrics_dir, f"confusion_matrix_{last_metrics['timestamp']}.png")
        if os.path.exists(cm_file):
            st.image(cm_file, caption="Матрица ошибок")
    else:
        st.info("Нет сохраненных метрик. Загрузите файл и рассчитайте метрики.")
