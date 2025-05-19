import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
import json
from datetime import datetime

from api_utils import get_token, clear_index, upload_data, predict, accuracy_score, train_test_split, classification_report, confusion_matrix, filter_high_quality_classes

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
            elif file_extension.lower() == "xlsx":
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            elif file_extension.lower() == "xls":
                df = pd.read_excel(uploaded_file, engine='xlrd')
            else:
                st.error(f"Неподдерживаемый формат файла: {file_extension}")
                return
                
            st.success(f"Файл {uploaded_file.name} успешно загружен")

            
            # Сохраняем файл локально
            with open(os.path.join(data_dir, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.expander("Предварительный просмотр данных", expanded=False):
                st.dataframe(df)
            
            # Выбор колонок для обработки
            st.subheader("Выбор колонок для обработки")
            
            # Получаем список всех колонок
            all_columns = df.columns.tolist()
            
            # Выбор колонок для subject и description
            col0, col1, col2 = st.columns(3)
            
            with col0:
                # Выбор колонки с идентификатором заявки (ID)
                id_col = st.selectbox(
                    "Выберите колонку с ID заявки:",
                    options=all_columns,
                    index=all_columns.index("id") if "id" in all_columns else 0
                )
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
            df = df.dropna(subset=[target_col])
            
            # Расчет статистик по таргету
            with st.expander("Статистика по целевой переменной", expanded=False):
                # Получаем value_counts для таргета
                target_counts = df[target_col].value_counts()
                
                # Отображаем статистику
                st.write("Распределение значений целевой переменной:")
                
                # Создаем DataFrame для отображения
                target_stats_df = pd.DataFrame({
                    'Значение': target_counts.index,
                    'Количество': target_counts.values,
                    'Процент': (target_counts.values / target_counts.sum() * 100).round(2)
                })
                
                st.dataframe(target_stats_df)
                
                # Визуализация распределения целевой переменной
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Ограничиваем количество отображаемых категорий для лучшей визуализации
                top_n = min(20, len(target_counts))
                target_counts.head(top_n).plot(kind='bar', ax=ax)
                
                plt.title(f'Топ-{top_n} значений целевой переменной')
                plt.xlabel('Значение')
                plt.ylabel('Количество')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                st.pyplot(fig)
            
            # Опция для замены редких значений
            st.subheader("Обработка редких значений")
            
            # Опции для фильтрации классов
            filter_method = st.radio(
                "Выберите метод фильтрации классов:",
                ["По частоте встречаемости", "По качеству классификации"]
            )
            df_processed = df.copy()
                        # Создаем новый DataFrame только с нужными колонками
            df_processed = df_processed[[id_col, subject_col, description_col, target_col]].rename(columns={
                id_col: "id",
                subject_col: "subject",
                description_col: "description",
                target_col: "class"
            })

            if filter_method == "По частоте встречаемости":
                # Существующий код для замены редких значений                
                top_n_values = st.slider("Количество сохраняемых наиболее частых значений", 
                                        min_value=1, 
                                        max_value=min(50, len(target_counts)), 
                                        value=10)
                
                # Заменяем редкие значения на "Другое", если выбрана эта опция
                    # Получаем топ-N наиболее частых значений
                top_values = target_counts.head(top_n_values).index.tolist()
                
                # Заменяем все остальные значения на "Другое"
                df_processed["class"] = df_processed["class"].apply(
                    lambda x: x if x in top_values else "Другое"
                )
                
            else:  # По качеству классификации
                min_samples = st.slider("Минимальное количество образцов для сохранения класса", 
                                    min_value=1, 
                                    max_value=50, 
                                    value=10)
                
                min_f1_score = st.slider("Минимальное значение F1-score для сохранения класса", 
                                        min_value=0.0, 
                                        max_value=1.0, 
                                        value=0.5, 
                                        step=0.05)
                
                if st.checkbox("Установите галочку если выбрали параметры", value=False):
                    
                    df_high_quality, report_dict = filter_high_quality_classes(
                        df_processed, 
                        min_samples=min_samples, 
                        min_f1_score=min_f1_score,
                        api_url=api_url,
                        username=username,
                        password=password
                    )
                    
                    if df_high_quality is not None:
                        # Обновляем df_processed с отфильтрованными данными
                        df_processed = df_high_quality
            
            if filter_method:  
                # Показываем статистику после замены
                with st.expander("Статистика после замены редких значений", expanded=False):
                    new_counts = df_processed["class"].value_counts()
                    
                    # Создаем DataFrame для отображения
                    new_stats_df = pd.DataFrame({
                        'Значение': new_counts.index,
                        'Количество': new_counts.values,
                        'Процент': (new_counts.values / new_counts.sum() * 100).round(2)
                    })
                    
                    st.dataframe(new_stats_df)
                    
                    # Визуализация нового распределения
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    new_counts.plot(kind='bar', ax=ax2)
                    plt.title('Распределение значений после обработки')
                    plt.xlabel('Значение')
                    plt.ylabel('Количество')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    st.pyplot(fig2)
                
                with st.expander("Данные после обработки", expanded=False):
                    st.dataframe(df_processed)
            
            # Опция очистки индекса перед загрузкой
            clear_index_flag = st.checkbox("Очистить существующий индекс перед загрузкой", value=True)
            
            if st.button("Загрузить данные и рассчитать метрики"):
                token = get_token(api_url, username, password)
                if token:
                    # Разбиваем данные на обучающую и тестовую выборки
                    train_df, test_df = train_test_split(
                        df_processed, test_size=0.2, random_state=42, stratify=df_processed["class"]
                    )
                    
                    with st.spinner("Загрузка данных в систему..."):
                        # Если нужно очистить индекс перед загрузкой
                        if clear_index_flag:
                            st.info("Очистка существующего индекса...")
                            clear_index(token, api_url)
                        
                        upload_data(train_df, token, api_url)
                    
                    st.success("Данные успешно загружены")
                    
                    with st.spinner("Получение предсказаний для тестовой выборки..."):
                        preds = predict(test_df, token, api_url)
                    
                    # Вычисляем метрики
                    y_true = test_df["class"].tolist()
                    y_pred = preds
                    
                    # Удаляем None из предсказаний
                    valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]
                    y_true_valid = [y_true[i] for i in valid_indices]
                    y_pred_valid = [y_pred[i] for i in valid_indices]
                    
                    acc = accuracy_score(y_true_valid, y_pred_valid)
                    
                    # Получаем отчет о классификации в виде словаря для лучшего отображения
                    report_dict = classification_report(y_true_valid, y_pred_valid, output_dict=True)
                    report_text = classification_report(y_true_valid, y_pred_valid)
                    
                    cm = confusion_matrix(y_true_valid, y_pred_valid)
                    
                    # Получаем уникальные классы для отображения в матрице ошибок
                    unique_classes = sorted(list(set(y_true_valid + y_pred_valid)))
                    
                    st.subheader("Результаты оценки")
                    st.write(f"Accuracy: {acc:.4f}")
                    
                    # Отображаем отчет о классификации как DataFrame
                    with st.expander("Отчет о классификации", expanded=True):
                        # Преобразуем словарь отчета в DataFrame
                        report_df = pd.DataFrame(report_dict).transpose()
                        
                        # Удаляем ненужные столбцы, если они есть
                        columns_to_keep = ['precision', 'recall', 'f1-score', 'support']
                        report_df = report_df[[col for col in columns_to_keep if col in report_df.columns]]
                        
                        # Округляем числовые значения для лучшей читаемости
                        for col in ['precision', 'recall', 'f1-score']:
                            if col in report_df.columns:
                                report_df[col] = report_df[col].round(2)
                        
                        # Преобразуем support в целое число
                        if 'support' in report_df.columns:
                            report_df['support'] = report_df['support'].astype(int)
                        
                        st.dataframe(report_df)
                    
                    # Визуализация метрик классификации
                    with st.expander("Визуализация метрик по классам", expanded=False):
                        # Извлекаем данные для визуализации
                        classes = [k for k in report_dict.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
                        
                        # Создаем DataFrame для визуализации
                        metrics_data = {
                            'Class': [],
                            'Metric': [],
                            'Value': []
                        }
                        
                        for cls in classes:
                            for metric in ['precision', 'recall', 'f1-score']:
                                metrics_data['Class'].append(cls)
                                metrics_data['Metric'].append(metric)
                                metrics_data['Value'].append(report_dict[cls][metric])
                        
                        metrics_df = pd.DataFrame(metrics_data)
                        
                        # Создаем визуализацию
                        fig_metrics, ax_metrics = plt.subplots(figsize=(12, 6))
                        
                        # Используем разные цвета для разных метрик
                        palette = {
                            'precision': 'blue',
                            'recall': 'green',
                            'f1-score': 'red'
                        }
                        
                        # Создаем сгруппированную столбчатую диаграмму
                        sns.barplot(
                            x='Class',
                            y='Value',
                            hue='Metric',
                            data=metrics_df,
                            palette=palette,
                            ax=ax_metrics
                        )
                        
                        plt.title('Метрики классификации по классам')
                        plt.ylabel('Значение')
                        plt.xlabel('Класс')
                        plt.xticks(rotation=45, ha='right')
                        plt.legend(title='Метрика')
                        plt.tight_layout()
                        
                        st.pyplot(fig_metrics)
                    
                    # Визуализация матрицы ошибок с именами классов
                    with st.expander("Матрица ошибок", expanded=False):
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(
                            cm, 
                            annot=True, 
                            fmt="d", 
                            cmap="Blues",
                            xticklabels=unique_classes,
                            yticklabels=unique_classes
                        )
                        ax.set_xlabel("Предсказанный класс")
                        ax.set_ylabel("Истинный класс")
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Загрузка тестовой выборки после оценки метрик
                    with st.expander("Загрузка тестовой выборки", expanded=False):
                        
                        upload_test_data = st.checkbox("Загрузить тестовую выборку в базу данных", value=True)
                        
                        if upload_test_data:
                            with st.spinner("Загрузка тестовой выборки в систему..."):
                                test_upload_result = upload_data(test_df, token, api_url)
                                if test_upload_result:
                                    st.success(f"Тестовая выборка успешно загружена ({len(test_df)} записей)")
                                else:
                                    st.error("Ошибка при загрузке тестовой выборки")
                    
                    # Сохраняем метрики
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    metrics_filename = f"metrics_{timestamp}.json"
                    metrics_data = {
                        "filename": uploaded_file.name,
                        "timestamp": timestamp,
                        "accuracy": float(acc),
                        "classification_report_text": report_text,
                        "classification_report_dict": report_dict,
                        "confusion_matrix": cm.tolist(),
                        "classes": unique_classes
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
                    
                    # Сохраняем фигуры
                    fig_path = os.path.join(metrics_dir, f"confusion_matrix_{timestamp}.png")
                    fig.savefig(fig_path)
                    
                    fig_metrics_path = os.path.join(metrics_dir, f"classification_metrics_{timestamp}.png")
                    fig_metrics.savefig(fig_metrics_path)
        
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {str(e)}")
    
    # Отображение последних метрик
    with st.expander("Последние метрики", expanded=True):
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
            
            # Отображаем отчет о классификации как DataFrame, если он доступен
            if 'classification_report_dict' in last_metrics:
                st.subheader("Отчет о классификации")
                
                # Преобразуем словарь отчета в DataFrame
                report_df = pd.DataFrame(last_metrics['classification_report_dict']).transpose()
                
                # Удаляем ненужные столбцы, если они есть
                columns_to_keep = ['precision', 'recall', 'f1-score', 'support']
                report_df = report_df[[col for col in columns_to_keep if col in report_df.columns]]
                
                # Округляем числовые значения для лучшей читаемости
                for col in ['precision', 'recall', 'f1-score']:
                    if col in report_df.columns:
                        report_df[col] = report_df[col].round(2)
                
                # Преобразуем support в целое число
                if 'support' in report_df.columns:
                    report_df['support'] = report_df['support'].astype(int)
                
                st.dataframe(report_df)
            else:
                st.text(last_metrics['classification_report_text'])
            
            # Отображаем сохраненную матрицу ошибок, если она существует
            cm_file = os.path.join(metrics_dir, f"confusion_matrix_{last_metrics['timestamp']}.png")
            if os.path.exists(cm_file):
                st.image(cm_file, caption="Матрица ошибок")
                
            # Отображаем сохраненную визуализацию метрик, если она существует
            metrics_file = os.path.join(metrics_dir, f"classification_metrics_{last_metrics['timestamp']}.png")
            if os.path.exists(metrics_file):
                st.image(metrics_file, caption="Метрики классификации по классам")
        else:
            st.info("Нет сохраненных метрик. Загрузите файл и рассчитайте метрики.")
