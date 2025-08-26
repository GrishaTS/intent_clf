# tab_data_upload.py
import os
import json
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from api_utils import (
    get_token,
    clear_index,
    upload_data,
    predict,
    accuracy_score,
    train_test_split,
    classification_report,
    confusion_matrix,
    filter_high_quality_classes,
)
from i18n import get_translations

def render_data_upload_tab(api_url, username, password):
    """Экран загрузки данных и оценки качества (i18n)."""
    # Текущий язык из session_state
    lang = st.session_state.get("lang", "ru")
    TR = get_translations(lang)
    _ = lambda k: TR[k]  # короткий алиас

    st.title(_("data_upload.title"))

    # Директории
    data_dir = "uploaded_data"
    metrics_dir = "metrics"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Загрузка файла
    uploaded_file = st.file_uploader(
        _("data_upload.file_uploader"),
        type=["csv", "xlsx", "xls"],
    )

    if uploaded_file:
        file_extension = uploaded_file.name.split(".")[-1].lower()

        try:
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file)
            elif file_extension in ("xlsx", "xls"):
                # engine подбирается автоматически, но явно укажем предпочтение
                engine = "openpyxl" if file_extension == "xlsx" else "xlrd"
                df = pd.read_excel(uploaded_file, engine=engine)
            else:
                st.error(_("data_upload.unsupported_format") + f": {file_extension}")
                return

            st.success(_("data_upload.file_loaded").format(filename=uploaded_file.name))

            # Сохраняем исходный файл локально
            with open(os.path.join(data_dir, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.expander(_("data_upload.preview"), expanded=False):
                st.dataframe(df)

            # --- Выбор колонок ---
            st.subheader(_("data_upload.columns.title"))
            all_columns = df.columns.tolist()
            
            NONE_LABEL = _("data_upload.columns.none")
            col0, col1, col2 = st.columns(3)
            with col0:
                id_col = st.selectbox(
                    _("data_upload.columns.id"),
                    options=all_columns,
                    index=all_columns.index("id") if "id" in all_columns else 0,
                )
            with col1:
                subject_options = [NONE_LABEL] + all_columns
                subj_default_idx = (
                    subject_options.index("subject") if "subject" in all_columns else 0
                )
                subject_choice = st.selectbox(
                    _("data_upload.columns.subject_opt"),
                    options=subject_options,
                    index=subj_default_idx,
                )
                subject_col = None if subject_choice == NONE_LABEL else subject_choice
            with col2:
                description_col = st.selectbox(
                    _("data_upload.columns.description"),
                    options=all_columns,
                    index=all_columns.index("description") if "description" in all_columns else 0,
                )

            target_col = st.selectbox(
                _("data_upload.columns.target"),
                options=all_columns,
                index=all_columns.index("class") if "class" in all_columns else 0,
            )

            # Выкидываем пустые таргеты
            df = df.dropna(subset=[target_col])

            # --- Статистика по таргету ---
            with st.expander(_("data_upload.target_stats.title"), expanded=False):
                target_counts = df[target_col].value_counts()
                st.write(_("data_upload.target_stats.caption"))

                target_stats_df = pd.DataFrame({
                    _("data_upload.common.value"): target_counts.index,
                    _("data_upload.common.count"): target_counts.values,
                    _("data_upload.common.percent"): (target_counts.values / target_counts.sum() * 100).round(2),
                })
                st.dataframe(target_stats_df)

                fig, ax = plt.subplots(figsize=(10, 6))
                top_n = min(20, len(target_counts))
                target_counts.head(top_n).plot(kind="bar", ax=ax)
                ax.set_title(_("data_upload.target_stats.chart_title").format(top_n=top_n))
                ax.set_xlabel(_("data_upload.common.value"))
                ax.set_ylabel(_("data_upload.common.count"))
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)

            # --- Обработка редких значений / фильтрация классов ---
            st.subheader(_("data_upload.filter.title"))

            filter_method = st.radio(
                _("data_upload.filter.method_label"),
                [_("data_upload.filter.by_freq"), _("data_upload.filter.by_quality")],
            )

            # Нормализуем df в требуемые колонки
            base_cols = [id_col, description_col, target_col]
            df_processed = df[base_cols].rename(
                columns={id_col: "id", description_col: "description", target_col: "class"}
            )

            if subject_col:
                df_processed["subject"] = df[subject_col]
            else:
                df_processed["subject"] = "no_subject"  # значение по умолчанию

            df_processed = df_processed[["id", "subject", "description", "class"]]

            if filter_method == _("data_upload.filter.by_freq"):
                top_n_values = st.slider(
                    _("data_upload.filter.by_freq_slider"),
                    min_value=1,
                    max_value=min(100, df_processed["class"].nunique()),
                    value=10,
                )
                top_values = df_processed["class"].value_counts().head(top_n_values).index.tolist()
                df_processed["class"] = df_processed["class"].apply(
                    lambda x: x if x in top_values else _("data_upload.common.others")
                )

            else:
                min_samples = st.slider(
                    _("data_upload.filter.min_samples"),
                    min_value=1,
                    max_value=100,
                    value=10,
                )

                min_f1_score = st.slider(
                    _("data_upload.filter.min_f1"),
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                )

                if st.checkbox(_("data_upload.filter.confirm_params"), value=False):
                    df_high_quality, report_dict = filter_high_quality_classes(
                        df_processed,
                        min_samples=min_samples,
                        min_f1_score=min_f1_score,
                        api_url=api_url,
                        username=username,
                        password=password,
                    )
                    if df_high_quality is not None:
                        df_processed = df_high_quality

            # --- Статистика после обработки ---
            with st.expander(_("data_upload.after_filter_stats.title"), expanded=False):
                new_counts = df_processed["class"].value_counts()
                new_stats_df = pd.DataFrame({
                    _("data_upload.common.value"): new_counts.index,
                    _("data_upload.common.count"): new_counts.values,
                    _("data_upload.common.percent"): (new_counts.values / new_counts.sum() * 100).round(2),
                })
                st.dataframe(new_stats_df)

                fig2, ax2 = plt.subplots(figsize=(10, 6))
                new_counts.plot(kind="bar", ax=ax2)
                ax2.set_title(_("data_upload.after_filter_stats.chart_title"))
                ax2.set_xlabel(_("data_upload.common.value"))
                ax2.set_ylabel(_("data_upload.common.count"))
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig2)

            with st.expander(_("data_upload.after_filter_data.title"), expanded=False):
                st.dataframe(df_processed)

            # --- Очистка индекса ---
            clear_index_flag = st.checkbox(
                _("data_upload.clear_index_checkbox"),
                value=True,
            )

            # --- Запуск вычислений ---
            if st.button(_("data_upload.run_button")):
                token = get_token(api_url, username, password)
                if token:
                    train_df, test_df = train_test_split(
                        df_processed, test_size=0.1, random_state=42, stratify=df_processed["class"]
                    )

                    with st.spinner(_("data_upload.spinner.upload")):
                        if clear_index_flag:
                            st.info(_("data_upload.info.clearing"))
                            clear_index(token, api_url)
                        upload_data(train_df, token, api_url)

                    st.success(_("data_upload.success.upload_done"))

                    with st.spinner(_("data_upload.spinner.predict")):
                        preds = predict(test_df, token, api_url)

                    # Метрики
                    y_true = test_df["class"].tolist()
                    y_pred = preds

                    valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]
                    y_true_valid = [y_true[i] for i in valid_indices]
                    y_pred_valid = [y_pred[i] for i in valid_indices]

                    acc = accuracy_score(y_true_valid, y_pred_valid)
                    report_dict = classification_report(y_true_valid, y_pred_valid, output_dict=True)
                    report_text = classification_report(y_true_valid, y_pred_valid)
                    cm = confusion_matrix(y_true_valid, y_pred_valid)
                    unique_classes = sorted(list(set(y_true_valid + y_pred_valid)))

                    st.subheader(_("data_upload.metrics.title"))
                    st.write(_("data_upload.metrics.accuracy").format(acc=acc))

                    # Табличный отчёт
                    with st.expander(_("data_upload.metrics.report_table"), expanded=True):
                        report_df = pd.DataFrame(report_dict).transpose()
                        columns_to_keep = ["precision", "recall", "f1-score", "support"]
                        report_df = report_df[[c for c in columns_to_keep if c in report_df.columns]]
                        for col in ["precision", "recall", "f1-score"]:
                            if col in report_df.columns:
                                report_df[col] = report_df[col].round(2)
                        if "support" in report_df.columns:
                            report_df["support"] = report_df["support"].astype(int)
                        st.dataframe(report_df)

                    # Визуализация метрик
                    with st.expander(_("data_upload.metrics.by_class_plot"), expanded=False):
                        classes = [k for k in report_dict.keys() if k not in ["accuracy", "macro avg", "weighted avg"]]
                        metrics_data = {"Class": [], "Metric": [], "Value": []}
                        for cls in classes:
                            for metric in ["precision", "recall", "f1-score"]:
                                metrics_data["Class"].append(cls)
                                metrics_data["Metric"].append(metric)
                                metrics_data["Value"].append(report_dict[cls][metric])
                        metrics_df = pd.DataFrame(metrics_data)

                        fig_metrics, ax_metrics = plt.subplots(figsize=(12, 6))
                        palette = {"precision": "blue", "recall": "green", "f1-score": "red"}
                        sns.barplot(
                            x="Class",
                            y="Value",
                            hue="Metric",
                            data=metrics_df,
                            palette=palette,
                            ax=ax_metrics,
                        )
                        ax_metrics.set_title(_("data_upload.metrics.by_class_plot_title"))
                        ax_metrics.set_ylabel(_("data_upload.common.value"))
                        ax_metrics.set_xlabel(_("data_upload.metrics.class_label"))
                        plt.xticks(rotation=45, ha="right")
                        plt.legend(title=_("data_upload.metrics.metric_label"))
                        plt.tight_layout()
                        st.pyplot(fig_metrics)

                    # Матрица ошибок
                    with st.expander(_("data_upload.metrics.cm_title"), expanded=False):
                        fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
                        sns.heatmap(
                            cm,
                            annot=True,
                            fmt="d",
                            cmap="Blues",
                            xticklabels=unique_classes,
                            yticklabels=unique_classes,
                            ax=ax_cm,
                        )
                        ax_cm.set_xlabel(_("data_upload.metrics.predicted"))
                        ax_cm.set_ylabel(_("data_upload.metrics.true"))
                        plt.xticks(rotation=45, ha="right")
                        plt.tight_layout()
                        st.pyplot(fig_cm)

                    # Загрузка тестовой выборки
                    with st.expander(_("data_upload.upload_test.title"), expanded=False):
                        upload_test_data = st.checkbox(_("data_upload.upload_test.checkbox"), value=True)
                        if upload_test_data:
                            with st.spinner(_("data_upload.upload_test.spinner")):
                                test_upload_result = upload_data(test_df, token, api_url)
                                if test_upload_result:
                                    st.success(_("data_upload.upload_test.success").format(n=len(test_df)))
                                else:
                                    st.error(_("data_upload.upload_test.error"))

                    # Сохранение метрик
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    metrics_filename = f"metrics_{timestamp}.json"
                    metrics_data = {
                        "filename": uploaded_file.name,
                        "timestamp": timestamp,
                        "accuracy": float(acc),
                        "classification_report_text": report_text,
                        "classification_report_dict": report_dict,
                        "confusion_matrix": cm.tolist(),
                        "classes": unique_classes,
                    }
                    with open(os.path.join(metrics_dir, metrics_filename), "w", encoding="utf-8") as f:
                        json.dump(metrics_data, f, ensure_ascii=False, indent=4)

                    # Кнопка скачать обработанные данные
                    csv_data = df_processed.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        _("data_upload.download.button"),
                        csv_data,
                        f"processed_{uploaded_file.name.split('.')[0]}.csv",
                        "text/csv",
                    )

                    # Сохранение картинок
                    fig_cm_path = os.path.join(metrics_dir, f"confusion_matrix_{timestamp}.png")
                    fig_cm.savefig(fig_cm_path)

                    fig_metrics_path = os.path.join(metrics_dir, f"classification_metrics_{timestamp}.png")
                    fig_metrics.savefig(fig_metrics_path)

        except Exception as e:
            st.error(_("data_upload.error.processing").format(error=str(e)))

    # --- Последние метрики ---
    with st.expander(_("data_upload.last_metrics.title"), expanded=True):
        metrics_files = [f for f in os.listdir(metrics_dir) if f.endswith(".json")] if os.path.exists(metrics_dir) else []
        if metrics_files:
            metrics_files.sort(reverse=True)
            with open(os.path.join(metrics_dir, metrics_files[0]), "r", encoding="utf-8") as f:
                last_metrics = json.load(f)

            st.write(_("data_upload.last_metrics.file").format(filename=last_metrics["filename"]))
            st.write(_("data_upload.last_metrics.date").format(ts=last_metrics["timestamp"]))
            st.write(_("data_upload.metrics.accuracy").format(acc=last_metrics["accuracy"]))

            if "classification_report_dict" in last_metrics:
                st.subheader(_("data_upload.metrics.report_table"))
                report_df = pd.DataFrame(last_metrics["classification_report_dict"]).transpose()
                columns_to_keep = ["precision", "recall", "f1-score", "support"]
                report_df = report_df[[c for c in columns_to_keep if c in report_df.columns]]
                for col in ["precision", "recall", "f1-score"]:
                    if col in report_df.columns:
                        report_df[col] = report_df[col].round(2)
                if "support" in report_df.columns:
                    report_df["support"] = report_df["support"].astype(int)
                st.dataframe(report_df)
            else:
                st.text(last_metrics.get("classification_report_text", ""))

            cm_file = os.path.join(metrics_dir, f"confusion_matrix_{last_metrics['timestamp']}.png")
            if os.path.exists(cm_file):
                st.image(cm_file, caption=_("data_upload.last_metrics.cm_caption"))

            metrics_img = os.path.join(metrics_dir, f"classification_metrics_{last_metrics['timestamp']}.png")
            if os.path.exists(metrics_img):
                st.image(metrics_img, caption=_("data_upload.last_metrics.metrics_caption"))
        else:
            st.info(_("data_upload.last_metrics.none"))